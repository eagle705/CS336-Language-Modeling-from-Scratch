"""
Mixture of Experts (MoE)
=========================
FFN을 여러 개의 "expert"로 나누고, router가 토큰별로 expert를 선택.

핵심 아이디어:
  - 모델 파라미터는 크지만, 각 토큰은 일부 expert만 활성화 → 연산량 제어
  - 예: 8 experts 중 top-2만 사용 → params 8x, FLOPs ~2x

구조:
  Input tokens
      │
      ▼
  Router: softmax(x @ W_gate) → token별 top-k expert와 gate score 선택
      │
      ▼
  Dispatch: token을 expert별 bucket으로 permute/pack하고 필요하면 전송
      │
      ▼
  Expert compute: 각 expert FFN이 자기 token bucket만 계산
      │
      ▼
  Combine: expert output을 reverse transfer 후 unpermute/scatter-add
      │
      ▼
  Output tokens

인터뷰 포인트:
  1. Router의 load balancing (expert 골고루 사용하게)
  2. Expert parallelism (expert를 다른 GPU에 배치)
  3. Dispatch/Combine 통신: all-to-all 또는 DeepEP 같은 MoE 전용 통신 kernel
  4. MoE Parallel Folding: attention과 expert MLP에 서로 다른 parallelism mapping 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ============================================================
# Part 1: Router (Gate)
# ============================================================

class TopKRouter(nn.Module):
    """
    각 토큰에 대해 top-k expert를 선택.

    router_logits = x @ W_gate          # (batch*seq, num_experts)
    router_probs  = softmax(router_logits)
    top_k_probs, top_k_indices = topk(router_probs, k)

    Load Balancing Loss:
      expert별 처리량이 균등하도록 auxiliary loss 추가.
      없으면 일부 expert에 토큰이 몰리는 "winner-take-all" 문제 발생.

      aux_loss = num_experts * sum_i(f_i * p_i)
        f_i = (expert i에 배정된 routing slot 비율)
        p_i = (expert i의 평균 gate probability)
      → 균등 분배면 aux_loss = 1, 쏠리면 > 1

      top-k에서는 token 하나가 k개의 expert에 들어가므로 routing slot은 tokens * k개다.
      예를 들어 top-2면 한 token이 expert A와 B에 동시에 dispatch된다.
      그래서 f_i를 "token 비율"이 아니라 "top-k slot 비율"로 정규화해야
      균등 분배 시 aux_loss가 1 근처가 된다.

    Router z-loss:
      logits가 너무 커지면 softmax가 매우 sharp해지고 routing이 불안정해질 수 있다.
      z_loss = mean(logsumexp(logits)^2)를 작은 계수로 더해 router logit scale을 누른다.
      실제 대형 MoE에서는 load-balancing loss, router z-loss, capacity/drop 정책 등을
      같이 써서 expert collapse와 overflow를 줄인다.

    Sequence auxiliary loss:
      global aux loss는 batch 전체 평균만 본다.
      그런데 긴 batch 안에서 일부 sequence는 expert 0에 몰리고, 다른 sequence는 expert 1에 몰려도
      batch 전체로는 균등해 보일 수 있다.
      seq_aux_loss는 sequence별로 f_i, p_i를 계산한 뒤 평균내서 각 sequence 내부에서도
      expert 사용이 너무 쏠리지 않게 만든다.

    DeepSeek-V3 style auxiliary-loss-free balancing:
      DeepSeek-V3는 main MoE balancing에 auxiliary loss gradient를 강하게 넣으면
      language modeling objective와 간섭할 수 있다고 보고, expert별 routing bias를 쓴다.
      최근 load가 높은 expert의 bias는 낮추고, load가 낮은 expert의 bias는 올려서
      다음 routing top-k 선택 빈도를 통계적으로 조절한다.
      중요한 차이:
        - aux loss: loss term이므로 router weight에 gradient가 직접 흐른다.
        - aux-loss-free bias: routing score에 더하는 non-gradient control signal에 가깝다.
      보통 top-k 선택에는 bias가 들어간 score를 쓰고, combine gate weight는 원래 probability를
      gather해서 정규화한다. 그래야 "선택 빈도 조절"과 "출력 가중치" 역할이 분리된다.
    """

    def __init__(
        self,
        embed_dim,
        num_experts,
        top_k=2,
        z_loss_weight=1e-3,
        seq_aux_loss_weight=0.0,
        use_aux_loss_free_bias=False,
        bias_update_rate=1e-3,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.z_loss_weight = z_loss_weight
        self.seq_aux_loss_weight = seq_aux_loss_weight
        self.use_aux_loss_free_bias = use_aux_loss_free_bias
        self.bias_update_rate = bias_update_rate
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.last_aux_loss_breakdown = {}

    def forward(self, x, sequence_ids=None):
        # x: (batch * seq, embed_dim)
        logits = self.gate(x)                         # (batch*seq, num_experts)
        probs = F.softmax(logits, dim=-1)             # (batch*seq, num_experts)

        # DeepSeek-V3식 auxiliary-loss-free balancing의 핵심 아이디어:
        # top-k "선택"에는 expert_bias를 더한 routing_scores를 쓴다.
        # expert_bias는 학습되는 parameter가 아니라 최근 expert load 통계를 보고 외부에서 업데이트되는 값이다.
        # 많이 선택된 expert는 bias를 낮추고, 적게 선택된 expert는 bias를 올리면
        # 다음 step에서 자연스럽게 덜/더 선택된다.
        routing_scores = probs + self.expert_bias if self.use_aux_loss_free_bias else probs

        # top-k 선택
        _, top_k_indices = torch.topk(routing_scores, self.top_k, dim=-1)
        # top_k_indices: (batch*seq, top_k)  → 선택된 expert의 인덱스
        # top_k_probs는 routing_scores가 아니라 원래 probs에서 gather한다.
        # bias는 "어떤 expert를 고를지"를 조절하고, 실제 combine 가중치는 router probability를 쓴다.
        top_k_probs = probs.gather(dim=-1, index=top_k_indices)
        # top_k_probs:   (batch*seq, top_k)  → 선택된 expert의 gate 점수
        # top_k_indices: (batch*seq, top_k)  → 선택된 expert의 인덱스

        # gate 정규화: 선택된 expert들의 확률 합이 1이 되도록
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Load balancing loss 계산
        # f_i: expert i에 배정된 routing slot 비율.
        # top-2라면 전체 routing slot 수는 tokens * 2개다.
        # expert_mask.mean(dim=0)는 token당 평균 선택 횟수라 합이 top_k가 되므로,
        # 다시 top_k로 나누면 expert별 slot fraction이 된다.
        expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1)  # (tokens, experts)
        f = expert_mask.float().mean(dim=0) / self.top_k  # (num_experts,)

        # p_i: expert i의 평균 gate probability
        p = probs.mean(dim=0)  # (num_experts,)

        # load_balance_loss: f와 p의 내적 → 균등하면 1 근처, 쏠리면 커진다.
        # 예: num_experts=4이고 완전히 균등하면
        #   f = [0.25, 0.25, 0.25, 0.25], p = [0.25, 0.25, 0.25, 0.25]
        #   4 * sum(f * p) = 4 * (4 * 0.25 * 0.25) = 1.0
        # 반대로 expert 0에 routing slot과 probability가 몰리면
        #   f = [1.0, 0, 0, 0], p = [1.0, 0, 0, 0]
        #   4 * sum(f * p) = 4.0
        # 즉 많이 선택되는 expert의 p도 같이 커질수록 loss가 커져서 쏠림을 벌준다.
        load_balance_loss = self.num_experts * (f * p).sum()

        # seq_aux_loss:
        # sequence_ids가 있으면 sequence별로 같은 loss를 계산한다.
        # 예를 들어 B=2, S=1024일 때 batch 전체로는 expert 사용량이 균등해 보여도,
        # seq0은 expert0/1에 몰리고 seq1은 expert2/3에 몰리면 dispatch imbalance가 생길 수 있다.
        # seq_aux_loss는 각 sequence 내부의 분포를 보므로 이런 local imbalance를 더 잘 잡는다.
        seq_aux_loss = logits.new_tensor(0.0)
        if sequence_ids is not None:
            num_sequences = int(sequence_ids.max().item()) + 1
            seq_mask = F.one_hot(sequence_ids, num_sequences).to(probs.dtype)  # (tokens, sequences)
            seq_counts = seq_mask.sum(dim=0).clamp_min(1.0)  # (sequences,)
            # (sequences, experts): sequence별 routing slot fraction
            f_seq = seq_mask.T @ expert_mask.to(probs.dtype)
            f_seq = f_seq / (seq_counts.unsqueeze(-1) * self.top_k)
            # (sequences, experts): sequence별 평균 router probability
            p_seq = seq_mask.T @ probs
            p_seq = p_seq / seq_counts.unsqueeze(-1)
            seq_aux_loss = self.num_experts * (f_seq * p_seq).sum(dim=-1).mean()

        # router z-loss: router logit scale을 너무 키우지 않도록 약하게 제약.
        router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        aux_loss = (
            load_balance_loss
            + self.seq_aux_loss_weight * seq_aux_loss
            + self.z_loss_weight * router_z_loss
        )
        self.last_aux_loss_breakdown = {
            "load_balance_loss": load_balance_loss.detach(),
            "seq_aux_loss": seq_aux_loss.detach(),
            "seq_aux_loss_weight": self.seq_aux_loss_weight,
            "router_z_loss": router_z_loss.detach(),
            "z_loss_weight": self.z_loss_weight,
            "expert_bias": self.expert_bias.detach().clone(),
        }

        return top_k_probs, top_k_indices, aux_loss

    @torch.no_grad()
    def update_aux_loss_free_bias(
        self,
        expert_load,
        group=None,
        bias_clip=1.0,
        allowed_expert_mask=None,
    ):
        """
        DeepSeek-V3 style auxiliary-loss-free load balancing을 단순화한 예시.

        expert_load: (num_experts,) 최근 step/window에서 expert별 routing slot 수 또는 비율.
        group: EP process group. 분산 학습이면 모든 EP rank의 load를 all-reduce로 합친다.
        bias_clip: bias가 너무 커져 routing을 압도하지 않도록 [-bias_clip, +bias_clip]으로 clamp.
        allowed_expert_mask:
          node-limited routing 예시용 mask. 1이면 target에 포함, 0이면 이번 routing 후보에서 제외.
          예를 들어 특정 node 안 expert만 우선 routing하려면 그 expert들만 1로 둔다.

        예: num_experts=4, top_k=2, 최근 window에 token 100개가 있었다면
          전체 routing slot 수 = 100 * 2 = 200
          균등한 load라면 expert_load = [50, 50, 50, 50]
          expert 0에 몰렸다면 expert_load = [120, 30, 25, 25] 같은 식이 된다.

        위 값을 비율로 바꾸면:
          load   = [0.60, 0.15, 0.125, 0.125]
          target = [0.25, 0.25, 0.25, 0.25]
          target - load = [-0.35, +0.10, +0.125, +0.125]

        따라서 expert 0의 bias는 내려가고, 덜 쓰인 expert 1/2/3의 bias는 올라간다.
        다음 top-k routing에서 expert 0은 조금 덜 뽑히고, expert 1/2/3은 조금 더 뽑히게 된다.

        평균보다 많이 선택된 expert:
          bias를 낮춰 다음 top-k에서 덜 선택되게 한다.
        평균보다 적게 선택된 expert:
          bias를 올려 다음 top-k에서 더 선택되게 한다.

        실제 구현은 global EP group의 load 통계를 all-reduce로 모으고,
        bias update rate, clipping, node-limited routing 같은 정책을 함께 쓴다.
        """
        load = expert_load.float()

        # 분산 EP에서 "dispatcher가 routing한다"는 말은 보통 각 rank가 자기 local token에 대해
        # router/dispatcher를 실행한다는 뜻이다. 중앙 rank 하나가 모든 DP/EP token의 routing을
        # 전부 알고 결정하는 구조가 아니다.
        #
        # 따라서 이 시점의 expert_load는 보통 local microbatch에서 나온 local expert load다.
        # 하지만 bias update는 "expert 0이 EP group 전체에서 과하게 선택됐는가?"를 봐야 하므로
        # EP group 전체의 load를 sum all-reduce로 합친 뒤 global load 기준으로 업데이트한다.
        #
        # 예:
        #   rank0 local load = [20, 5, 0, 0]
        #   rank1 local load = [0, 0, 40, 10]
        #   all-reduce 후 global load = [20, 5, 40, 10]
        # 이 global load를 보고 많이 쓰인 expert의 bias를 낮추고 덜 쓰인 expert의 bias를 올린다.
        if _dist_ready():
            dist.all_reduce(load, op=dist.ReduceOp.SUM, group=group)

        load = load / load.sum().clamp_min(1.0)

        if allowed_expert_mask is None:
            # 기본 target: 모든 expert가 같은 routing slot 비율을 받는 것.
            # num_experts=4라면 [0.25, 0.25, 0.25, 0.25].
            target = torch.full_like(load, 1.0 / self.num_experts)
        else:
            # node-limited routing 예시:
            # allowed_expert_mask=[1, 1, 0, 0]이면 이번 routing target은
            # [0.5, 0.5, 0.0, 0.0]이 된다. 즉 허용된 expert 안에서만 균등화한다.
            mask = allowed_expert_mask.to(device=load.device, dtype=load.dtype)
            target = mask / mask.sum().clamp_min(1.0)

        self.expert_bias += self.bias_update_rate * (target - load)

        # clipping: bias가 계속 누적되어 router probability보다 훨씬 커지면
        # 원래 router score를 무시하고 load-balancing controller가 routing을 지배할 수 있다.
        # 그래서 보통 작은 update rate와 함께 bias 범위를 제한한다.
        if bias_clip is not None:
            self.expert_bias.clamp_(min=-bias_clip, max=bias_clip)


# ============================================================
# Part 2: Expert Layer
# ============================================================

class Expert(nn.Module):
    """단일 expert = 일반 FFN."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================
# Part 3: MoE Layer (full)
# ============================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    동작:
    1. Router가 각 토큰의 top-k expert 선택
    2. Dispatch: expert별 token bucket을 만들기 위해 permute/pack하고 필요하면 전송한다
    3. Compute: 각 expert가 자기 bucket만 FFN 계산
    4. Combine: output을 reverse transfer 후 unpermute/scatter-add하고 gate score로 가중합

    여기서는 single process라 dispatch/combine을 mask와 index_select로 시뮬레이션한다.
    Expert Parallelism에서는 dispatch/combine 단계가 실제 all-to-all 통신이 된다.
    """

    def __init__(self, embed_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.router = TopKRouter(
            embed_dim,
            num_experts,
            top_k,
            # 학습용 예시라 seq aux loss를 작게 켜 둔다.
            # 실제 대형 MoE에서는 global aux, seq aux, z-loss, aux-loss-free bias 중
            # 어떤 조합을 쓸지 모델/프레임워크 정책에 따라 달라진다.
            seq_aux_loss_weight=0.1,
        )
        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        B, S, D = x.shape
        x_flat = x.view(B * S, D)  # (tokens, embed_dim)
        # sequence_ids[t] = token t가 몇 번째 sequence(batch row)에서 왔는지.
        # seq_aux_loss는 이 id를 사용해서 sequence별 expert 분포를 따로 본다.
        sequence_ids = torch.arange(B, device=x.device).repeat_interleave(S)

        # (1) Router: 각 토큰의 top-k expert 결정
        top_k_probs, top_k_indices, aux_loss = self.router(x_flat, sequence_ids=sequence_ids)
        # top_k_probs:   (tokens, top_k)
        # top_k_indices: (tokens, top_k)

        # (2) Dispatch: token들을 expert별 bucket으로 모은다.
        # dispatch라는 큰 단계 안에 보통 permutation/pack이 포함된다.
        # routing 결과는 원래 token 순서(batch/seq order)로 나오지만,
        # expert FFN은 "expert별로 연속된 token block"을 받는 편이 효율적이다.
        # 그래서 실제 MoE는 token을 expert id/destination rank 기준으로 정렬(permutation)한 뒤 계산한다.
        dispatch_batches, output = self._dispatch(x_flat, top_k_indices, top_k_probs)

        # (3) Expert compute: 각 expert FFN은 자기에게 온 token들만 계산한다.
        expert_results = self._expert_compute(dispatch_batches)

        # (4) Combine: expert output을 원래 token 위치에 gate score로 weighted scatter-add.
        # combine이라는 큰 단계 안에 보통 reverse transfer와 unpermutation/scatter-add가 포함된다.
        self._combine(output, expert_results)

        return output.view(B, S, D), aux_loss

    def _dispatch(self, x_flat, top_k_indices, top_k_probs):
        """
        Dispatch = token을 expert별 bucket으로 모으는 큰 단계.
        보통 dispatch 안에 permutation/pack이 함께 들어간다.

        원래 token layout:
          [tok0, tok1, tok2, tok3, ...]

        routing 결과가:
          tok0 -> expert2
          tok1 -> expert0
          tok2 -> expert2
          tok3 -> expert1

        이라면 expert compute 전에 보통 다음처럼 재배열한다.
          expert0 bucket: [tok1]
          expert1 bucket: [tok3]
          expert2 bucket: [tok0, tok2]

        이 재배열이 permutation이다. 하지만 실전 코드에서는 이걸 dispatch 단계의 일부로
        부르는 경우가 많다. token_positions는 나중에 output을 원래 위치로 되돌리는
        combine 단계의 unpermutation/scatter-add에 필요한 inverse mapping 역할을 한다.

        반환하는 dispatch_batches의 각 원소:
          (expert_id, token_positions, tokens_for_expert, gate_scores)

        분산 EP에서는 이 단계가 "내 GPU token 중 remote expert가 필요한 token을
        해당 expert가 있는 GPU로 보내는 all-to-all"이 된다.
        """
        dispatch_batches = []
        output = torch.zeros_like(x_flat)  # combine이 채워 넣을 원래 token layout

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]     # (tokens,) 각 토큰의 k번째 expert
            gate_scores = top_k_probs[:, k]          # (tokens,) 해당 expert의 gate 점수

            for expert_id in range(len(self.experts)):
                mask = expert_indices == expert_id
                if not mask.any():
                    continue

                token_positions = mask.nonzero(as_tuple=False).squeeze(-1)
                tokens_for_expert = x_flat[token_positions]
                dispatch_batches.append((
                    expert_id,
                    token_positions,
                    tokens_for_expert,
                    gate_scores[token_positions],
                ))

        return dispatch_batches, output

    def _expert_compute(self, dispatch_batches):
        """Compute = 각 expert가 dispatch로 받은 token bucket만 FFN 처리."""
        expert_results = []
        for expert_id, token_positions, tokens_for_expert, gate_scores in dispatch_batches:
            expert_output = self.experts[expert_id](tokens_for_expert)
            expert_results.append((token_positions, expert_output, gate_scores))
        return expert_results

    def _combine(self, output, expert_results):
        """
        Combine = expert output을 원래 token 위치로 되돌리는 큰 단계.
        EP에서는 보통 reverse all-to-all transfer + unpermutation/scatter-add를 함께 포함한다.
        여기서는 dispatch에서 expert별로 permute된 output을 token_positions를 사용해
        원래 layout으로 unpermute한다.

        top-k에서는 같은 token 위치에 여러 expert output이 들어올 수 있으므로
        단순 대입이 아니라 gate score를 곱한 뒤 더한다(scatter-add).
        """
        for token_positions, expert_output, gate_scores in expert_results:
            output[token_positions] += gate_scores.unsqueeze(-1) * expert_output


# ============================================================
# Part 4: Expert Parallelism 개념
# ============================================================
#
# Expert Parallelism: expert를 다른 GPU에 배치.
#
# 예: 8 experts, 4 GPUs → 각 GPU에 2 experts
#
#   GPU 0: [Expert 0, Expert 1]
#   GPU 1: [Expert 2, Expert 3]
#   GPU 2: [Expert 4, Expert 5]
#   GPU 3: [Expert 6, Expert 7]
#
# Dispatch / Compute / Combine 관점:
#   1. Router:
#      각 GPU가 자기 local token에 대해 top-k expert를 고른다.
#   2. Dispatch:
#      routing 결과를 보고 token을 expert id 또는 destination rank 기준으로 permute/pack한다.
#      그 다음 token을 "expert가 있는 GPU" 기준으로 전송한다.
#      즉 실전에서 dispatch는 보통 permutation/pack + all-to-all transfer를 함께 뜻한다.
#      GPU 0의 token 중 Expert 2,3에 배정된 것 → GPU 1로
#      GPU 1의 token 중 Expert 0,1에 배정된 것 → GPU 0로
#   3. Expert compute:
#      각 GPU는 자기가 소유한 local expert에 도착한 token bucket만 FFN 계산한다.
#   4. Combine:
#      expert output을 원래 token이 있던 GPU로 reverse all-to-all 한다.
#      돌아온 output을 inverse mapping으로 unpermute/scatter-add하고,
#      같은 token의 top-k expert output들을 gate score로 weighted sum한다.
#      즉 실전에서 combine은 보통 reverse transfer + unpermutation/scatter-add를 함께 뜻한다.
#
# all-to-all: 각 GPU가 다른 모든 GPU에 서로 다른 데이터를 전송
#   (all-reduce와 달리 각 GPU가 보내는/받는 데이터가 다름)
#
# dist.all_to_all(output_list, input_list)
#   input_list[i]  → GPU i로 보낼 데이터
#   output_list[i] ← GPU i로부터 받은 데이터
#
# 일반적인 all-to-all 구현:
#   - dispatch 전에 token을 destination rank별로 permute/pack한다.
#   - dist.all_to_all 또는 all_to_all_single로 token buckets를 교환한다.
#   - local expert compute 후 결과를 다시 pack해서 combine all-to-all을 한다.
#   - 돌아온 결과를 원래 token position으로 unpermute/scatter-add한다.
#   - 구현은 단순하지만 pack/unpack, padding, token count exchange, small-message overhead가 커질 수 있다.
#
# DeepEP 스타일:
#   - DeepEP는 DeepSeek 계열에서 공개한 MoE expert-parallel communication library.
#   - 목적은 "MoE의 dispatch/combine all-to-all"을 일반 collective보다 MoE 패턴에 맞게 빠르게 처리하는 것.
#   - normal kernel: training/prefill처럼 throughput이 중요한 상황에서 NVLink/RDMA 경로를 잘 활용.
#   - low-latency kernel: decoding처럼 token 수가 작고 latency가 중요한 상황에 맞춤.
#   - FP8 dispatch, BF16 combine, RDMA/NVLink forwarding, communication-compute overlap 같은 최적화를 제공.
#   - 개념적으로는 여전히 dispatch와 combine이다.
#     다만 PyTorch dist.all_to_all을 직접 호출하는 대신 MoE token routing에 특화된 kernel/runtime이
#     permutation/pack + transfer + unpack/unpermutation 일부를 더 가깝게 처리한다.
#
# 그래서 MoE 성능을 볼 때는 expert FFN FLOPs만 보면 안 된다.
#   total MoE time ≈ router + dispatch + expert compute + combine + aux/load-balance overhead
# 보통 expert parallel 규모가 커질수록 dispatch/combine 통신이 병목이 되기 쉬워서
# DeepEP 같은 전용 통신 layer가 중요해진다.
#
# MoE Parallel Folding:
#   dense Transformer block 안에서도 attention과 MoE MLP는 병렬화 특성이 다르다.
#
#   Attention:
#     - 모든 token이 매 layer에서 attention을 계산한다(dense).
#     - QKV/projection은 TP로 shard하기 좋고, 긴 context는 CP/SP로 sequence를 나누기 좋다.
#     - expert routing이 없으므로 EP를 크게 쓰는 이점은 없다.
#
#   Expert MLP:
#     - token마다 top-k expert만 활성화된다(sparse).
#     - expert weight 자체를 여러 GPU에 나눠 두는 EP가 중요하다.
#     - token을 expert 위치로 보내는 dispatch/combine all-to-all이 핵심 비용이 된다.
#
#   Parallel Folding은 이 둘에 같은 GPU mesh 해석을 강제하지 않는다는 뜻에 가깝다.
#   예를 들어 전체 16 GPU가 있을 때:
#
#     attention 관점:
#       TP=4, CP=2, DP=2  처럼 보고 dense attention/projection을 효율화
#
#     MoE MLP 관점:
#       EP=8, TP=1, DP=2  처럼 보고 expert를 넓게 펼쳐 token dispatch를 수행
#
#   물리 GPU는 같아도 layer type마다 "이 GPU들을 어떤 parallel dimension으로 볼 것인가"가 달라진다.
#   그래서 attention을 계산할 때 쓰는 process group과 expert MLP를 계산할 때 쓰는 process group이
#   서로 다를 수 있다. 이 folding/unfolding 때문에 MoE 학습은 일반 dense 모델보다 group 관리가 복잡하다.
#
#   직관:
#     - attention은 dense 연산이므로 TP/CP로 행렬과 sequence를 잘게 나누는 쪽이 유리하다.
#     - expert MLP는 sparse expert 선택이므로 EP로 expert 수용량을 늘리고 all-to-all을 최적화하는 쪽이 유리하다.
#     - Parallel Folding은 "attention 최적 GPU 배치"와 "expert 최적 GPU 배치"를 같은 모델 안에서 공존시키는 기법이다.


# ============================================================
# Part 5: dist API로 보는 Expert Parallel Dispatch/Combine
# ============================================================

def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def _expert_owner_ranks(expert_ids, num_experts, ep_world_size):
    """
    expert_id -> owner EP rank 매핑.

    여기서는 설명을 단순하게 하기 위해 expert를 rank별 contiguous block으로 배치한다.
      num_experts=8, ep_world_size=4라면
        rank0: expert 0,1
        rank1: expert 2,3
        rank2: expert 4,5
        rank3: expert 6,7

    Megatron/DeepEP 실제 구현에서는 node-limited routing, grouped GEMM layout,
    expert placement policy에 따라 매핑이 더 복잡할 수 있다.
    """
    if num_experts % ep_world_size != 0:
        raise ValueError("This simple example assumes num_experts is divisible by EP world size.")
    experts_per_rank = num_experts // ep_world_size
    return torch.div(expert_ids, experts_per_rank, rounding_mode="floor")


def ep_dispatch_all_to_all(x_flat, top_k_indices, top_k_probs, num_experts, group=None):
    """
    실제 Expert Parallel dispatch all-to-all 예시.

    입력:
      x_flat:        (local_tokens, hidden)
      top_k_indices: (local_tokens, top_k), global expert id
      top_k_probs:   (local_tokens, top_k), combine 때 쓸 gate score

    출력:
      recv_tokens:    이 rank가 소유한 expert들이 처리해야 할 token들
      recv_expert_ids: 각 token이 들어갈 global expert id
      recv_meta:      원래 token 위치와 gate score. combine 때 원래 rank/위치로 되돌리는 데 필요.
      routing_state:  combine all-to-all에서 split size를 맞추기 위한 metadata

    핵심:
      dispatch는 "remote expert가 필요한 token을 expert owner rank로 보내는 all-to-all"이다.
      token payload만 보내면 안 되고, combine을 위해 원래 token position과 gate score도 같이 보내야 한다.
      여기서 order = argsort(dest_ranks)가 permutation이다. 원래 token order를 destination rank별
      연속 bucket으로 바꿔 all-to-all 입력 형식에 맞춘다.
    """
    if not _dist_ready():
        raise RuntimeError("torch.distributed must be initialized for EP all-to-all dispatch.")

    ep_world_size = dist.get_world_size(group)
    local_tokens, hidden = x_flat.shape
    top_k = top_k_indices.shape[-1]

    # token 하나가 top-k개의 route를 만들기 때문에 dispatch 단위는 token이 아니라 routing slot이다.
    # 예: local_tokens=4, top_k=2 -> 총 8개의 route가 생긴다.
    route_token_positions = (
        torch.arange(local_tokens, device=x_flat.device)
        .unsqueeze(1)
        .expand(local_tokens, top_k)
        .reshape(-1)
    )
    route_expert_ids = top_k_indices.reshape(-1)
    route_gate_scores = top_k_probs.reshape(-1)
    route_tokens = x_flat[route_token_positions]

    # 각 route가 어느 EP rank로 가야 하는지 계산하고, all-to-all 입력을 destination rank별로 pack한다.
    # order가 바로 permutation index다.
    #   route_tokens: 원래 token/routing slot 순서
    #   route_tokens[order]: destination rank별로 모인 순서
    # 이렇게 permute해야 all_to_all_single의 input_split_sizes가
    # [rank0로 보낼 개수, rank1로 보낼 개수, ...]처럼 연속 chunk를 가리킬 수 있다.
    dest_ranks = _expert_owner_ranks(route_expert_ids, num_experts, ep_world_size)
    order = torch.argsort(dest_ranks)
    send_tokens = route_tokens[order].contiguous()
    send_expert_ids = route_expert_ids[order].contiguous()
    send_gate_scores = route_gate_scores[order].unsqueeze(-1).contiguous()

    # 실제 metadata는 dtype이 섞이므로 token_position(int64)과 gate_score(float)를 따로 보낸다.
    send_token_positions = route_token_positions[order].unsqueeze(-1).contiguous()
    send_counts = torch.bincount(dest_ranks, minlength=ep_world_size).to(torch.long)
    recv_counts = torch.empty_like(send_counts)

    # 먼저 각 rank가 몇 개의 route를 보낼지 교환한다.
    # variable-size all-to-all은 split size를 알아야 output buffer를 만들 수 있다.
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    input_splits = send_counts.cpu().tolist()
    output_splits = recv_counts.cpu().tolist()
    recv_total = int(recv_counts.sum().item())

    recv_tokens = torch.empty((recv_total, hidden), dtype=x_flat.dtype, device=x_flat.device)
    recv_expert_ids = torch.empty((recv_total,), dtype=send_expert_ids.dtype, device=x_flat.device)
    recv_token_positions = torch.empty((recv_total, 1), dtype=send_token_positions.dtype, device=x_flat.device)
    recv_gate_scores = torch.empty((recv_total, 1), dtype=send_gate_scores.dtype, device=x_flat.device)

    # 1) token payload dispatch
    dist.all_to_all_single(
        recv_tokens,
        send_tokens,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    # 2) 어떤 expert로 들어갈지
    dist.all_to_all_single(
        recv_expert_ids,
        send_expert_ids,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    # 3) combine 때 원래 token 위치로 되돌리기 위한 metadata
    dist.all_to_all_single(
        recv_token_positions,
        send_token_positions,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    # 4) top-k combine weight
    dist.all_to_all_single(
        recv_gate_scores,
        send_gate_scores,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    routing_state = {
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "local_tokens": local_tokens,
        "hidden": hidden,
        "group": group,
    }
    recv_meta = {
        "token_positions": recv_token_positions.squeeze(-1),
        "gate_scores": recv_gate_scores.squeeze(-1),
    }
    return recv_tokens, recv_expert_ids, recv_meta, routing_state


def ep_compute_local_experts(recv_tokens, recv_expert_ids, local_experts, num_experts, group=None):
    """
    dispatch로 받은 token bucket을 local expert별로 계산.

    이 함수는 각 EP rank가 contiguous expert block을 가진다고 가정한다.
    실제 고성능 MoE는 expert별 token을 더 촘촘하게 pack한 뒤 grouped GEMM으로 여러 expert FFN을
    한 번에 처리한다. 여기서는 dist 통신 흐름을 보여주기 위해 expert별 loop를 유지한다.
    """
    ep_rank = dist.get_rank(group) if _dist_ready() else 0
    ep_world_size = dist.get_world_size(group) if _dist_ready() else 1
    experts_per_rank = num_experts // ep_world_size
    first_expert = ep_rank * experts_per_rank

    computed = torch.empty_like(recv_tokens)
    for local_idx, expert in enumerate(local_experts):
        global_expert_id = first_expert + local_idx
        mask = recv_expert_ids == global_expert_id
        if mask.any():
            computed[mask] = expert(recv_tokens[mask])
    return computed


def ep_combine_all_to_all(expert_outputs, recv_meta, routing_state):
    """
    실제 Expert Parallel combine all-to-all 예시.

    dispatch의 반대 방향:
      expert owner rank에서 계산된 output을 원래 token이 있던 rank로 돌려보낸다.
      받은 rank는 token_positions에 gate_scores를 곱해서 scatter-add한다.
      즉 combine은 reverse all-to-all + unpermutation + weighted scatter-add로 볼 수 있다.
    """
    if not _dist_ready():
        raise RuntimeError("torch.distributed must be initialized for EP all-to-all combine.")

    group = routing_state["group"]
    recv_counts = routing_state["recv_counts"]
    send_counts = routing_state["send_counts"]
    local_tokens = routing_state["local_tokens"]
    hidden = routing_state["hidden"]
    ep_world_size = dist.get_world_size(group)

    # dispatch에서 받은 recv buffer는 source rank 순서의 chunk로 구성된다.
    # 따라서 combine에서는 그 source rank로 다시 돌려보내면 된다.
    # 이 order도 combine 방향의 permutation이다. expert output을 destination(source) rank별로
    # 연속하게 pack해서 reverse all-to-all 입력으로 만든다.
    source_ranks = torch.repeat_interleave(
        torch.arange(ep_world_size, device=expert_outputs.device),
        recv_counts.to(expert_outputs.device),
    )
    order = torch.argsort(source_ranks)

    send_outputs = expert_outputs[order].contiguous()
    send_token_positions = recv_meta["token_positions"][order].unsqueeze(-1).contiguous()
    send_gate_scores = recv_meta["gate_scores"][order].unsqueeze(-1).contiguous()

    # combine은 dispatch의 reverse 방향이므로 local rank가 다시 받을 route 수는
    # 처음 dispatch 때 자신이 각 rank로 보냈던 send_counts의 합이다.
    input_splits = recv_counts.cpu().tolist()
    output_splits = send_counts.cpu().tolist()
    recv_total = int(send_counts.sum().item())

    recv_outputs = torch.empty((recv_total, hidden), dtype=expert_outputs.dtype, device=expert_outputs.device)
    recv_token_positions = torch.empty((recv_total, 1), dtype=send_token_positions.dtype, device=expert_outputs.device)
    recv_gate_scores = torch.empty((recv_total, 1), dtype=send_gate_scores.dtype, device=expert_outputs.device)

    dist.all_to_all_single(
        recv_outputs,
        send_outputs,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    dist.all_to_all_single(
        recv_token_positions,
        send_token_positions,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    dist.all_to_all_single(
        recv_gate_scores,
        send_gate_scores,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    output = torch.zeros((local_tokens, hidden), dtype=expert_outputs.dtype, device=expert_outputs.device)
    token_positions = recv_token_positions.squeeze(-1)
    gate_scores = recv_gate_scores.squeeze(-1)
    # token_positions가 dispatch 때 저장해 둔 inverse mapping이다.
    # 이 mapping으로 expert별/permuted output을 원래 local token 위치로 unpermute한다.
    # top-k route가 같은 token 위치로 여러 번 돌아올 수 있으므로 scatter-add가 필요하다.
    output.index_add_(0, token_positions, gate_scores.unsqueeze(-1) * recv_outputs)
    return output


def expert_parallel_forward_dist_example(x_flat, top_k_indices, top_k_probs, local_experts, num_experts, group=None):
    """
    분산 Expert Parallel MoE forward의 최소 형태.

    이 예시는 현재 MoELayer.forward에서 직접 호출하지 않는다.
    이유는 학습용 single-process demo를 깨지 않기 위해서다.
    하지만 실제 EP에서는 아래 순서가 MoELayer의 _dispatch/_expert_compute/_combine을 대체한다.

      recv_tokens, recv_expert_ids, meta, state = ep_dispatch_all_to_all(...)
      expert_outputs = ep_compute_local_experts(...)
      output = ep_combine_all_to_all(expert_outputs, meta, state)
    """
    recv_tokens, recv_expert_ids, recv_meta, routing_state = ep_dispatch_all_to_all(
        x_flat,
        top_k_indices,
        top_k_probs,
        num_experts,
        group=group,
    )
    expert_outputs = ep_compute_local_experts(
        recv_tokens,
        recv_expert_ids,
        local_experts,
        num_experts,
        group=group,
    )
    return ep_combine_all_to_all(expert_outputs, recv_meta, routing_state)


# ============================================================
# Part 6: Demo
# ============================================================

def demo():
    print("=" * 60)
    print("Mixture of Experts Demo")
    print("=" * 60)

    torch.manual_seed(42)
    B, S, D = 2, 8, 64
    num_experts = 4
    top_k = 2

    moe = MoELayer(embed_dim=D, hidden_dim=256, num_experts=num_experts, top_k=top_k)
    x = torch.randn(B, S, D)

    output, aux_loss = moe(x)
    aux_breakdown = moe.router.last_aux_loss_breakdown

    print(f"\n  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f}")
    print(f"    load-balance: {aux_breakdown['load_balance_loss'].item():.4f} (균등하면 1.0 근처)")
    print(
        f"    seq aux: {aux_breakdown['seq_aux_loss'].item():.4f} "
        f"* {aux_breakdown['seq_aux_loss_weight']} "
        "(sequence별 expert 쏠림 완화)"
    )
    print(
        f"    router z-loss: {aux_breakdown['router_z_loss'].item():.4f} "
        f"* {aux_breakdown['z_loss_weight']}"
    )

    # Expert 배정 분석
    x_flat = x.view(B * S, D)
    _, top_k_indices, _ = moe.router(x_flat)
    print(f"\n  Expert assignments (top-{top_k}):")
    for e in range(num_experts):
        count = (top_k_indices == e).sum().item()
        pct = count / (B * S * top_k) * 100
        print(f"    Expert {e}: {count} tokens ({pct:.0f}%)")

    # FLOPs 비교
    dense_flops = D * 256 * 2 * num_experts  # 모든 expert 다 쓴다면
    moe_flops = D * 256 * 2 * top_k          # top-k만 사용
    print(f"\n  FLOPs per token:")
    print(f"    Dense (all experts): {dense_flops:,}")
    print(f"    MoE (top-{top_k}):         {moe_flops:,}")
    print(f"    절약: {(1 - moe_flops/dense_flops)*100:.0f}%")
    print(f"\n  총 파라미터: {sum(p.numel() for p in moe.parameters()):,}")


if __name__ == "__main__":
    demo()
