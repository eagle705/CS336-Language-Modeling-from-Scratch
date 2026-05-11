"""
Context Parallelism (CP)
==========================
시퀀스를 여러 GPU에 나눠서 처리. Long context 학습의 핵심.

문제: seq_len이 길면 attention의 O(S^2) 메모리가 단일 GPU에 안 들어감.
  예: S=128K, H=32, D=128, BF16 → attention scores만 ~128GB

TP/PP/DP로는 해결 안 됨:
  TP: layer 내부를 나누지만, 각 GPU가 전체 시퀀스의 attention 계산
  PP: layer를 나누지만, 각 stage에서 전체 시퀀스 필요
  DP: 데이터를 나누지만, 각 GPU가 전체 시퀀스 처리

CP 해결: 시퀀스 자체를 GPU에 분산!

    전체 시퀀스: [tok_0, tok_1, ..., tok_S-1]

    GPU 0: [tok_0, ..., tok_{S/4-1}]         ← Q chunk 0
    GPU 1: [tok_{S/4}, ..., tok_{S/2-1}]     ← Q chunk 1
    GPU 2: [tok_{S/2}, ..., tok_{3S/4-1}]    ← Q chunk 2
    GPU 3: [tok_{3S/4}, ..., tok_{S-1}]      ← Q chunk 3

    각 GPU가 자기 Q chunk에 대해 전체 KV를 순회하며 attention 계산.
    KV를 ring 형태로 GPU 간 전달 (= Ring Attention).

Ring Attention 동작:
    ┌─────────────────────────────────────────────────────┐
    │ Step 0: 각 GPU가 local KV로 attention 계산          │
    │   GPU 0: Q0 × KV0   GPU 1: Q1 × KV1   ...         │
    │                                                     │
    │ Step 1: KV를 오른쪽 이웃에게 전달 (ring)              │
    │   GPU 0: Q0 × KV3   GPU 1: Q1 × KV0   ...         │
    │         (KV3 받음)          (KV0 받음)               │
    │                                                     │
    │ Step 2: 다시 전달                                    │
    │   GPU 0: Q0 × KV2   GPU 1: Q1 × KV3   ...         │
    │                                                     │
    │ Step 3: 마지막 KV chunk                              │
    │   GPU 0: Q0 × KV1   GPU 1: Q1 × KV2   ...         │
    │                                                     │
    │ → 각 GPU가 전체 KV를 한 바퀴 돌며 attention 완성     │
    │ → 통신과 연산을 overlap 가능!                         │
    └─────────────────────────────────────────────────────┘

핵심 트릭: Online Softmax (Flash Attention과 동일)
  KV chunk가 하나씩 올 때마다 running max, running sum 업데이트.
  전체 attention matrix를 만들지 않아도 정확한 softmax 결과.

인터뷰 포인트:
  1. CP는 시퀀스 차원을 분산 → S^2 메모리 문제 해결
  2. Ring Attention = CP + online softmax
  3. 통신: KV chunk를 ring으로 전달 (send/recv, overlap 가능)
  4. Causal mask 최적화: 자기보다 미래 토큰의 KV는 skip 가능
  5. 4D parallelism: TP × PP × DP × CP
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
from enum import Enum


class CPCommType(Enum):
    """Context Parallelism 통신 유형.

    P2P (Ring Attention):
        KV를 이웃 GPU로 순차 전달. 통신과 연산 overlap 가능.
        + 메모리 효율 최고 (한 번에 KV chunk 1개만 보유)
        + 통신-연산 overlap → 통신 숨기기
        - 구현 복잡 (send/recv 동기화)
        - latency = (cp_size-1) × per-step latency

    ALL_GATHER:
        모든 GPU가 전체 KV를 한 번에 모음.
        중요한 점: 전체 KV를 모은다고 해서 모든 GPU가 전체 sequence output을 계산하는 것은 아님.
        Q/output은 여전히 sequence shard 단위로 유지되고, K/V만 attention 계산 순간 full로 복제된다.

        예: CP=2, seq=8이면
          GPU0: Q=token 0-3, K/V=token 0-7 → output token 0-3만 계산
          GPU1: Q=token 4-7, K/V=token 0-7 → output token 4-7만 계산

        따라서 layer 사이에 저장되는 activation/residual은 shard(seq/CP)라 작지만,
        attention 계산 순간에는 temporary full K/V 때문에 peak memory가 커질 수 있다.
        128k 같은 long context에서는 이 peak memory 때문에 ALL_GATHER 방식이 터질 수 있어
        실전에서는 P2P/Ring처럼 KV chunk를 순차로 받아 full KV를 한 번에 만들지 않는 방식이 중요하다.
        반대로 말하면, full K/V를 한 번에 만든다는 점 때문에 구현은 매우 쉽다.
        특히 THD(packed tokens) + SWA/varlen attention 커널처럼 "전체 token axis를 보고"
        window/document boundary를 커널 metadata로 처리하는 코드에는 all-gather가 붙이기 쉽다.
        그래서 최근 구현/PR에서 ALL_GATHER 기반 CP가 쓰이는 경우도 있다.
        핵심 trade-off는 구현 단순성/커널 재사용성 vs attention 순간 peak memory다.

        + 구현 단순 (한 번의 collective)
        + latency 낮음 (한 번에 끝)
        - attention 계산 중 peak memory 큼 (모든 rank가 temporary full KV 보유)
        - overlap 불가 (gather 후 compute)

    A2A (All-to-All, DeepSpeed-Ulysses):
        시퀀스 분할 → 헤드 분할로 layout 변환.
        Q,K,V를 all-to-all로 재배치: 각 GPU가 모든 시퀀스의 일부 헤드 담당.

        왜 head를 분할하는데도 Context Parallelism인가?
        layer 입출력/residual/MLP activation은 여전히 context(seq) 차원으로 shard되어 있다.
        attention 계산 순간에만 all-to-all로 layout을 바꿔
          seq-shard(all heads) → full-seq(partial heads)
        형태로 만든 뒤, attention 후 다시
          full-seq(partial heads) → seq-shard(all heads)
        로 되돌린다. 즉 전체 layer 관점의 저장/전달 단위는 context shard라서 CP로 분류된다.

        long context 주의:
        A2A도 attention 내부에서는 각 rank가 full seq를 본다. 대신 heads는 H/CP만 담당한다.
        여기서 H는 전체 attention head 수, CP는 context parallel rank 수,
        H/CP는 A2A 후 rank 하나가 맡는 head 개수다.
        예: H=32 heads, CP=4이면 각 rank는 full seq에 대해 8 heads만 계산한다.
        따라서 head 수가 충분하면 KV/activation head 축 메모리는 줄지만,
        naive attention score처럼 seq×seq를 materialize하면 128k에서는 여전히 터질 수 있다.
        실제로는 FlashAttention/online softmax 같은 block attention 커널과 같이 써야 한다.

        SWA(Sliding Window Attention)와 잘 맞는 이유:
        SWA는 각 token이 전체 과거가 아니라 가까운 window만 보므로 attention의 유효 KV 범위가 작다.
        A2A로 head를 나누면 각 rank가 담당 head에 대해 window attention을 독립 계산할 수 있고,
        전체 dense attention보다 통신/메모리 압력이 낮아진다. 그래서 Ulysses/A2A 계열과 함께 쓰기 좋다.
        다만 SWA/THD 구현이 항상 A2A만 쓰는 것은 아니다.
        ALL_GATHER도 full K/V를 만들어 전체 token axis를 볼 수 있게 하므로,
        기존 THD + SWA/varlen attention 커널을 거의 그대로 붙이기 쉽다.
        그래서 구현 단순성이나 커널 재사용성이 중요하고 peak memory가 허용되면 ALL_GATHER PR/구현도 등장한다.

        + attention 내부에서 추가 통신 불필요
        + GQA에서도 효율적
        - all-to-all 2회 필요 (forward: seq→head, head→seq)
        - head 수가 cp_size로 나눠져야 함
    """
    P2P = "p2p"
    ALL_GATHER = "all_gather"
    A2A = "a2a"


# ============================================================
# Part 1: Ring Attention 구현 (시뮬레이션 + torch.distributed)
# ============================================================

def ring_attention(Q_chunks, K_chunks, V_chunks, causal=False,
                   comm_type=CPCommType.P2P, group=None):
    """
    Context Parallelism attention.

    두 가지 모드로 동작한다.

    1. 단일 프로세스 시뮬레이션:
       Q_chunks/K_chunks/V_chunks가 list이면 기존처럼 모든 GPU를 한 프로세스에서 흉내 낸다.

    2. 실제 torch.distributed 통신:
       Q_chunks/K_chunks/V_chunks가 Tensor이고 process group이 초기화되어 있으면,
       현재 rank의 local Q/K/V shard로 실제 collective/P2P 통신을 수행한다.

       예:
         # torchrun --nproc_per_node=4 train.py
         dist.init_process_group("nccl")
         out_local = ring_attention(
             Q_local, K_local, V_local,
             causal=True,
             comm_type=CPCommType.P2P,
         )

    Q_chunks: list of (chunk_seq, head_dim) per GPU
              또는 local Tensor (chunk_seq, head_dim) / (chunk_seq, heads, head_dim)
    K_chunks: list of (chunk_seq, head_dim) per GPU
    V_chunks: list of (chunk_seq, head_dim) per GPU
    comm_type: CPCommType — P2P, ALL_GATHER, A2A
    group: 어떤 rank 집합 안에서 통신할지 정하는 process group.
           None이면 default world group이라 전체 rank가 참여한다.
           실제 4D parallelism에서는 전체 world가 아니라 같은 TP/PP/DP 좌표를 가진
           CP rank들끼리만 KV를 교환해야 하므로 cp_group을 넘긴다.
           예: TP×CP×PP×DP = 2×4×2×8이면 world는 128 ranks지만,
               CP 통신은 각 cp_group 안의 4 ranks끼리만 수행한다.
    """
    if torch.is_tensor(Q_chunks):
        if not _dist_ready(group):
            raise RuntimeError(
                "Tensor input은 torch.distributed process group이 초기화되어 있을 때만 사용하세요. "
                "단일 프로세스 데모는 list chunks를 넘기면 됩니다."
            )
        return _cp_distributed(Q_chunks, K_chunks, V_chunks, causal, comm_type, group)

    if comm_type == CPCommType.P2P:
        return _cp_p2p(Q_chunks, K_chunks, V_chunks, causal)
    elif comm_type == CPCommType.ALL_GATHER:
        return _cp_all_gather(Q_chunks, K_chunks, V_chunks, causal)
    elif comm_type == CPCommType.A2A:
        return _cp_a2a(Q_chunks, K_chunks, V_chunks, causal)
    else:
        raise ValueError(f"Unknown comm_type: {comm_type}")


def _dist_ready(group=None):
    """
    torchrun 등으로 process group이 초기화되어 있으면 실제 통신 경로를 사용한다.

    group=None이면 default world group의 크기를 본다.
    group=cp_group이면 CP group 안의 rank 수만 본다.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def _cp_distributed(Q_local, K_local, V_local, causal, comm_type, group=None):
    """현재 rank의 local shard로 실제 torch.distributed CP attention을 수행."""
    if comm_type == CPCommType.P2P:
        return _cp_p2p_dist(Q_local, K_local, V_local, causal, group)
    elif comm_type == CPCommType.ALL_GATHER:
        return _cp_all_gather_dist(Q_local, K_local, V_local, causal, group)
    elif comm_type == CPCommType.A2A:
        return _cp_a2a_dist(Q_local, K_local, V_local, causal, group)
    else:
        raise ValueError(f"Unknown comm_type: {comm_type}")


def _causal_positions(rank, chunk_size, device):
    """현재 rank의 local chunk가 전체 sequence에서 차지하는 token position."""
    return torch.arange(rank * chunk_size, (rank + 1) * chunk_size, device=device)


def _online_attention_update(Q_local, K_block, V_block, m, l, O_acc,
                             causal, q_positions, k_positions):
    """Ring/P2P에서 KV block 하나를 보고 online softmax 상태를 갱신."""
    D = Q_local.shape[-1]
    S_block = Q_local @ K_block.T / math.sqrt(D)

    if causal:
        # causal_mask=True인 위치만 attention 가능: query position >= key position
        # 예: q_positions=[2], k_positions=[0,1,2,3]이면
        #     causal_mask = [True, True, True, False]
        #     token 2는 key 0,1,2는 볼 수 있지만 미래 token 3은 볼 수 없다.
        causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
        # ~causal_mask는 boolean을 뒤집어서 "볼 수 없는 위치"를 고른다.
        #     ~causal_mask = [False, False, False, True]
        # masked_fill은 그 위치의 score를 -inf로 바꿔 softmax 확률이 0이 되게 한다.
        S_block = S_block.masked_fill(~causal_mask, float('-inf'))

    # Online softmax update (row/query별로 독립):
    # 지금까지 본 score들을 old, 새 KV block score를 S_block이라 하면
    #   m      = max(old scores)
    #   l      = sum(exp(old scores - m))
    #   O_acc  = sum(exp(old scores - m) * V_old)
    #
    # 새 block을 합칠 때 전체 max는
    #   m_new = max(m, max(S_block))
    #
    # 기준 max가 m에서 m_new로 바뀌면 이전 누적값도 같은 기준으로 다시 스케일해야 한다:
    #   exp(old - m_new) = exp(old - m) * exp(m - m_new)
    # 그래서 correction = exp(m - m_new).
    #
    # 최종 갱신식:
    #   l_new     = correction * l + sum(exp(S_block - m_new))
    #   O_acc_new = correction * O_acc + exp(S_block - m_new) @ V_block
    # 마지막 output은 O_acc_new / l_new.
    # 이 덕분에 전체 S×S attention matrix를 만들지 않고 KV block을 하나씩 처리할 수 있다.
    m_block = S_block.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m, m_block)
    correction = torch.exp(m - m_new)
    P_block = torch.exp(S_block - m_new)
    l_new = correction * l + P_block.sum(dim=-1, keepdim=True)
    O_acc = correction * O_acc + P_block @ V_block
    return m_new, l_new, O_acc


# -------------------------------------------------------
# (1) P2P: Ring Attention
# -------------------------------------------------------
# KV를 ring 형태로 이웃 GPU에 순차 전달.
# 한 번에 KV chunk 1개만 보유 → 메모리 효율 최고.
# 통신과 연산을 overlap할 수 있어 실전에서 가장 많이 사용.
#
# Timeline (GPU 0 기준):
#   Step 0: compute(Q0, KV0)  |  send KV0→GPU1, recv KV3←GPU3
#   Step 1: compute(Q0, KV3)  |  send KV3→GPU1, recv KV2←GPU3
#   Step 2: compute(Q0, KV2)  |  send KV2→GPU1, recv KV1←GPU3
#   Step 3: compute(Q0, KV1)  |  (마지막, 통신 없음)
# -------------------------------------------------------

def _cp_p2p(Q_chunks, K_chunks, V_chunks, causal):
    """P2P Ring Attention: KV를 ring으로 돌리며 online softmax."""
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    D = Q_chunks[0].shape[-1]
    outputs = []

    for gpu_id in range(num_gpus):
        Q_local = Q_chunks[gpu_id]

        # Online softmax 상태 초기화
        m = torch.full((chunk_size, 1), float('-inf'))  # running max
        l = torch.zeros(chunk_size, 1)                   # running sum(exp)
        O_acc = torch.zeros(chunk_size, D)                # running weighted sum

        # Ring으로 KV chunk 순회 (num_gpus steps)
        for step in range(num_gpus):
            # 실제 분산 환경:
            #   kv_buf = local_kv if step==0 else recv_from(prev_gpu)
            #   if step < num_gpus-1: async_send(kv_buf, next_gpu)
            # send next / recv prev 방향이면 GPU0 기준 KV0 → KV3 → KV2 → KV1 순서로 본다.
            kv_idx = (gpu_id - step) % num_gpus

            K_block = K_chunks[kv_idx]
            V_block = V_chunks[kv_idx]

            # --- Causal mask 최적화 ---
            if causal and kv_idx > gpu_id:
                continue

            S_block = Q_local @ K_block.T / math.sqrt(D)

            if causal:
                q_positions = torch.arange(gpu_id * chunk_size,
                                           (gpu_id + 1) * chunk_size)
                k_positions = torch.arange(kv_idx * chunk_size,
                                           (kv_idx + 1) * chunk_size)
                causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
                S_block = S_block.masked_fill(~causal_mask, float('-inf'))

            # --- Online softmax update ---
            m_block = S_block.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_block)
            correction = torch.exp(m - m_new)
            P_block = torch.exp(S_block - m_new)
            l_new = correction * l + P_block.sum(dim=-1, keepdim=True)
            O_acc = correction * O_acc + P_block @ V_block
            m = m_new
            l = l_new

        outputs.append(O_acc / l)

    return outputs


def _cp_p2p_dist(Q_local, K_local, V_local, causal, group=None):
    """
    실제 torch.distributed P2P Ring Attention.

    각 rank 입력:
      Q_local: (chunk_seq, head_dim)  현재 rank가 담당하는 query token shard
      K_local: (chunk_seq, head_dim)  현재 rank가 가진 local key shard
      V_local: (chunk_seq, head_dim)  현재 rank가 가진 local value shard

    동작:
      Step 0: local KV로 attention partial 계산
      Step 1..N-1: 현재 KV buffer를 next rank로 보내고 prev rank에서 KV를 받아 계산

    full K/V를 한 번에 만들지 않고 KV chunk 하나씩만 보므로 peak memory가 작다.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    Q_local = Q_local.contiguous()
    current_K = K_local.contiguous()
    current_V = V_local.contiguous()

    chunk_size = Q_local.shape[0]
    D = Q_local.shape[-1]
    device = Q_local.device

    # Online softmax 상태. 전체 attention score matrix를 만들지 않고 block 단위로 누적한다.
    m = torch.full((chunk_size, 1), float('-inf'), device=device, dtype=Q_local.dtype)
    l = torch.zeros(chunk_size, 1, device=device, dtype=Q_local.dtype)
    O_acc = torch.zeros(chunk_size, D, device=device, dtype=Q_local.dtype)
    q_positions = _causal_positions(rank, chunk_size, device)

    for step in range(world_size):
        # 이 ring 방향에서는 step=1에 prev rank의 KV를 받는다.
        # rank0 기준: KV0 → KV(world-1) → KV(world-2) ...
        kv_rank = (rank - step) % world_size

        # Causal attention에서는 내 query보다 미래에 있는 KV block은 전부 볼 필요가 없다.
        if not (causal and kv_rank > rank):
            k_positions = _causal_positions(kv_rank, chunk_size, device)
            m, l, O_acc = _online_attention_update(
                Q_local, current_K, current_V, m, l, O_acc,
                causal, q_positions, k_positions,
            )

        if step == world_size - 1:
            break

        # 현재 들고 있는 KV block을 next로 보내고, prev에서 다음 KV block을 받는다.
        # batch_isend_irecv를 쓰면 모든 rank가 같은 순서로 send/recv를 걸어 deadlock을 피할 수 있다.
        recv_K = torch.empty_like(current_K)
        recv_V = torch.empty_like(current_V)
        # group은 이 P2P 통신이 일어나는 rank 집합.
        # group=None이면 전체 world 기준 rank 번호를 쓰고,
        # group=cp_group이면 cp_group 내부 rank들끼리만 ring을 돈다.
        # 실제 TP/PP/DP/CP가 섞인 학습에서는 반드시 cp_group으로 제한해야
        # 엉뚱한 TP/PP/DP rank와 KV를 교환하지 않는다.
        # dist.P2POp는 "어떤 P2P 연산을 누구와 할지"를 담은 descriptor.
        # isend/irecv의 i는 immediate: 통신을 시작만 하고 즉시 Work handle을 반환한다는 뜻.
        # 완료까지 기다리는 blocking send/recv와 달리, 완료 보장은 아래 req.wait()에서 한다.
        #   dist.isend(tensor, dst): tensor를 dst rank로 비동기 전송
        #   dist.irecv(tensor, src): src rank에서 오는 데이터를 tensor 버퍼에 비동기 수신
        # 여기서는 K/V 두 텐서를 next_rank로 보내면서, 동시에 prev_rank에서 다음 K/V를 받는다.
        ops = [
            dist.P2POp(dist.isend, current_K, next_rank, group=group),
            dist.P2POp(dist.isend, current_V, next_rank, group=group),
            dist.P2POp(dist.irecv, recv_K, prev_rank, group=group),
            dist.P2POp(dist.irecv, recv_V, prev_rank, group=group),
        ]
        # batch_isend_irecv는 위 P2POp들을 한 번에 launch하고 Work handle 목록을 반환한다.
        # 모든 rank가 같은 패턴으로 batch를 걸면 send/recv 순서 꼬임이나 deadlock 위험이 줄어든다.
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            # wait() 전까지 통신은 아직 끝났다고 보장되지 않는다.
            # recv_K/recv_V를 compute에 쓰기 전에 반드시 완료를 기다린다.
            req.wait()
        current_K, current_V = recv_K, recv_V

    return O_acc / l


# -------------------------------------------------------
# (2) ALL_GATHER: 전체 KV를 한 번에 모은 뒤 local attention
# -------------------------------------------------------
# 모든 GPU가 all_gather로 전체 K, V를 받음.
# 그 후 각 GPU가 자기 Q chunk에 대해 전체 KV로 표준 attention.
#
# 메모리 흐름:
#   평소/layer 사이:
#     Q, K, V, residual, MLP activation 등은 seq chunk만 보유 → (seq/CP, D)
#   attention 계산 직전:
#     K/V를 all_gather해서 temporary full K/V 생성 → (seq, D)
#   attention 계산:
#     Q는 여전히 local shard. 즉 output도 local shard만 계산.
#     "full KV를 본다"와 "full sequence output을 중복 계산한다"는 다르다.
#   attention 계산 후:
#     output shard만 다음 layer로 넘기고 full K/V temporary buffer는 버릴 수 있음.
#
# 예: CP=2, seq=8
#   GPU0: Q0-3 @ K0-7.T → O0-3
#   GPU1: Q4-7 @ K0-7.T → O4-7
#   K/V는 full로 복제되지만 O는 shard로 남는다.
#
# 장점: 구현이 매우 단순, 한 번의 collective로 끝
# 단점: attention 순간 전체 KV를 들고 있어야 해서 peak memory가 큼
#       gather 완료 후에야 연산 시작 → overlap 불가
#       128k long context에서는 temporary full KV만으로도 OOM 가능
#
# 최근 ALL_GATHER 기반 CP가 다시 쓰이는 이유:
#   full K/V를 만든 뒤에는 각 rank가 자기 Q shard에 대해 "일반 attention"처럼 계산할 수 있다.
#   THD(packed token) + SWA/varlen FlashAttention 같은 커널은 전체 token axis와
#   cu_seqlens/window metadata를 보고 boundary를 처리하는 형태가 많다.
#   all-gather는 이 full-token view를 쉽게 만들어 주므로 기존 커널을 재사용하기 쉽다.
#   대신 full K/V temporary를 감당할 수 있는 context length/CP size/메모리 조건에서만 현실적이다.
#
# 실제 코드 (torch.distributed):
#   K_full = [torch.empty_like(K_local)] * cp_size
#   dist.all_gather(K_full, K_local, group=cp_group)
#   K_full = torch.cat(K_full, dim=0)  # (full_seq, D)
# -------------------------------------------------------

def _cp_all_gather(Q_chunks, K_chunks, V_chunks, causal):
    """All-Gather: 전체 KV를 모은 뒤 각 GPU에서 local attention."""
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    total_seq = chunk_size * num_gpus
    outputs = []

    # all_gather 시뮬레이션: 모든 GPU가 전체 K, V를 갖게 됨
    # 실제 CP에서는 각 rank의 local K/V는 (chunk_size, D)이지만,
    # attention을 계산하는 동안만 K_full/V_full = (total_seq, D)를 temporary로 만든다.
    # 이 temporary 때문에 peak memory는 커지지만, 다음 layer로 넘기는 output은 여전히 shard다.
    K_full = torch.cat(K_chunks, dim=0)  # (total_seq, D)
    V_full = torch.cat(V_chunks, dim=0)  # (total_seq, D)

    for gpu_id in range(num_gpus):
        Q_local = Q_chunks[gpu_id]  # (chunk_size, D)

        # 표준 attention: Q_local @ K_full^T
        # Q_local은 이 GPU가 담당하는 seq shard라서 scores/out의 0번 차원도 chunk_size.
        # 즉 full KV를 보지만 전체 seq output을 중복 계산하지는 않는다.
        scores = Q_local @ K_full.T / math.sqrt(Q_local.shape[-1])

        if causal:
            q_positions = torch.arange(gpu_id * chunk_size,
                                       (gpu_id + 1) * chunk_size)
            k_positions = torch.arange(total_seq)
            causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V_full
        outputs.append(out)  # (chunk_size, D): 이 GPU의 local output shard

    return outputs


def _cp_all_gather_dist(Q_local, K_local, V_local, causal, group=None):
    """
    실제 torch.distributed All-Gather CP attention.

    각 rank는 Q_local shard만 계산하지만, K/V는 all_gather로 full sequence를 임시 복제한다.
    반환값은 현재 rank의 local output shard: (chunk_seq, head_dim)
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    Q_local = Q_local.contiguous()
    K_local = K_local.contiguous()
    V_local = V_local.contiguous()

    K_parts = [torch.empty_like(K_local) for _ in range(world_size)]
    V_parts = [torch.empty_like(V_local) for _ in range(world_size)]
    # dist.all_gather(output_list, input_tensor):
    #   첫 번째 인자(K_parts/V_parts)는 모든 rank의 tensor를 받을 빈 리스트.
    #   두 번째 인자(K_local/V_local)는 현재 rank가 다른 rank들에게 보낼 local tensor.
    # 예: CP group이 4 ranks이고 각 rank가 K0,K1,K2,K3를 들고 있으면,
    #     호출 후 모든 rank의 K_parts가 [K0, K1, K2, K3]로 채워진다.
    dist.all_gather(K_parts, K_local, group=group)
    dist.all_gather(V_parts, V_local, group=group)

    # rank 순서대로 concat되어 전체 sequence K/V가 된다.
    K_full = torch.cat(K_parts, dim=0)
    V_full = torch.cat(V_parts, dim=0)

    scores = Q_local @ K_full.T / math.sqrt(Q_local.shape[-1])
    if causal:
        chunk_size = Q_local.shape[0]
        q_positions = _causal_positions(rank, chunk_size, Q_local.device)
        k_positions = torch.arange(K_full.shape[0], device=Q_local.device)
        causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    return F.softmax(scores, dim=-1) @ V_full


# -------------------------------------------------------
# (3) A2A: All-to-All (DeepSpeed-Ulysses 스타일)
# -------------------------------------------------------
# 핵심 아이디어: 시퀀스 분할 → 헤드 분할로 layout 변환.
#
# CP의 기본 저장 layout은 seq-shard:
#   각 GPU가 일부 context token과 모든 heads를 들고 있음.
# A2A는 attention 계산을 위해 잠깐 head-shard layout으로 바꾼다:
#   각 GPU가 전체 context token과 일부 heads를 담당.
# attention 후에는 다시 seq-shard로 되돌리므로 다음 layer/residual/MLP는 CP 상태를 유지한다.
#
# Before all-to-all:
#   GPU 0: Q[0:S/N, :], K[0:S/N, :], V[0:S/N, :]  (시퀀스 분할)
#   GPU 1: Q[S/N:2S/N, :], K[S/N:2S/N, :], V[S/N:2S/N, :]
#   ...
#
# After all-to-all:
#   GPU 0: Q[:, 0:H/N], K[:, 0:H/N], V[:, 0:H/N]  (헤드 분할)
#   GPU 1: Q[:, H/N:2H/N], K[:, H/N:2H/N], V[:, H/N:2H/N]
#   ...
#
# 각 GPU가 전체 시퀀스의 일부 헤드에 대해 완전한 attention 계산.
# → attention 내부에서 추가 통신 불필요!
# → 결과를 다시 all-to-all로 시퀀스 분할로 되돌림.
#
# long context에서는 주의:
#   A2A도 full sequence를 각 rank가 보므로, dense score(S×S)를 직접 만들면 128k에서 OOM 가능.
#   이 방식의 메모리 이득은 head 축을 H/N으로 나누는 데서 오고,
#   H는 전체 attention head 수, N은 CP rank 수(cp_size)다.
#   예: H=32, N=4이면 A2A 후 각 rank는 full seq에 대해 8 heads(H/N)만 담당.
#   seq 축의 quadratic attention 메모리는 FlashAttention/online softmax가 막아줘야 한다.
#
# SWA(Sliding Window Attention)와의 관계:
#   SWA는 각 query가 가까운 window의 K/V만 보므로 full dense S×S보다 유효 attention 범위가 작다.
#   A2A로 head를 나누면 각 rank가 자기 heads의 window attention을 독립적으로 계산할 수 있고,
#   window 밖 token은 볼 필요가 없어 통신/메모리 압력이 더 낮다.
#   그래서 long context 모델에서 SWA + A2A/Ulysses 조합이 자주 등장한다.
#   하지만 ALL_GATHER도 선택지가 될 수 있다:
#     full K/V를 모으면 각 rank가 전체 packed token axis를 볼 수 있어
#     THD + SWA/varlen FlashAttention 커널을 그대로 호출하기 쉽다.
#     즉 A2A는 head-shard로 메모리/통신을 줄이기 위한 선택,
#     ALL_GATHER는 peak memory를 감수하고 구현과 커널 재사용을 쉽게 하는 선택이다.
#
# 단, single-head 시뮬레이션에서는 헤드 차원이 없으므로
# head_dim을 가상의 헤드로 나눠서 시뮬레이션.
# -------------------------------------------------------

def _cp_a2a(Q_chunks, K_chunks, V_chunks, causal):
    """
    All-to-All (DeepSpeed-Ulysses): 시퀀스 분할 ↔ 헤드 분할 변환.

    핵심: multi-head attention에서 각 head는 독립적으로 attention 계산.
    → head를 GPU에 분배해도 수학적으로 동일!

    Input:  list of (chunk_seq, num_heads, head_dim) per GPU  [seq 분할]
    After all-to-all: (total_seq, heads_per_gpu, head_dim)    [head 분할]
    Attention 후 다시 all-to-all로 원래 layout 복원.

    실제 torch.distributed:
      # Forward all-to-all: seq-split → head-split
      input_list = Q_local.chunk(cp_size, dim=1)   # head 방향으로 나눔
      output_list = [torch.empty_like(input_list[0])] * cp_size
      dist.all_to_all(output_list, input_list, group=cp_group)
      Q_head_local = torch.cat(output_list, dim=0)  # seq 방향으로 합침
    """
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    num_heads = Q_chunks[0].shape[1]
    head_dim = Q_chunks[0].shape[2]
    total_seq = chunk_size * num_gpus

    assert num_heads % num_gpus == 0, \
        f"num_heads({num_heads}) must be divisible by num_gpus({num_gpus})"
    heads_per_gpu = num_heads // num_gpus

    # ===== Forward All-to-All: seq-split → head-split =====
    Q_full = torch.cat(Q_chunks, dim=0)  # (total_seq, num_heads, head_dim)
    K_full = torch.cat(K_chunks, dim=0)
    V_full = torch.cat(V_chunks, dim=0)

    all_head_outputs = []
    for gpu_id in range(num_gpus):
        h_start = gpu_id * heads_per_gpu
        h_end = h_start + heads_per_gpu

        # 이 GPU가 담당하는 head들 (전체 시퀀스)
        Q_h = Q_full[:, h_start:h_end, :]  # (total_seq, heads_per_gpu, head_dim)
        K_h = K_full[:, h_start:h_end, :]
        V_h = V_full[:, h_start:h_end, :]

        # 각 head에 대해 독립적으로 attention 계산
        head_outs = []
        for h in range(heads_per_gpu):
            q = Q_h[:, h, :]  # (total_seq, head_dim)
            k = K_h[:, h, :]
            v = V_h[:, h, :]

            scores = q @ k.T / math.sqrt(head_dim)
            if causal:
                mask = torch.tril(torch.ones(total_seq, total_seq))
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            head_outs.append(attn @ v)  # (total_seq, head_dim)

        all_head_outputs.append(torch.stack(head_outs, dim=1))  # (total_seq, heads_per_gpu, head_dim)

    # ===== Backward All-to-All: head-split → seq-split =====
    full_output = torch.cat(all_head_outputs, dim=1)  # (total_seq, num_heads, head_dim)
    return list(full_output.chunk(num_gpus, dim=0))  # list of (chunk_seq, num_heads, head_dim)


def _all_to_all_seq_to_head(x_local, group=None):
    """
    seq-shard(all heads) → full-seq(partial heads).

    입력:  (chunk_seq, num_heads, head_dim)
    출력:  (total_seq, heads_per_rank, head_dim)

    각 rank는 자기 seq chunk의 heads를 cp_size개로 나눠 다른 rank에 보낸다.
    받은 조각들은 seq 차원으로 concat하면 full sequence가 된다.
    """
    world_size = dist.get_world_size(group)
    num_heads = x_local.shape[1]
    assert num_heads % world_size == 0, \
        f"num_heads({num_heads}) must be divisible by cp_size({world_size})"

    # Tensor.chunk(chunks, dim)는 dim 차원을 chunks개 조각으로 나눈 tuple을 반환한다.
    # 여기서는 dim=1이 head 차원이므로, local tensor의 모든 heads를 rank 수만큼 쪼갠다.
    # 예: x_local.shape = (chunk_seq=4, num_heads=8, head_dim=64), world_size=4이면
    #     x_local.chunk(4, dim=1) → 4개 조각, 각 shape = (4, 2, 64)
    #     send_chunks[0] = 이 rank의 seq shard + heads 0-1
    #     send_chunks[1] = 이 rank의 seq shard + heads 2-3
    #     ...
    # all_to_all은 이 head 조각들을 각 rank로 보내서, 각 rank가 특정 head subset을 담당하게 만든다.
    send_chunks = [chunk.contiguous() for chunk in x_local.chunk(world_size, dim=1)]
    recv_chunks = [torch.empty_like(send_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(recv_chunks, send_chunks, group=group)
    return torch.cat(recv_chunks, dim=0)


def _all_to_all_head_to_seq(x_head_local, group=None):
    """
    full-seq(partial heads) → seq-shard(all heads).

    입력:  (total_seq, heads_per_rank, head_dim)
    출력:  (chunk_seq, num_heads, head_dim)

    각 rank가 가진 full sequence output을 seq chunk별로 나눠 원래 seq owner에게 돌려준다.
    받은 조각들은 head 차원으로 concat하면 local seq의 모든 heads가 복원된다.

    예: CP=4, H=8, heads_per_rank=2
      all_to_all 전 rank0: full seq, heads 0-1
      all_to_all 전 rank1: full seq, heads 2-3
      ...

      all_to_all 후 rank0이 받는 recv_chunks:
        recv_chunks[0] = seq 0-3, heads 0-1
        recv_chunks[1] = seq 0-3, heads 2-3
        recv_chunks[2] = seq 0-3, heads 4-5
        recv_chunks[3] = seq 0-3, heads 6-7

      따라서 dim=1(head 차원)으로 cat하면
        rank0 output = seq 0-3, heads 0-7
      즉 CP의 기본 layout인 seq-shard(all heads)가 복원된다.
    """
    world_size = dist.get_world_size(group)
    send_chunks = [chunk.contiguous() for chunk in x_head_local.chunk(world_size, dim=0)]
    recv_chunks = [torch.empty_like(send_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(recv_chunks, send_chunks, group=group)
    # recv_chunks는 같은 seq shard에 대한 heads_per_rank 조각들의 리스트.
    # dim=1이 head 차원이므로 여기서 붙여야 local seq의 all heads가 복원된다.
    return torch.cat(recv_chunks, dim=1)


def _cp_a2a_dist(Q_local, K_local, V_local, causal, group=None):
    """
    실제 torch.distributed A2A/Ulysses CP attention.

    입력 layout:
      Q/K/V local: (chunk_seq, num_heads, head_dim)  [seq-shard, all heads]

    A2A 후 attention layout:
      Q/K/V head-local: (total_seq, heads_per_rank, head_dim)  [full seq, partial heads]

    attention 후 다시 A2A로 local seq shard를 복원한다.
    """
    Q_head = _all_to_all_seq_to_head(Q_local.contiguous(), group)
    K_head = _all_to_all_seq_to_head(K_local.contiguous(), group)
    V_head = _all_to_all_seq_to_head(V_local.contiguous(), group)

    total_seq, heads_per_rank, head_dim = Q_head.shape
    head_outputs = []
    for h in range(heads_per_rank):
        q = Q_head[:, h, :]
        k = K_head[:, h, :]
        v = V_head[:, h, :]
        scores = q @ k.T / math.sqrt(head_dim)
        if causal:
            mask = torch.tril(torch.ones(total_seq, total_seq, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
        head_outputs.append(F.softmax(scores, dim=-1) @ v)

    output_head = torch.stack(head_outputs, dim=1)
    return _all_to_all_head_to_seq(output_head, group)


# ============================================================
# Part 2: 정확성 검증
# ============================================================

def verify_ring_attention():
    """모든 CP 통신 유형이 표준 attention과 동일한 결과를 내는지 검증."""
    print("=" * 60)
    print("Context Parallelism Verification (P2P / All-Gather / A2A)")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    total_seq = 16
    chunk_size = total_seq // num_gpus
    D = 8  # head_dim (num_gpus로 나눠떨어져야 A2A 가능)

    Q = torch.randn(total_seq, D)
    K = torch.randn(total_seq, D)
    V = torch.randn(total_seq, D)

    # --- 표준 attention (정답) ---
    scores = Q @ K.T / math.sqrt(D)
    out_standard = F.softmax(scores, dim=-1) @ V

    # --- 표준 causal attention (정답) ---
    causal_mask = torch.tril(torch.ones(total_seq, total_seq))
    scores_causal = scores.masked_fill(causal_mask == 0, float('-inf'))
    out_causal = F.softmax(scores_causal, dim=-1) @ V

    # chunk 분할
    Q_chunks = list(Q.chunk(num_gpus))
    K_chunks = list(K.chunk(num_gpus))
    V_chunks = list(V.chunk(num_gpus))

    # --- P2P, All-Gather 검증 (single-head) ---
    for comm_type in [CPCommType.P2P, CPCommType.ALL_GATHER]:
        print(f"\n  ── {comm_type.value.upper()} ──")

        # Bidirectional
        cp_outputs = ring_attention(Q_chunks, K_chunks, V_chunks,
                                    causal=False, comm_type=comm_type)
        out_cp = torch.cat(cp_outputs, dim=0)
        diff = (out_standard - out_cp).abs().max().item()
        print(f"    [Bidirectional] diff: {diff:.2e}  {'PASSED' if diff < 1e-5 else 'FAILED'}")

        # Causal
        cp_causal = ring_attention(Q_chunks, K_chunks, V_chunks,
                                   causal=True, comm_type=comm_type)
        out_cp_causal = torch.cat(cp_causal, dim=0)
        diff_c = (out_causal - out_cp_causal).abs().max().item()
        print(f"    [Causal]        diff: {diff_c:.2e}  {'PASSED' if diff_c < 1e-5 else 'FAILED'}")

    # --- A2A 검증 (multi-head, head 분할이므로 num_heads >= num_gpus 필요) ---
    print(f"\n  ── A2A (multi-head) ──")
    num_heads = 8  # num_gpus로 나눠떨어져야 함
    head_dim = 4

    Q_mh = torch.randn(total_seq, num_heads, head_dim)
    K_mh = torch.randn(total_seq, num_heads, head_dim)
    V_mh = torch.randn(total_seq, num_heads, head_dim)

    # 표준 multi-head attention (정답)
    out_mh_standard = []
    out_mh_causal = []
    for h in range(num_heads):
        s = Q_mh[:, h] @ K_mh[:, h].T / math.sqrt(head_dim)
        out_mh_standard.append(F.softmax(s, dim=-1) @ V_mh[:, h])
        s_c = s.masked_fill(torch.tril(torch.ones(total_seq, total_seq)) == 0, float('-inf'))
        out_mh_causal.append(F.softmax(s_c, dim=-1) @ V_mh[:, h])
    out_mh_standard = torch.stack(out_mh_standard, dim=1)  # (total_seq, num_heads, head_dim)
    out_mh_causal = torch.stack(out_mh_causal, dim=1)

    # A2A: seq 분할 → head 분할 → attention → head 분할 → seq 분할
    Q_mh_chunks = list(Q_mh.chunk(num_gpus, dim=0))
    K_mh_chunks = list(K_mh.chunk(num_gpus, dim=0))
    V_mh_chunks = list(V_mh.chunk(num_gpus, dim=0))

    a2a_out = _cp_a2a(Q_mh_chunks, K_mh_chunks, V_mh_chunks, causal=False)
    out_a2a = torch.cat(a2a_out, dim=0)
    diff_a2a = (out_mh_standard - out_a2a).abs().max().item()
    print(f"    [Bidirectional] diff: {diff_a2a:.2e}  {'PASSED' if diff_a2a < 1e-5 else 'FAILED'}")

    a2a_causal = _cp_a2a(Q_mh_chunks, K_mh_chunks, V_mh_chunks, causal=True)
    out_a2a_c = torch.cat(a2a_causal, dim=0)
    diff_a2a_c = (out_mh_causal - out_a2a_c).abs().max().item()
    print(f"    [Causal]        diff: {diff_a2a_c:.2e}  {'PASSED' if diff_a2a_c < 1e-5 else 'FAILED'}")
    print(f"    (num_heads={num_heads}, heads_per_gpu={num_heads//num_gpus})")

    # Causal에서 P2P의 skip 분석
    total_blocks = num_gpus * num_gpus
    skipped = 0
    for gpu_id in range(num_gpus):
        for step in range(num_gpus):
            kv_idx = (gpu_id + step) % num_gpus
            if kv_idx > gpu_id:
                skipped += 1
    print(f"\n  P2P Causal mask 최적화:")
    print(f"    전체 QK blocks: {total_blocks}")
    print(f"    Skip된 blocks:  {skipped} ({skipped/total_blocks*100:.0f}%)")
    print(f"    → causal mask로 거의 절반의 연산 절약!")


# ============================================================
# Part 3: 통신 분석
# ============================================================

def communication_analysis():
    """CP의 통신량과 overlap 분석."""
    print("\n" + "=" * 60)
    print("Context Parallelism Communication Analysis")
    print("=" * 60)

    cp_size = 4
    seq_len = 131072   # 128K
    num_heads = 32
    head_dim = 128
    num_layers = 32
    batch = 1
    bf16 = 2

    chunk_seq = seq_len // cp_size  # 32K per GPU

    # KV chunk size: (chunk_seq, num_heads, head_dim) × 2 (K+V)
    kv_chunk_bytes = chunk_seq * num_heads * head_dim * bf16 * 2
    # Ring에서 cp_size-1번 전송
    kv_per_layer = kv_chunk_bytes * (cp_size - 1)
    kv_total = kv_per_layer * num_layers

    # Attention scores per GPU: (chunk_seq, chunk_seq) per head per KV step
    attn_mem_standard = seq_len * seq_len * num_heads * bf16 / 1e9  # 전체
    attn_mem_cp = chunk_seq * chunk_seq * num_heads * bf16 / 1e9    # CP 시

    print(f"\n  Config: seq={seq_len}, CP={cp_size}, heads={num_heads}")
    print(f"  Chunk per GPU: {chunk_seq} tokens")

    print(f"\n  Attention memory (per layer, single head):")
    print(f"    No CP:  {seq_len}×{seq_len} = {seq_len**2/1e6:.0f}M entries"
          f" ({attn_mem_standard:.1f} GB total)")
    print(f"    CP={cp_size}: {chunk_seq}×{chunk_seq} = {chunk_seq**2/1e6:.0f}M entries"
          f" ({attn_mem_cp:.2f} GB per GPU)")
    print(f"    절약: {attn_mem_standard / attn_mem_cp:.0f}x (= CP_size^2)")

    print(f"\n  Communication (per layer):")
    print(f"    KV chunk: {kv_chunk_bytes / 1e6:.1f} MB")
    print(f"    Ring steps: {cp_size - 1}")
    print(f"    Total: {kv_per_layer / 1e6:.1f} MB per layer")
    print(f"    All layers: {kv_total / 1e9:.2f} GB per step")

    print(f"\n  Overlap 전략:")
    print( "    Step i:  GPU에서 KV_i로 attention 계산")
    print( "             동시에 KV_{i+1}을 다음 GPU로부터 recv")
    print( "    → 통신이 연산에 완전히 숨겨짐 (연산 > 통신이면)")

    # 연산 vs 통신 비교
    # attention 연산: 2 * chunk_seq * chunk_seq * head_dim * num_heads
    compute_flops = 2 * chunk_seq * chunk_seq * head_dim * num_heads * batch
    compute_tflops = compute_flops / 1e12
    transfer_gb = kv_chunk_bytes / 1e9

    print(f"\n  연산 vs 통신 (per ring step, per layer):")
    print(f"    Compute: {compute_tflops:.2f} TFLOP")
    print(f"    Transfer: {transfer_gb:.3f} GB")
    print(f"    H100 기준: {compute_tflops*1000/990:.1f}ms compute"
          f" vs {transfer_gb*1000/900:.1f}ms transfer (NVLink)")
    print(f"    → {'Compute-bound (overlap 가능!)' if compute_tflops/990 > transfer_gb/900 else 'Transfer-bound'}")

    # ========== 통신 유형별 비교 ==========
    print(f"\n  {'─' * 56}")
    print(f"  CP 통신 유형별 비교 (seq={seq_len}, CP={cp_size})")
    print(f"  {'─' * 56}")

    # QKV 한 세트: (chunk_seq, num_heads, head_dim) × 3
    qkv_chunk_bytes = chunk_seq * num_heads * head_dim * bf16 * 3

    # --- P2P (Ring) ---
    p2p_per_step = kv_chunk_bytes  # send KV chunk to neighbor
    p2p_total_per_layer = p2p_per_step * (cp_size - 1)
    p2p_peak_mem = kv_chunk_bytes  # KV chunk 1개만 추가 보유

    # --- All-Gather ---
    ag_total_per_layer = kv_chunk_bytes * (cp_size - 1)  # all_gather = 각자 chunk 보냄
    ag_peak_mem = kv_chunk_bytes * cp_size  # 전체 KV 보유

    # --- A2A (Ulysses) ---
    # forward: Q,K,V 각각 all-to-all (seq→head), backward: output all-to-all (head→seq)
    a2a_per_call = qkv_chunk_bytes * (cp_size - 1) / cp_size  # all-to-all 전송량
    a2a_total_per_layer = a2a_per_call * 2  # forward + backward all-to-all
    a2a_peak_mem = chunk_seq * num_heads * head_dim * bf16  # 추가 메모리 적음

    print(f"""
  ┌──────────────┬───────────────┬───────────────┬──────────────────────┐
  │ 유형          │ 통신량/layer   │ Peak 추가 메모리│ Overlap 가능 여부     │
  ├──────────────┼───────────────┼───────────────┼──────────────────────┤
  │ P2P (Ring)   │ {p2p_total_per_layer/1e6:>8.1f} MB │ {p2p_peak_mem/1e6:>8.1f} MB │ Yes (compute+comm)  │
  │ All-Gather   │ {ag_total_per_layer/1e6:>8.1f} MB │ {ag_peak_mem/1e6:>8.1f} MB │ No (gather→compute) │
  │ A2A (Ulysses)│ {a2a_total_per_layer/1e6:>8.1f} MB │ {a2a_peak_mem/1e6:>8.1f} MB │ Partial             │
  └──────────────┴───────────────┴───────────────┴──────────────────────┘

  선택 가이드:
    P2P (Ring):    CP_size가 크고, 긴 시퀀스에서 overlap 효과 극대화
                   Megatron-Core, PyTorch FSDP2 기본 방식
    All-Gather:    CP_size 작고 (2~4), 단순 구현 원할 때
                   메모리 여유 있으면 가장 쉬운 선택
    A2A (Ulysses): head 수가 충분하고, 노드 내 NVLink에서 효율적
                   DeepSpeed-Ulysses, Megatron-Core (--cp-comm-type=a2a)""")


# ============================================================
# Part 4: 4D Parallelism (TP × CP × PP × DP)
# ============================================================

def parallelism_4d():
    """4D parallelism 구성."""
    print("\n" + "=" * 60)
    print("4D Parallelism: TP × CP × PP × DP")
    print("=" * 60)

    print("""
  최신 대규모 학습은 4D parallelism 사용:

    Total GPUs = TP × CP × PP × DP

  예: LLaMA-3 405B on 16384 GPUs
    TP = 8    (노드 내 NVLink, layer 내부 weight split)
    CP = 2    (시퀀스를 2등분, ring attention)
    PP = 16   (32 layers를 16 stage로)
    DP = 64   (데이터 병렬)
    → 8 × 2 × 16 × 64 = 16,384 GPUs

  각 parallelism이 나누는 차원:
    ┌──────────┬────────────┬──────────────────────────┐
    │          │ 나누는 것   │ 통신                      │
    ├──────────┼────────────┼──────────────────────────┤
    │ TP       │ hidden dim │ all-reduce (NVLink)       │
    │ CP       │ seq dim    │ ring send/recv (NVLink)   │
    │ PP       │ layers     │ point-to-point (IB 가능)  │
    │ DP       │ data       │ all-reduce (IB)           │
    └──────────┴────────────┴──────────────────────────┘

  DeviceMesh 구성:
    mesh = init_device_mesh("cuda",
        (dp_size, pp_size, cp_size, tp_size),
        mesh_dim_names=("dp", "pp", "cp", "tp"),
    )

  Megatron-Core에서:
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=8,
        context_parallel_size=2,
        pipeline_model_parallel_size=16,
        # DP는 자동 계산
    )
    """)

    # GPU 배치 예시
    tp, cp, pp, dp = 2, 2, 2, 2
    total = tp * cp * pp * dp
    print(f"  예시: {tp}TP × {cp}CP × {pp}PP × {dp}DP = {total} GPUs\n")

    print(f"  {'GPU':>4} {'TP':>4} {'CP':>4} {'PP':>4} {'DP':>4}   역할")
    print(f"  {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}   {'-'*30}")

    for rank in range(total):
        tp_r = rank % tp
        cp_r = (rank // tp) % cp
        pp_r = (rank // (tp * cp)) % pp
        dp_r = rank // (tp * cp * pp)

        roles = []
        if tp_r == 0 and cp_r == 0:
            roles.append(f"PP stage {pp_r}")
        if tp_r == 0:
            roles.append(f"seq chunk {cp_r}")

        role = ", ".join(roles) if roles else ""
        print(f"  {rank:>4} {tp_r:>4} {cp_r:>4} {pp_r:>4} {dp_r:>4}   {role}")


# ============================================================
# Part 5: CP vs 다른 접근법 비교
# ============================================================

def cp_comparison():
    print("\n" + "=" * 60)
    print("Context Parallelism vs Alternatives")
    print("=" * 60)

    print("""
  긴 시퀀스를 처리하는 방법들:

  ┌───────────────────┬────────────┬────────────┬──────────────────┐
  │ 방법               │ 메모리     │ 정확도     │ 구현 복잡도       │
  ├───────────────────┼────────────┼────────────┼──────────────────┤
  │ Flash Attention    │ O(S)      │ 정확       │ 낮음 (라이브러리) │
  │ Sliding Window     │ O(S×W)    │ 근사       │ 낮음             │
  │ Gradient Ckpt      │ 절반      │ 정확       │ 낮음             │
  │ Context Parallel   │ O(S²/N²)  │ 정확       │ 높음             │
  │ Ring Attention     │ O(S²/N²)  │ 정확       │ 높음             │
  └───────────────────┴────────────┴────────────┴──────────────────┘

  조합해서 사용:
    Flash Attention + CP가 가장 효과적
      Flash: 단일 GPU 내에서 O(S) 메모리
      CP:    GPU 간에 시퀀스 분산 → 각 GPU의 S가 S/N으로 감소

  언제 CP를 쓰나?
    - seq_len > 32K 이상일 때 (activation 메모리 문제)
    - Flash Attention만으로 부족할 때
    - 예: 128K context → CP=4 → 각 GPU 32K → Flash로 충분히 처리
    """)


if __name__ == "__main__":
    verify_ring_attention()
    communication_analysis()
    parallelism_4d()
    cp_comparison()
