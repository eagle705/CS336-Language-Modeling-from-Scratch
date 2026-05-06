"""
Mixture of Experts (MoE)
=========================
FFN을 여러 개의 "expert"로 나누고, router가 토큰별로 expert를 선택.

핵심 아이디어:
  - 모델 파라미터는 크지만, 각 토큰은 일부 expert만 활성화 → 연산량 제어
  - 예: 8 experts 중 top-2만 사용 → params 8x, FLOPs ~2x

구조:
  Input token
      │
      ▼
  ┌─────────┐
  │  Router  │ → softmax(x @ W_gate) → top-k expert 선택
  └─────────┘
      │ gate scores (top-k)
      ▼
  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
  │Expert0│ │Expert1│ │Expert2│ │Expert3│ ...
  └───────┘ └───────┘ └───────┘ └───────┘
      │ 선택된 expert들의 output
      ▼
  weighted sum (gate score로 가중합)
      │
      ▼
  Output

인터뷰 포인트:
  1. Router의 load balancing (expert 골고루 사용하게)
  2. Expert parallelism (expert를 다른 GPU에 배치)
  3. 통신: all-to-all (토큰을 담당 expert가 있는 GPU로 전송)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        f_i = (expert i에 배정된 토큰 비율)
        p_i = (expert i의 평균 gate probability)
      → 균등 분배면 aux_loss = 1, 쏠리면 > 1
    """

    def __init__(self, embed_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x):
        # x: (batch * seq, embed_dim)
        logits = self.gate(x)                         # (batch*seq, num_experts)
        probs = F.softmax(logits, dim=-1)             # (batch*seq, num_experts)

        # top-k 선택
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # top_k_probs:   (batch*seq, top_k)  → 선택된 expert의 gate 점수
        # top_k_indices: (batch*seq, top_k)  → 선택된 expert의 인덱스

        # gate 정규화: 선택된 expert들의 확률 합이 1이 되도록
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Load balancing loss 계산
        # f_i: expert i에 배정된 토큰 비율
        num_tokens = x.shape[0]
        expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1)  # (tokens, experts)
        f = expert_mask.float().mean(dim=0)  # (num_experts,) 각 expert의 토큰 비율

        # p_i: expert i의 평균 gate probability
        p = probs.mean(dim=0)  # (num_experts,)

        # aux_loss: f와 p의 내적 → 균등하면 최소
        aux_loss = self.num_experts * (f * p).sum()

        return top_k_probs, top_k_indices, aux_loss


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
    2. 각 expert에 해당 토큰들을 보내서 계산
    3. 결과를 gate score로 가중합
    """

    def __init__(self, embed_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.router = TopKRouter(embed_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        B, S, D = x.shape
        x_flat = x.view(B * S, D)  # (tokens, embed_dim)

        # (1) Router: 각 토큰의 top-k expert 결정
        top_k_probs, top_k_indices, aux_loss = self.router(x_flat)
        # top_k_probs:   (tokens, top_k)
        # top_k_indices: (tokens, top_k)

        # (2) 각 expert별로 해당 토큰 모아서 계산
        output = torch.zeros_like(x_flat)  # (tokens, embed_dim)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]     # (tokens,) 각 토큰의 k번째 expert
            gate_scores = top_k_probs[:, k]          # (tokens,) 해당 gate 점수

            for expert_id in range(len(self.experts)):
                # 이 expert에 배정된 토큰 마스크
                mask = (expert_indices == expert_id)
                if mask.any():
                    tokens_for_expert = x_flat[mask]             # 해당 토큰들
                    expert_output = self.experts[expert_id](tokens_for_expert)
                    # gate score로 가중합
                    output[mask] += gate_scores[mask].unsqueeze(-1) * expert_output

        return output.view(B, S, D), aux_loss


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
# 통신 패턴 (all-to-all):
#   1. Router가 토큰별 expert 결정
#   2. All-to-All: 토큰을 해당 expert가 있는 GPU로 전송
#      GPU 0의 토큰 중 Expert 2,3에 배정된 것 → GPU 1로
#      GPU 1의 토큰 중 Expert 0,1에 배정된 것 → GPU 0로
#   3. 각 GPU에서 local expert 계산
#   4. All-to-All: 결과를 원래 GPU로 되돌림
#
# all-to-all: 각 GPU가 다른 모든 GPU에 서로 다른 데이터를 전송
#   (all-reduce와 달리 각 GPU가 보내는/받는 데이터가 다름)
#
# dist.all_to_all(output_list, input_list)
#   input_list[i]  → GPU i로 보낼 데이터
#   output_list[i] ← GPU i로부터 받은 데이터


# ============================================================
# Part 5: Demo
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

    print(f"\n  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f} (이상적 = 1.0)")

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
