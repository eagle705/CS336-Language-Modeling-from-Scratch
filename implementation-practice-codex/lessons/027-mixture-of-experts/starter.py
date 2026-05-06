"""Mixture of Experts (MoE)
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

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """각 토큰에 대해 top-k expert를 선택.

router_logits = x @ W_gate          # (batch*seq, num_experts)
router_probs  = softmax(router_logits)
top_k_probs, top_k_indices = topk(router_probs, k)

Load Balancing Loss:
  expert별 처리량이 균등하도록 auxiliary loss 추가.
  없으면 일부 expert에 토큰이 몰리는 "winner-take-all" 문제 발생.

  aux_loss = num_experts * sum_i(f_i * p_i)
    f_i = (expert i에 배정된 토큰 비율)
    p_i = (expert i의 평균 gate probability)
  → 균등 분배면 aux_loss = 1, 쏠리면 > 1"""

    def __init__(self, embed_dim, num_experts, top_k=2):
        raise NotImplementedError('TODO: implement TopKRouter.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement TopKRouter.forward; compare with solution.py only after trying.')

class Expert(nn.Module):
    """단일 expert = 일반 FFN."""

    def __init__(self, embed_dim, hidden_dim):
        raise NotImplementedError('TODO: implement Expert.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement Expert.forward; compare with solution.py only after trying.')

class MoELayer(nn.Module):
    """Mixture of Experts layer.

동작:
1. Router가 각 토큰의 top-k expert 선택
2. 각 expert에 해당 토큰들을 보내서 계산
3. 결과를 gate score로 가중합"""

    def __init__(self, embed_dim, hidden_dim, num_experts=8, top_k=2):
        raise NotImplementedError('TODO: implement MoELayer.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement MoELayer.forward; compare with solution.py only after trying.')

def demo():
    raise NotImplementedError('TODO: implement demo; compare with solution.py only after trying.')
if __name__ == '__main__':
    demo()
