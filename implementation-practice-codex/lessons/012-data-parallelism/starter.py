"""Data Parallelism (DP) for MLP
================================
모든 GPU가 동일한 모델을 가지고, 다른 데이터로 학습.

핵심 아이디어:
  - 각 GPU가 전체 모델의 복사본 보유
  - batch를 GPU 수만큼 나눠서 각 GPU에 분배
  - 각 GPU가 독립적으로 forward/backward
  - gradient를 all-reduce로 평균 → 모든 GPU가 동일 update

    Data batch: [B0, B1, B2, B3]
                  |   |   |   |
                  v   v   v   v
    GPU 0: Model(B0)  GPU 1: Model(B1)  GPU 2: Model(B2)  GPU 3: Model(B3)
       ↓ grad_0          ↓ grad_1          ↓ grad_2          ↓ grad_3
       └─────────── All-Reduce (mean) ──────────────┘
                         ↓
                 avg_grad = mean(grad_0..3)
                         ↓
                 모든 GPU가 동일한 update
                 → 모델이 항상 동기화 상태

통신 패턴:
  All-Reduce: 2 * model_size (reduce-scatter + all-gather)
  통신은 backward 중에 overlap 가능 (DDP bucketing)

인터뷰 포인트:
  1. DP는 가장 단순하고 확장성 좋은 parallelism
  2. 한계: 모델이 1개 GPU 메모리에 들어가야 함 → 큰 모델은 FSDP/ZeRO
  3. Effective batch size = micro_batch × num_gpus
  4. gradient all-reduce는 backward과 overlap 가능

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def simulate_ddp():
    """DDP의 forward → backward → all-reduce → update를 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_ddp; compare with solution.py only after trying.')

def ddp_vs_dp():
    """nn.DataParallel vs DistributedDataParallel 비교."""
    raise NotImplementedError('TODO: implement ddp_vs_dp; compare with solution.py only after trying.')

def simulate_gradient_accumulation():
    """Gradient Accumulation: micro-batch를 여러 번 backward 후 한 번 update.

용도: GPU 메모리에 큰 batch가 안 들어갈 때
effective_batch = micro_batch × accum_steps × num_gpus"""
    raise NotImplementedError('TODO: implement simulate_gradient_accumulation; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_ddp()
    ddp_vs_dp()
    simulate_gradient_accumulation()
