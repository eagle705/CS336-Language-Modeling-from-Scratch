"""Distributed Optimizer
=======================
Optimizer states를 DP ranks에 분산하여 메모리 절약.

문제: Adam optimizer는 param당 2개 추가 state (m, v) 보유 → FP32 기준 8x overhead
  7B model: params 14GB (BF16) + optimizer 56GB (FP32 m,v) + master 28GB (FP32)
  → optimizer가 메모리의 대부분!

해결: optimizer states를 DP group에 분산 (= ZeRO Stage 1)

일반 DDP:
  모든 GPU: [full params] + [full grads] + [full opt states (m, v)]

Distributed Optimizer (ZeRO-1):
  GPU 0: [full params] + [full grads] + [opt states for params 0-24%]
  GPU 1: [full params] + [full grads] + [opt states for params 25-49%]
  GPU 2: [full params] + [full grads] + [opt states for params 50-74%]
  GPU 3: [full params] + [full grads] + [opt states for params 75-100%]

Megatron Distributed Optimizer (ZeRO-1 + 최적화):
  DDP의 gradient all-reduce 대신:
  1. reduce-scatter: 각 GPU가 담당 파라미터의 gradient만 받음
  2. 담당 파라미터만 optimizer step (FP32)
  3. all-gather: 업데이트된 파라미터를 전체 GPU에 broadcast

  일반 DDP:           all-reduce(grads) → full optimizer step
  Dist Optimizer:     reduce-scatter(grads) → partial step → all-gather(params)
  통신량: 동일! (reduce-scatter + all-gather = all-reduce)

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import numpy as np

def simulate_distributed_optimizer():
    """Distributed Optimizer의 동작을 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_distributed_optimizer; compare with solution.py only after trying.')

def communication_comparison():
    """DDP vs Distributed Optimizer의 통신 패턴."""
    raise NotImplementedError('TODO: implement communication_comparison; compare with solution.py only after trying.')

def megatron_dist_optimizer():
    """Megatron-Core의 Distributed Optimizer 구현 상세."""
    raise NotImplementedError('TODO: implement megatron_dist_optimizer; compare with solution.py only after trying.')

def memory_analysis():
    """Distributed Optimizer의 메모리 절약 효과."""
    raise NotImplementedError('TODO: implement memory_analysis; compare with solution.py only after trying.')

def comparison_with_zero():
    raise NotImplementedError('TODO: implement comparison_with_zero; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_distributed_optimizer()
    communication_comparison()
    megatron_dist_optimizer()
    memory_analysis()
    comparison_with_zero()
