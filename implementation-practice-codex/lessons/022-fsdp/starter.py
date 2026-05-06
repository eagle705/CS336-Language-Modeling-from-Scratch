"""FSDP (Fully Sharded Data Parallel)
=====================================
PyTorch 네이티브 ZeRO-3 구현. 모든 모델 상태를 GPU에 분산.

FSDP vs DDP:
  DDP:  각 GPU가 전체 모델 복사본 보유 + gradient all-reduce
  FSDP: 모델을 shard로 쪼개서 분산 + 필요할 때만 all-gather

FSDP 동작 (각 FSDP unit = 보통 1개 Transformer block):
  ┌──────────────────────────────────────────────────────┐
  │ Forward:                                             │
  │   all-gather params → forward 계산 → params 해제     │
  │                                                      │
  │ Backward:                                            │
  │   all-gather params → backward 계산 →                │
  │   reduce-scatter grads → params 해제                 │
  │                                                      │
  │ Optimizer step:                                      │
  │   각 GPU가 자기 shard만 update (local operation)      │
  └──────────────────────────────────────────────────────┘

FSDP1 vs FSDP2:
  FSDP1 (torch.distributed.fsdp.FullyShardedDataParallel):
    - FlatParameter: 여러 params를 하나로 flatten → 통신 효율적
    - 단점: flatten 때문에 디버깅 어렵고 유연성 부족

  FSDP2 (torch.distributed.fsdp.fully_shard, PyTorch 2.x):
    - DTensor 기반: 각 param이 독립적인 DTensor
    - per-parameter sharding → 더 유연하고 디버깅 쉬움
    - DeviceMesh와 자연스럽게 통합

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import os
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def simulate_fsdp():
    """GPU 없이 FSDP의 shard/gather 동작을 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_fsdp; compare with solution.py only after trying.')

class TinyFSDPBlock(nn.Module):
    """FSDP로 감쌀 작은 Transformer FFN block."""

    def __init__(self, embed_dim, hidden_dim):
        raise NotImplementedError('TODO: implement TinyFSDPBlock.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement TinyFSDPBlock.forward; compare with solution.py only after trying.')

class TinyFSDPModel(nn.Module):
    """GPU smoke test용 작은 decoder-like 모델."""

    def __init__(self, vocab_size=256, embed_dim=64, hidden_dim=256, num_layers=2):
        raise NotImplementedError('TODO: implement TinyFSDPModel.__init__; compare with solution.py only after trying.')

    def forward(self, input_ids):
        raise NotImplementedError('TODO: implement TinyFSDPModel.forward; compare with solution.py only after trying.')

def _find_free_port():
    raise NotImplementedError('TODO: implement _find_free_port; compare with solution.py only after trying.')

def _init_cuda_dist():
    """torchrun 또는 단일 GPU 직접 실행 모두 지원하는 process group 초기화."""
    raise NotImplementedError('TODO: implement _init_cuda_dist; compare with solution.py only after trying.')

def run_fsdp_gpu_smoke_test():
    """
    실제 CUDA GPU에서 PyTorch FSDP forward/backward/optimizer step을 확인.

    단일 GPU:
      python implementation-practice-codex/lessons/022-fsdp/starter.py gpu

    멀티 GPU:
      torchrun --nproc_per_node=2 implementation-practice-codex/lessons/022-fsdp/starter.py gpu
    """
    raise NotImplementedError('TODO: implement run_fsdp_gpu_smoke_test; compare with solution.py only after trying.')

def memory_comparison():
    raise NotImplementedError('TODO: implement memory_comparison; compare with solution.py only after trying.')
if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'simulate'
    if mode == 'gpu':
        run_fsdp_gpu_smoke_test()
    else:
        simulate_fsdp()
        memory_comparison()
