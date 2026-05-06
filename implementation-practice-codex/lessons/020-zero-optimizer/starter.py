"""ZeRO (Zero Redundancy Optimizer) Stage 1 / 2 / 3
====================================================
핵심: 모든 GPU가 전체 모델 상태를 중복 저장하는 것을 제거.

DDP의 문제:
  각 GPU가 model weights + optimizer states + gradients를 전부 보유
  → N개 GPU여도 메모리 사용량은 1개 GPU와 동일

ZeRO 해결책: 모델 상태를 GPU들에 분산(partition)

  ┌──────────────────────────────────────────────────────┐
  │               각 GPU당 메모리 (1B params, Adam)       │
  │                                                      │
  │  Component            DDP    ZeRO-1  ZeRO-2  ZeRO-3 │
  │  ─────────────────── ────── ─────── ─────── ─────── │
  │  Optimizer states     8 GB   8/N GB  8/N GB  8/N GB │
  │  Gradients            4 GB   4 GB    4/N GB  4/N GB │
  │  Parameters           4 GB   4 GB    4 GB    4/N GB │
  │  ─────────────────── ────── ─────── ─────── ─────── │
  │  Total (N=4)         16 GB   6 GB    5 GB    4 GB   │
  └──────────────────────────────────────────────────────┘

  ZeRO-1: Optimizer states만 분산
  ZeRO-2: + Gradients도 분산
  ZeRO-3: + Parameters도 분산 (= FSDP와 동일 개념)

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import numpy as np

def simulate_zero_stage1():
    """ZeRO-1: Optimizer states를 GPU에 분산.

동작:
1. Forward/Backward: 일반 DDP와 동일 (all-reduce gradients)
2. Optimizer step: 각 GPU가 자기 담당 파라미터만 update
3. All-gather: 업데이트된 파라미터를 모든 GPU에 broadcast

통신량: DDP와 동일 (all-reduce gradients)
메모리 절약: optimizer states만 1/N"""
    raise NotImplementedError('TODO: implement simulate_zero_stage1; compare with solution.py only after trying.')

def simulate_zero_stage2():
    """ZeRO-2: Optimizer states + Gradients 분산.

ZeRO-1과 차이:
- All-reduce 대신 Reduce-scatter 사용
- 각 GPU가 담당 파라미터의 gradient만 보유 (나머지 버림)

통신량: DDP와 동일 (reduce-scatter = all-reduce의 절반 + all-gather의 절반)
메모리 절약: optimizer states + gradients 모두 1/N"""
    raise NotImplementedError('TODO: implement simulate_zero_stage2; compare with solution.py only after trying.')

def simulate_zero_stage3():
    """ZeRO-3: 모든 것(params + grads + optimizer)을 분산. = FSDP와 동일 개념.

핵심: forward/backward 시에도 파라미터를 필요할 때만 all-gather로 모음.

동작:
1. Forward의 각 layer:
   - all-gather로 해당 layer params 수집
   - forward 계산
   - 사용 끝난 params 버림 (메모리 해제)
2. Backward의 각 layer:
   - all-gather로 해당 layer params 수집 (다시!)
   - backward 계산
   - reduce-scatter로 gradient 분산
   - params 다시 버림

통신량: forward에 all-gather 추가 (DDP 대비 1.5x 통신)
메모리 절약: 모든 것이 1/N → 모델 크기에 비례하여 GPU 추가 가능"""
    raise NotImplementedError('TODO: implement simulate_zero_stage3; compare with solution.py only after trying.')

def comparison_table():
    raise NotImplementedError('TODO: implement comparison_table; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_zero_stage1()
    simulate_zero_stage2()
    simulate_zero_stage3()
    comparison_table()
