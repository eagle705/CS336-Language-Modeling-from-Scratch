"""Mixed Precision Training
=========================
FP32 / FP16 / BF16을 섞어서 학습 → 속도 2x 향상 + 메모리 절약.

숫자 표현 비교:
  FP32:  1 sign + 8 exp + 23 mantissa = 32 bits  (기본)
  FP16:  1 sign + 5 exp + 10 mantissa = 16 bits  (범위 좁음, overflow 위험)
  BF16:  1 sign + 8 exp +  7 mantissa = 16 bits  (FP32과 같은 범위, 정밀도 낮음)

  FP16 범위: ±65504       → gradient가 이 범위 밖이면 overflow/underflow
  BF16 범위: ±3.4 × 10^38 → FP32와 동일한 범위, overflow 걱정 없음
  → 최근 모델은 대부분 BF16 사용 (loss scaling 불필요)

Mixed Precision의 3가지 규칙:
  1. Forward/Backward는 FP16/BF16으로 (빠른 연산)
  2. Weight master copy는 FP32로 유지 (정밀도 보존)
  3. Loss scaling (FP16만): gradient underflow 방지

    ┌──────────────────────────────────────────────────┐
    │  FP32 master weights                             │
    │       │ copy to FP16                             │
    │       ▼                                          │
    │  FP16 forward → FP16 loss → scale loss           │
    │       │                                          │
    │  FP16 backward (scaled gradients)                │
    │       │ unscale + clip                           │
    │       ▼                                          │
    │  FP32 optimizer step (master weights update)     │
    └──────────────────────────────────────────────────┘

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn

def explore_dtypes():
    """각 dtype의 범위와 정밀도를 직접 확인."""
    raise NotImplementedError('TODO: implement explore_dtypes; compare with solution.py only after trying.')

def manual_mixed_precision():
    """AMP 없이 mixed precision을 수동으로 구현."""
    raise NotImplementedError('TODO: implement manual_mixed_precision; compare with solution.py only after trying.')

def pytorch_amp_example():
    """torch.amp를 사용한 mixed precision (실전 코드)."""
    raise NotImplementedError('TODO: implement pytorch_amp_example; compare with solution.py only after trying.')

def memory_analysis():
    """Mixed precision의 메모리 절약 효과 계산."""
    raise NotImplementedError('TODO: implement memory_analysis; compare with solution.py only after trying.')

def fp8_info():
    """FP8 format 비교."""
    raise NotImplementedError('TODO: implement fp8_info; compare with solution.py only after trying.')

def fp4_info():
    """FP4/NVFP4 정보."""
    raise NotImplementedError('TODO: implement fp4_info; compare with solution.py only after trying.')
if __name__ == '__main__':
    explore_dtypes()
    manual_mixed_precision()
    pytorch_amp_example()
    memory_analysis()
    fp8_info()
    fp4_info()
