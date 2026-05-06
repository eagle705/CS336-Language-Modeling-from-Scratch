"""Sequence Parallelism (SP)
===========================
TP와 결합하여 LayerNorm, Dropout 등의 activation도 시퀀스 차원으로 분산.

문제: TP만 쓰면 LayerNorm, Dropout은 여전히 전체 시퀀스를 각 GPU가 중복 보유.
     activation 메모리의 상당 부분이 이 non-TP 영역에서 발생.

TP만 적용 (SP 없음):
    모든 GPU가 동일한 전체 activation 보유 (중복!)

    GPU 0: [LayerNorm(전체 seq)] → [TP fc1(절반)] → [TP fc2(절반)] → [LayerNorm(전체 seq)]
    GPU 1: [LayerNorm(전체 seq)] → [TP fc1(절반)] → [TP fc2(절반)] → [LayerNorm(전체 seq)]
                  ↑ 중복!                                                    ↑ 중복!

TP + SP (Megatron-LM):
    non-TP 영역은 seq 차원으로 split, TP 영역에서만 전체 seq 복원.

    GPU 0: [LN(seq 앞절반)] → gather → [TP fc1] → [TP fc2] → scatter → [LN(seq 앞절반)]
    GPU 1: [LN(seq 뒷절반)] → gather → [TP fc1] → [TP fc2] → scatter → [LN(seq 뒷절반)]
                                 ↑ all-gather          ↑ reduce-scatter

통신 변화:
    TP only:  all-reduce = reduce-scatter + all-gather (2번)
    TP + SP:  reduce-scatter 1번 + all-gather 1번 (총량 동일! 위치만 다름)
    → 통신 총량은 같지만, activation 메모리가 1/TP로 감소!

    ┌──────────────────────────────────────────────────────────────────┐
    │  TP only (forward):                                             │
    │                                                                 │
    │  [LN] ──→ [ColumnParallel] ──→ [RowParallel] ──all-reduce──→   │
    │   ↑ 전체 seq                                    ↑ 전체 seq      │
    │                                                                 │
    │  TP + SP (forward):                                             │
    │                                                                 │
    │  [LN] ─all-gather─→ [ColParallel] ─→ [RowParallel] ─r-scatter→ │
    │   ↑ seq/TP                                           ↑ seq/TP   │
    │                                                                 │
    │  통신 총량 동일, activation 메모리 1/TP 절약!                     │
    └──────────────────────────────────────────────────────────────────┘

인터뷰 포인트:
  1. SP는 TP의 all-reduce를 (all-gather + reduce-scatter)로 분리
  2. 통신량은 동일하지만, non-TP 영역의 activation이 1/TP로 감소
  3. Megatron-Core에서 sequence_parallel=True 한 줄로 활성화

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class _AllGatherFromSP(torch.autograd.Function):
    """Forward: all-gather (seq/TP → 전체 seq)
Backward: reduce-scatter (gradient를 seq/TP로 분산)"""

    @staticmethod
    def forward(ctx, x, tp_size):
        raise NotImplementedError('TODO: implement _AllGatherFromSP.forward; compare with solution.py only after trying.')

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError('TODO: implement _AllGatherFromSP.backward; compare with solution.py only after trying.')

class _ReduceScatterToSP(torch.autograd.Function):
    """Forward: reduce-scatter (전체 seq → seq/TP)
Backward: all-gather (seq/TP gradient → 전체 seq gradient)"""

    @staticmethod
    def forward(ctx, x, tp_size):
        raise NotImplementedError('TODO: implement _ReduceScatterToSP.forward; compare with solution.py only after trying.')

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError('TODO: implement _ReduceScatterToSP.backward; compare with solution.py only after trying.')

def simulate_sequence_parallelism():
    """TP + SP의 activation 분산을 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_sequence_parallelism; compare with solution.py only after trying.')

def memory_analysis():
    """실제 모델 규모에서 SP의 메모리 절약 효과."""
    raise NotImplementedError('TODO: implement memory_analysis; compare with solution.py only after trying.')

def communication_comparison():
    raise NotImplementedError('TODO: implement communication_comparison; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_sequence_parallelism()
    memory_analysis()
    communication_comparison()
