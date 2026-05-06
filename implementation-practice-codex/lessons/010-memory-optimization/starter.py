"""Memory Optimization
====================
GPU 메모리 사용을 최적화하는 기법들.

GPU 메모리 구성 (학습 시):
  ┌─────────────────────────────────┐
  │ Model Parameters (weights)      │  → FSDP/ZeRO로 분산
  │ Optimizer States (Adam m, v)    │  → ZeRO-1+로 분산
  │ Gradients                       │  → ZeRO-2+로 분산
  │ Activations (forward 중간값)     │  → 이 파일에서 다루는 핵심!
  │ Temporary buffers               │
  └─────────────────────────────────┘

Activation 메모리가 왜 문제?
  - batch_size, seq_len에 비례하여 증가
  - 각 layer의 중간 결과를 backward까지 보관해야 함
  - 예: 7B model, seq=2048, batch=32 → activation ~50GB

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn

def activation_memory_analysis():
    """Transformer layer별 activation 메모리 상세 분석."""
    raise NotImplementedError('TODO: implement activation_memory_analysis; compare with solution.py only after trying.')

class CheckpointedBlock(nn.Module):
    """Gradient Checkpointing 원리:

일반:    forward 시 모든 중간값 저장 → backward에서 사용
체크포인트: forward 시 중간값 버림 → backward에서 다시 forward 계산

트레이드오프:
  메모리: O(L) → O(sqrt(L))  (L개 중 sqrt(L)개만 저장)
  연산:   1x → ~1.33x forward  (backward에서 재계산)

PyTorch API:
  torch.utils.checkpoint.checkpoint(fn, *args)"""

    def __init__(self, dim):
        raise NotImplementedError('TODO: implement CheckpointedBlock.__init__; compare with solution.py only after trying.')

    def _inner(self, x):
        """체크포인트로 감쌀 함수. forward 중간값이 해제됨."""
        raise NotImplementedError('TODO: implement CheckpointedBlock._inner; compare with solution.py only after trying.')

    def forward(self, x, use_checkpoint=False):
        raise NotImplementedError('TODO: implement CheckpointedBlock.forward; compare with solution.py only after trying.')

def demo_gradient_checkpointing():
    """체크포인트 유무에 따른 메모리 비교."""
    raise NotImplementedError('TODO: implement demo_gradient_checkpointing; compare with solution.py only after trying.')

def other_optimizations():
    raise NotImplementedError('TODO: implement other_optimizations; compare with solution.py only after trying.')
if __name__ == '__main__':
    activation_memory_analysis()
    demo_gradient_checkpointing()
    other_optimizations()
