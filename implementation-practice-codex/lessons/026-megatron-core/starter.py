"""Megatron-Core Tutorial
========================
기존 PyTorch MLP를 Megatron-Core의 TP 적용 MLP로 대체하는 과정.

Megatron-Core: NVIDIA의 대규모 모델 학습 라이브러리.
Megatron-LM에서 핵심 로직을 분리한 재사용 가능한 라이브러리.

핵심 특징:
  - ColumnParallelLinear / RowParallelLinear: TP가 내장된 Linear
  - TransformerConfig: 모델 설정을 하나의 config 객체로 관리
  - ModuleSpec 패턴: 어떤 구현체(mcore local / Transformer Engine)를 쓸지 선택
  - 입력 텐서 형태: [seq, batch, hidden] (HuggingFace와 다름!)

설치:
  pip install megatron-core   # 또는 NVIDIA/Megatron-LM repo에서 직접

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaMLP(nn.Module):
    """일반 PyTorch MLP. 이걸 Megatron-Core 버전으로 바꿀 것.

구조: Linear(hidden → 4*hidden) → GELU → Linear(4*hidden → hidden)"""

    def __init__(self, hidden_size, ffn_hidden_size):
        raise NotImplementedError('TODO: implement VanillaMLP.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement VanillaMLP.forward; compare with solution.py only after trying.')

def comparison():
    """기존 PyTorch 코드와 Megatron-Core 코드를 나란히 비교."""
    raise NotImplementedError('TODO: implement comparison; compare with solution.py only after trying.')

def codebase_guide():
    raise NotImplementedError('TODO: implement codebase_guide; compare with solution.py only after trying.')

def launch_guide():
    raise NotImplementedError('TODO: implement launch_guide; compare with solution.py only after trying.')

def simulate_mcore_mlp():
    """Megatron-Core MLP의 weight 분할을 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_mcore_mlp; compare with solution.py only after trying.')

def benchmark_vanilla_vs_tp_split():
    """Vanilla MLP vs TP-split MLP의 single-device 성능 비교."""
    raise NotImplementedError('TODO: implement benchmark_vanilla_vs_tp_split; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_mcore_mlp()
    comparison()
    benchmark_vanilla_vs_tp_split()
    codebase_guide()
    launch_guide()
