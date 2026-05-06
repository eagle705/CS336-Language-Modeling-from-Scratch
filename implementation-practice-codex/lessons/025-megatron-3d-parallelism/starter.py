"""Megatron-LM 3D Parallelism & Codebase Guide
==============================================
TP + PP + DP를 결합한 대규모 모델 학습.

3D Parallelism 구조:
  전체 N GPUs = TP × PP × DP

  예: 64 GPUs = 8 TP × 4 PP × 2 DP

  TP (Tensor Parallelism):  같은 layer를 여러 GPU가 나눠 계산
  PP (Pipeline Parallelism): 다른 layer를 다른 GPU에 배치
  DP (Data Parallelism):    같은 모델을 다른 데이터로 학습

  Visualization (8 TP × 4 PP × 2 DP = 64 GPUs):

    DP group 0:                        DP group 1:
    ┌─────────────────────────────┐    ┌─────────────────────────────┐
    │ PP stage 0: GPU[0:8]   (TP)│    │ PP stage 0: GPU[32:40] (TP)│
    │ PP stage 1: GPU[8:16]  (TP)│    │ PP stage 1: GPU[40:48] (TP)│
    │ PP stage 2: GPU[16:24] (TP)│    │ PP stage 2: GPU[48:56] (TP)│
    │ PP stage 3: GPU[24:32] (TP)│    │ PP stage 3: GPU[56:64] (TP)│
    └─────────────────────────────┘    └─────────────────────────────┘

  통신 패턴:
    TP:  all-reduce (노드 내 NVLink, 빠름)
    PP:  send/recv  (노드 간 가능)
    DP:  all-reduce (노드 간 InfiniBand)

  배치 우선순위: TP는 노드 내 (NVLink), PP/DP는 노드 간 (IB) 허용

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn

def simulate_3d_parallelism():
    """3D parallelism의 GPU 배치와 통신 패턴을 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_3d_parallelism; compare with solution.py only after trying.')

def parallelism_strategy_guide():
    """모델 크기별 권장 parallelism 전략."""
    raise NotImplementedError('TODO: implement parallelism_strategy_guide; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_3d_parallelism()
    parallelism_strategy_guide()
