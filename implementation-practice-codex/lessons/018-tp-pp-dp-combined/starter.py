"""TP + PP + DP Combined (3D Parallelism)
========================================
세 가지 parallelism을 동시에 적용하는 전체 시뮬레이션.

예시: 16 GPUs = 2 TP × 2 PP × 4 DP

  DP group 0          DP group 1          DP group 2          DP group 3
  ┌───────────┐       ┌───────────┐       ┌───────────┐       ┌───────────┐
  │PP0: GPU0,1│       │PP0: GPU4,5│       │PP0: GPU8,9│       │PP0:GPU12,13│
  │    (TP=2) │       │    (TP=2) │       │    (TP=2) │       │    (TP=2)  │
  │PP1: GPU2,3│       │PP1: GPU6,7│       │PP1:GPU10,11│      │PP1:GPU14,15│
  │    (TP=2) │       │    (TP=2) │       │    (TP=2)  │      │    (TP=2)  │
  └───────────┘       └───────────┘       └───────────┘       └───────────┘

  통신 패턴:
    TP:  all-reduce (노드 내 NVLink)    — 같은 layer를 나눠 계산
    PP:  send/recv  (stage 경계)        — 다른 layer를 순차 실행
    DP:  all-reduce (노드 간 IB 가능)   — 같은 모델, 다른 데이터

  각 GPU의 역할:
    GPU 0: DP group 0, PP stage 0, TP rank 0
    GPU 1: DP group 0, PP stage 0, TP rank 1  ← GPU 0과 같은 layer를 TP
    GPU 2: DP group 0, PP stage 1, TP rank 0  ← GPU 0과 다른 layer
    GPU 3: DP group 0, PP stage 1, TP rank 1
    GPU 4: DP group 1, PP stage 0, TP rank 0  ← GPU 0과 같은 모델, 다른 데이터
    ...

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def build_process_groups(world_size, tp_size, pp_size, dp_size):
    """3D parallelism의 process group을 구성.

rank 배치 순서: [DP][PP][TP]  (TP가 가장 안쪽 = 같은 노드)

rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank"""
    raise NotImplementedError('TODO: implement build_process_groups; compare with solution.py only after trying.')

def simulate_3d_parallelism():
    """TP + PP + DP를 모두 적용한 MLP 학습을 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_3d_parallelism; compare with solution.py only after trying.')

def communication_analysis():
    """3D parallelism의 통신량을 상세 분석."""
    raise NotImplementedError('TODO: implement communication_analysis; compare with solution.py only after trying.')

def memory_analysis():
    """3D parallelism에서 각 GPU의 메모리 사용량."""
    raise NotImplementedError('TODO: implement memory_analysis; compare with solution.py only after trying.')

def strategy_guide():
    raise NotImplementedError('TODO: implement strategy_guide; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_3d_parallelism()
    communication_analysis()
    memory_analysis()
    strategy_guide()
