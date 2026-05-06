"""Communication Overlaps
========================
통신과 연산을 겹쳐서 학습 throughput 극대화.

문제: GPU가 통신 기다리는 동안 idle → GPU utilization 저하
해결: 비동기 통신으로 연산과 통신을 동시에 수행

    Without overlap:
    GPU:  [compute][  wait  ][compute][  wait  ][compute]
    NIC:  [  idle  ][ comm  ][  idle  ][ comm  ][  idle ]

    With overlap:
    GPU:  [compute][compute][compute]
    NIC:  [ comm  ][ comm  ][ comm  ]
    → GPU와 NIC가 동시에 일함!

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import time

def simulate_overlap():
    """비동기 통신으로 overlap하는 효과를 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_overlap; compare with solution.py only after trying.')

def cuda_streams_demo():
    """CUDA stream으로 연산 overlap (GPU 필요)."""
    raise NotImplementedError('TODO: implement cuda_streams_demo; compare with solution.py only after trying.')

def performance_tips():
    raise NotImplementedError('TODO: implement performance_tips; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_overlap()
    cuda_streams_demo()
    performance_tips()
