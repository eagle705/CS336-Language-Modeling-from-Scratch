"""Virtual Pipeline Parallelism (VPP)
====================================
일반 PP의 bubble을 줄이는 Megatron-LM의 핵심 기법.

일반 PP의 문제: Pipeline bubble
  PP=4, M=8 micro-batches일 때:
  Bubble ratio = (PP-1)/M = 3/8 = 37.5%

VPP 아이디어: 각 GPU에 연속된 layer 대신 "비연속 layer 묶음"을 배치.
  virtual_pipeline_model_parallel_size (= V) = 각 GPU가 담당하는 chunk 수

일반 PP (V=1): 각 GPU가 연속된 layer 묶음 하나
  GPU 0: [Layer 0-3]
  GPU 1: [Layer 4-7]
  GPU 2: [Layer 8-11]
  GPU 3: [Layer 12-15]

VPP (V=2): 각 GPU가 2개의 비연속 chunk
  GPU 0: [Layer 0-1] + [Layer 8-9]     ← 2 chunks
  GPU 1: [Layer 2-3] + [Layer 10-11]
  GPU 2: [Layer 4-5] + [Layer 12-13]
  GPU 3: [Layer 6-7] + [Layer 14-15]

  Forward: 0→1→2→3→0→1→2→3 (2 round trips)
           chunk0        chunk1

왜 bubble이 줄어드나?
  - 일반 PP: forward가 stage 0→3까지 가야 backward 시작 (3 time slots 대기)
  - VPP: chunk가 2개이므로 chunk 0의 forward가 빨리 끝남 → 더 빨리 backward 시작

  Bubble ratio = (PP-1) / (M × V)
  V=1: (4-1)/(8×1) = 37.5%
  V=2: (4-1)/(8×2) = 18.75%  ← 절반!
  V=4: (4-1)/(8×4) = 9.375%

  대가: 통신 증가 (round trip이 V번)

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn

def visualize_schedules():
    """일반 PP와 VPP의 schedule을 나란히 비교."""
    raise NotImplementedError('TODO: implement visualize_schedules; compare with solution.py only after trying.')

def simulate_vpp_layer_assignment():
    """VPP에서 layer가 GPU에 어떻게 배치되는지 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_vpp_layer_assignment; compare with solution.py only after trying.')

def simulate_vpp_forward():
    """VPP forward pass를 실제 데이터로 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_vpp_forward; compare with solution.py only after trying.')

def bubble_analysis():
    """다양한 설정에서 bubble 비율 비교."""
    raise NotImplementedError('TODO: implement bubble_analysis; compare with solution.py only after trying.')

def memory_impact():
    raise NotImplementedError('TODO: implement memory_impact; compare with solution.py only after trying.')
if __name__ == '__main__':
    visualize_schedules()
    simulate_vpp_layer_assignment()
    simulate_vpp_forward()
    bubble_analysis()
    memory_impact()
