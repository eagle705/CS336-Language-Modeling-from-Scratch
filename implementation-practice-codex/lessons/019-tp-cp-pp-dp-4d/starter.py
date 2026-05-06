"""4D Parallelism: TP × CP × PP × DP
=====================================
네 가지 parallelism을 동시에 적용하는 전체 시뮬레이션.

예시: 32 GPUs = 2 TP × 2 CP × 2 PP × 4 DP

  rank 배치 순서 (안쪽 → 바깥):  TP → CP → PP → DP
  rank = dp * (pp*cp*tp) + pp * (cp*tp) + cp * tp + tp_rank

  TP:  같은 layer, 같은 seq chunk → hidden dim split  (NVLink)
  CP:  같은 layer, 같은 hidden   → seq dim split      (NVLink)
  PP:  다른 layer                 → activation send/recv
  DP:  같은 모델, 다른 data       → gradient all-reduce

  시각화 (2TP × 2CP × 2PP × 2DP = 16 GPUs):

    DP group 0                            DP group 1
    ┌──────────────────────────┐          ┌──────────────────────────┐
    │ PP stage 0:              │          │ PP stage 0:              │
    │   CP0: GPU0(TP0) GPU1(TP1) │        │   CP0: GPU8   GPU9       │
    │   CP1: GPU2(TP0) GPU3(TP1) │        │   CP1: GPU10  GPU11      │
    │                          │          │                          │
    │ PP stage 1:              │          │ PP stage 1:              │
    │   CP0: GPU4(TP0) GPU5(TP1) │        │   CP0: GPU12  GPU13      │
    │   CP1: GPU6(TP0) GPU7(TP1) │        │   CP1: GPU14  GPU15      │
    └──────────────────────────┘          └──────────────────────────┘

  Forward 흐름 (DP group 0):
    1. PP stage 0의 layers:
       a. CP: seq를 2등분 → GPU(0,1)은 seq 앞절반, GPU(2,3)은 뒷절반
       b. TP: 각 seq chunk 내에서 hidden split → GPU0은 좌반, GPU1은 우반
       c. Attention: Ring Attention으로 CP간 KV 교환
       d. FFN: TP all-reduce
    2. PP stage 경계:
       activation을 stage 1으로 send/recv
    3. PP stage 1의 layers: 1과 동일
    4. DP:
       backward 후 gradient all-reduce (DP group 0 ↔ DP group 1)

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn.functional as F
import math

def build_4d_process_groups(tp_size, cp_size, pp_size, dp_size):
    """4D parallelism process group을 구성.

rank 배치: [DP][PP][CP][TP]
rank = dp*(pp*cp*tp) + pp*(cp*tp) + cp*tp + tp_rank"""
    raise NotImplementedError('TODO: implement build_4d_process_groups; compare with solution.py only after trying.')

def ring_attention_with_tp(Q_full, K_full, V_full, cp_size, tp_size):
    """TP + CP를 동시에 적용한 attention 시뮬레이션.

CP: seq를 cp_size 등분 → 각 CP rank가 Q chunk 담당
TP: head를 tp_size 등분 → 각 TP rank가 head subset 담당
Ring Attention: CP rank 간 KV chunk를 ring으로 순회

Q_full: (seq, num_heads, head_dim)"""
    raise NotImplementedError('TODO: implement ring_attention_with_tp; compare with solution.py only after trying.')

def ffn_with_tp(x, W1, W2, tp_size):
    """FFN에 TP를 적용. (CP 내 각 seq chunk에 독립 적용)

x: (seq_chunk, embed_dim)
W1: (embed_dim, ffn_hidden) → column split
W2: (ffn_hidden, embed_dim) → row split"""
    raise NotImplementedError('TODO: implement ffn_with_tp; compare with solution.py only after trying.')

def simulate_4d_forward():
    """TP + CP + PP + DP를 모두 적용한 forward pass 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_4d_forward; compare with solution.py only after trying.')

def communication_summary():
    raise NotImplementedError('TODO: implement communication_summary; compare with solution.py only after trying.')

def detailed_analysis():
    raise NotImplementedError('TODO: implement detailed_analysis; compare with solution.py only after trying.')

def setup_guide():
    raise NotImplementedError('TODO: implement setup_guide; compare with solution.py only after trying.')
if __name__ == '__main__':
    diff = simulate_4d_forward()
    communication_summary()
    detailed_analysis()
    setup_guide()
