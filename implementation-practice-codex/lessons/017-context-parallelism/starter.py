"""Context Parallelism (CP)
==========================
시퀀스를 여러 GPU에 나눠서 처리. Long context 학습의 핵심.

문제: seq_len이 길면 attention의 O(S^2) 메모리가 단일 GPU에 안 들어감.
  예: S=128K, H=32, D=128, BF16 → attention scores만 ~128GB

TP/PP/DP로는 해결 안 됨:
  TP: layer 내부를 나누지만, 각 GPU가 전체 시퀀스의 attention 계산
  PP: layer를 나누지만, 각 stage에서 전체 시퀀스 필요
  DP: 데이터를 나누지만, 각 GPU가 전체 시퀀스 처리

CP 해결: 시퀀스 자체를 GPU에 분산!

    전체 시퀀스: [tok_0, tok_1, ..., tok_S-1]

    GPU 0: [tok_0, ..., tok_{S/4-1}]         ← Q chunk 0
    GPU 1: [tok_{S/4}, ..., tok_{S/2-1}]     ← Q chunk 1
    GPU 2: [tok_{S/2}, ..., tok_{3S/4-1}]    ← Q chunk 2
    GPU 3: [tok_{3S/4}, ..., tok_{S-1}]      ← Q chunk 3

    각 GPU가 자기 Q chunk에 대해 전체 KV를 순회하며 attention 계산.
    KV를 ring 형태로 GPU 간 전달 (= Ring Attention).

Ring Attention 동작:
    ┌─────────────────────────────────────────────────────┐
    │ Step 0: 각 GPU가 local KV로 attention 계산          │
    │   GPU 0: Q0 × KV0   GPU 1: Q1 × KV1   ...         │
    │                                                     │
    │ Step 1: KV를 오른쪽 이웃에게 전달 (ring)              │
    │   GPU 0: Q0 × KV3   GPU 1: Q1 × KV0   ...         │
    │         (KV3 받음)          (KV0 받음)               │
    │                                                     │
    │ Step 2: 다시 전달                                    │
    │   GPU 0: Q0 × KV2   GPU 1: Q1 × KV3   ...         │
    │                                                     │
    │ Step 3: 마지막 KV chunk                              │
    │   GPU 0: Q0 × KV1   GPU 1: Q1 × KV2   ...         │
    │                                                     │
    │ → 각 GPU가 전체 KV를 한 바퀴 돌며 attention 완성     │
    │ → 통신과 연산을 overlap 가능!                         │
    └─────────────────────────────────────────────────────┘

핵심 트릭: Online Softmax (Flash Attention과 동일)
  KV chunk가 하나씩 올 때마다 running max, running sum 업데이트.
  전체 attention matrix를 만들지 않아도 정확한 softmax 결과.

인터뷰 포인트:
  1. CP는 시퀀스 차원을 분산 → S^2 메모리 문제 해결
  2. Ring Attention = CP + online softmax
  3. 통신: KV chunk를 ring으로 전달 (send/recv, overlap 가능)
  4. Causal mask 최적화: 자기보다 미래 토큰의 KV는 skip 가능
  5. 4D parallelism: TP × PP × DP × CP

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn.functional as F
import math
from enum import Enum

class CPCommType(Enum):
    """Context Parallelism 통신 유형.

P2P (Ring Attention):
    KV를 이웃 GPU로 순차 전달. 통신과 연산 overlap 가능.
    + 메모리 효율 최고 (한 번에 KV chunk 1개만 보유)
    + 통신-연산 overlap → 통신 숨기기
    - 구현 복잡 (send/recv 동기화)
    - latency = (cp_size-1) × per-step latency

ALL_GATHER:
    모든 GPU가 전체 KV를 한 번에 모음.
    + 구현 단순 (한 번의 collective)
    + latency 낮음 (한 번에 끝)
    - 메모리 cp_size배 (전체 KV 보유)
    - overlap 불가 (gather 후 compute)

A2A (All-to-All, DeepSpeed-Ulysses):
    시퀀스 분할 → 헤드 분할로 layout 변환.
    Q,K,V를 all-to-all로 재배치: 각 GPU가 모든 시퀀스의 일부 헤드 담당.
    + attention 내부에서 추가 통신 불필요
    + GQA에서도 효율적
    - all-to-all 2회 필요 (forward: seq→head, head→seq)
    - head 수가 cp_size로 나눠져야 함"""
    P2P = 'p2p'
    ALL_GATHER = 'all_gather'
    A2A = 'a2a'

def ring_attention(Q_chunks, K_chunks, V_chunks, causal=False, comm_type=CPCommType.P2P):
    """Context Parallelism 시뮬레이션 — 통신 유형별 구현.

각 GPU가 Q chunk 하나를 들고, KV를 통신 유형에 따라 교환하며
attention을 계산.

Q_chunks: list of (chunk_seq, head_dim) per GPU
K_chunks: list of (chunk_seq, head_dim) per GPU
V_chunks: list of (chunk_seq, head_dim) per GPU
comm_type: CPCommType — P2P, ALL_GATHER, A2A"""
    raise NotImplementedError('TODO: implement ring_attention; compare with solution.py only after trying.')

def _cp_p2p(Q_chunks, K_chunks, V_chunks, causal):
    """P2P Ring Attention: KV를 ring으로 돌리며 online softmax."""
    raise NotImplementedError('TODO: implement _cp_p2p; compare with solution.py only after trying.')

def _cp_all_gather(Q_chunks, K_chunks, V_chunks, causal):
    """All-Gather: 전체 KV를 모은 뒤 각 GPU에서 local attention."""
    raise NotImplementedError('TODO: implement _cp_all_gather; compare with solution.py only after trying.')

def _cp_a2a(Q_chunks, K_chunks, V_chunks, causal):
    """All-to-All (DeepSpeed-Ulysses): 시퀀스 분할 ↔ 헤드 분할 변환.

핵심: multi-head attention에서 각 head는 독립적으로 attention 계산.
→ head를 GPU에 분배해도 수학적으로 동일!

Input:  list of (chunk_seq, num_heads, head_dim) per GPU  [seq 분할]
After all-to-all: (total_seq, heads_per_gpu, head_dim)    [head 분할]
Attention 후 다시 all-to-all로 원래 layout 복원.

실제 torch.distributed:
  # Forward all-to-all: seq-split → head-split
  input_list = Q_local.chunk(cp_size, dim=1)   # head 방향으로 나눔
  output_list = [torch.empty_like(input_list[0])] * cp_size
  dist.all_to_all(output_list, input_list, group=cp_group)
  Q_head_local = torch.cat(output_list, dim=0)  # seq 방향으로 합침"""
    raise NotImplementedError('TODO: implement _cp_a2a; compare with solution.py only after trying.')

def verify_ring_attention():
    """모든 CP 통신 유형이 표준 attention과 동일한 결과를 내는지 검증."""
    raise NotImplementedError('TODO: implement verify_ring_attention; compare with solution.py only after trying.')

def communication_analysis():
    """CP의 통신량과 overlap 분석."""
    raise NotImplementedError('TODO: implement communication_analysis; compare with solution.py only after trying.')

def parallelism_4d():
    """4D parallelism 구성."""
    raise NotImplementedError('TODO: implement parallelism_4d; compare with solution.py only after trying.')

def cp_comparison():
    raise NotImplementedError('TODO: implement cp_comparison; compare with solution.py only after trying.')
if __name__ == '__main__':
    verify_ring_attention()
    communication_analysis()
    parallelism_4d()
    cp_comparison()
