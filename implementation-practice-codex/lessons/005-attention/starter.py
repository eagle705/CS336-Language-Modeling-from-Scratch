"""Attention Mechanisms
=====================
Scaled Dot-Product Attention → Multi-Head Attention → Flash Attention 개념

    Q, K, V  (query, key, value)
      |
    Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """가장 기본적인 attention.

Q: (batch, seq_q, d_k)
K: (batch, seq_k, d_k)
V: (batch, seq_k, d_v)

수식: softmax(Q @ K^T / sqrt(d_k)) @ V

왜 sqrt(d_k)로 나누나?
  Q @ K^T의 분산이 d_k에 비례 → 값이 커지면 softmax가 극단적 분포 →
  gradient vanishing → sqrt(d_k)로 나눠서 분산을 1로 정규화."""
    raise NotImplementedError('TODO: implement scaled_dot_product_attention; compare with solution.py only after trying.')

class MultiHeadAttention(nn.Module):
    """여러 head가 서로 다른 subspace에서 attention 수행.

전체 흐름:
  1) Q, K, V를 각각 linear projection
  2) num_heads개로 split
  3) 각 head에서 독립적으로 attention
  4) concat → linear projection

왜 multi-head?
  단일 attention은 하나의 유사도 패턴만 학습.
  multi-head는 여러 관점(문법, 의미, 위치 등)을 동시에 학습.

head_dim = embed_dim // num_heads
각 head는 작은 차원에서 동작 → 총 연산량은 single-head와 동일."""

    def __init__(self, embed_dim, num_heads):
        raise NotImplementedError('TODO: implement MultiHeadAttention.__init__; compare with solution.py only after trying.')

    def forward(self, x, mask=None):
        raise NotImplementedError('TODO: implement MultiHeadAttention.forward; compare with solution.py only after trying.')

class GroupedQueryAttention(nn.Module):
    """GQA: K, V head 수를 줄여서 KV cache 메모리 절약.

MHA:  Q heads = K heads = V heads = num_heads        (예: 32, 32, 32)
MQA:  Q heads = num_heads, K heads = V heads = 1     (예: 32, 1, 1)
GQA:  Q heads = num_heads, K heads = V heads = num_kv_heads  (예: 32, 8, 8)

KV cache 크기: 2 * num_kv_heads * head_dim * seq_len * batch
→ MHA 대비 num_heads/num_kv_heads 배 절약"""

    def __init__(self, embed_dim, num_heads, num_kv_heads):
        raise NotImplementedError('TODO: implement GroupedQueryAttention.__init__; compare with solution.py only after trying.')

    def forward(self, x, mask=None):
        raise NotImplementedError('TODO: implement GroupedQueryAttention.forward; compare with solution.py only after trying.')

def flash_attention_minimal(Q, K, V, block_size=32):
    """Flash Attention의 핵심 로직을 순수 Python으로 구현.
실제로는 CUDA kernel이지만, 알고리즘 이해용."""
    raise NotImplementedError('TODO: implement flash_attention_minimal; compare with solution.py only after trying.')

def demo():
    raise NotImplementedError('TODO: implement demo; compare with solution.py only after trying.')

def benchmark():
    """각 attention 구현의 forward 속도 비교."""
    raise NotImplementedError('TODO: implement benchmark; compare with solution.py only after trying.')
if __name__ == '__main__':
    demo()
    benchmark()
