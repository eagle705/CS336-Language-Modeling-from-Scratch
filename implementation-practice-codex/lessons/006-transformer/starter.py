"""Transformer Architecture (GPT-style decoder-only)
===================================================
전체 구조를 바닥부터 구현.

  Input IDs → Token Embedding + Position Embedding
       ↓
  ┌─────────────────────────┐
  │  Transformer Block × N  │
  │  ┌───────────────────┐  │
  │  │ LayerNorm         │  │
  │  │ Multi-Head Attn   │──│── + (residual)
  │  │ LayerNorm         │  │
  │  │ FFN (MLP)         │──│── + (residual)
  │  └───────────────────┘  │
  └─────────────────────────┘
       ↓
  LayerNorm → Linear (vocab projection) → logits

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """RMSNorm: LayerNorm의 간소화 버전 (LLaMA 등에서 사용).
mean을 빼지 않고 RMS(root mean square)로만 정규화.

RMSNorm(x) = x / RMS(x) * gamma
RMS(x) = sqrt(mean(x^2) + eps)

LayerNorm 대비 장점: mean 계산 불필요 → 약간 빠름."""

    def __init__(self, dim, eps=1e-06):
        raise NotImplementedError('TODO: implement RMSNorm.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement RMSNorm.forward; compare with solution.py only after trying.')

class RotaryPositionalEmbedding(nn.Module):
    """RoPE: 상대 위치 정보를 Q, K에 회전으로 인코딩.

핵심 아이디어: position m의 벡터를 m*theta만큼 회전.
Q_m @ K_n = f(Q, m) @ f(K, n) → 내적이 (m-n)에만 의존 → 상대 위치 인코딩!

구현: 인접한 두 차원을 한 쌍으로 묶어서 2D 회전 적용."""

    def __init__(self, head_dim, max_seq_len=4096, base=10000.0):
        raise NotImplementedError('TODO: implement RotaryPositionalEmbedding.__init__; compare with solution.py only after trying.')

    def forward(self, x, start_pos=0):
        raise NotImplementedError('TODO: implement RotaryPositionalEmbedding.forward; compare with solution.py only after trying.')

class FeedForward(nn.Module):
    """SwiGLU FFN (LLaMA style).
FFN(x) = W2 @ (SiLU(W_gate @ x) * (W_up @ x))

일반 FFN: 2개 행렬 (W1, W2), activation 1번
SwiGLU:   3개 행렬 (W_gate, W_up, W2), gating mechanism → 성능 향상"""

    def __init__(self, embed_dim, hidden_dim):
        raise NotImplementedError('TODO: implement FeedForward.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement FeedForward.forward; compare with solution.py only after trying.')

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (GPT-2+ / LLaMA style)."""

    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        raise NotImplementedError('TODO: implement TransformerBlock.__init__; compare with solution.py only after trying.')

    def forward(self, x, start_pos=0):
        raise NotImplementedError('TODO: implement TransformerBlock.forward; compare with solution.py only after trying.')

class CausalSelfAttention(nn.Module):
    """Causal (decoder) self-attention with RoPE."""

    def __init__(self, embed_dim, num_heads):
        raise NotImplementedError('TODO: implement CausalSelfAttention.__init__; compare with solution.py only after trying.')

    def forward(self, x, start_pos=0):
        raise NotImplementedError('TODO: implement CausalSelfAttention.forward; compare with solution.py only after trying.')

class GPT(nn.Module):
    """Minimal GPT (decoder-only transformer)."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_len=512):
        raise NotImplementedError('TODO: implement GPT.__init__; compare with solution.py only after trying.')

    def forward(self, input_ids):
        raise NotImplementedError('TODO: implement GPT.forward; compare with solution.py only after trying.')

def demo():
    raise NotImplementedError('TODO: implement demo; compare with solution.py only after trying.')
if __name__ == '__main__':
    demo()
