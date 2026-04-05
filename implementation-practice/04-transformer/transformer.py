"""
Transformer Architecture (GPT-style decoder-only)
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Components
# ============================================================

class RMSNorm(nn.Module):
    """
    RMSNorm: LayerNorm의 간소화 버전 (LLaMA 등에서 사용).
    mean을 빼지 않고 RMS(root mean square)로만 정규화.

    RMSNorm(x) = x / RMS(x) * gamma
    RMS(x) = sqrt(mean(x^2) + eps)

    LayerNorm 대비 장점: mean 계산 불필요 → 약간 빠름.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE: 상대 위치 정보를 Q, K에 회전으로 인코딩.

    핵심 아이디어: position m의 벡터를 m*theta만큼 회전.
    Q_m @ K_n = f(Q, m) @ f(K, n) → 내적이 (m-n)에만 의존 → 상대 위치 인코딩!

    구현: 인접한 두 차원을 한 쌍으로 묶어서 2D 회전 적용.
    """

    def __init__(self, head_dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        # theta_i = 1 / base^(2i/d) for i = 0, 1, ..., d/2-1
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("freqs", freqs)

        # position별 angle: (seq, d/2)
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)  # (seq, d/2)
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(self, x, start_pos=0):
        # x: (batch, heads, seq, head_dim)
        seq_len = x.size(2)
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq, d/2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]

        # 인접 쌍으로 분리: (x0, x1, x2, x3, ...) → (x0, x1), (x2, x3), ...
        x_even = x[..., 0::2]  # (batch, heads, seq, d/2)
        x_odd = x[..., 1::2]

        # 2D 회전: [cos -sin] [x_even]   [x_even*cos - x_odd*sin]
        #          [sin  cos] [x_odd ] = [x_even*sin + x_odd*cos]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 다시 interleave
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class FeedForward(nn.Module):
    """
    SwiGLU FFN (LLaMA style).
    FFN(x) = W2 @ (SiLU(W_gate @ x) * (W_up @ x))

    일반 FFN: 2개 행렬 (W1, W2), activation 1번
    SwiGLU:   3개 행렬 (W_gate, W_up, W2), gating mechanism → 성능 향상
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (GPT-2+ / LLaMA style)."""

    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, x, start_pos=0):
        x = x + self.attn(self.norm1(x), start_pos)   # residual + attention
        x = x + self.ffn(self.norm2(x))                # residual + FFN
        return x


class CausalSelfAttention(nn.Module):
    """Causal (decoder) self-attention with RoPE."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, start_pos=0):
        B, S, D = x.shape
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        Q = self.rope(Q, start_pos)
        K = self.rope(K, start_pos)

        # causal mask
        mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


# ============================================================
# Full Model
# ============================================================

class GPT(nn.Module):
    """Minimal GPT (decoder-only transformer)."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # weight tying: embedding과 output head 공유
        self.head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


# ============================================================
# Demo
# ============================================================

def demo():
    print("=" * 60)
    print("Transformer (GPT) Demo")
    print("=" * 60)

    torch.manual_seed(42)
    config = dict(vocab_size=1000, embed_dim=128, num_heads=4,
                  num_layers=4, ff_hidden_dim=512, max_seq_len=64)

    model = GPT(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Config: {config}")
    print(f"  Params: {n_params:,}")

    # Forward
    input_ids = torch.randint(0, 1000, (2, 32))
    logits = model(input_ids)
    print(f"\n  Input:  {input_ids.shape}")
    print(f"  Output: {logits.shape}")

    # Loss + backward
    targets = torch.randint(0, 1000, (2, 32))
    loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
    loss.backward()
    print(f"  Loss:   {loss.item():.4f}")

    # 각 component 파라미터
    print(f"\n  Component params:")
    print(f"    Embedding:   {model.tok_emb.weight.numel():>8,}")
    print(f"    Per block:   {sum(p.numel() for p in model.blocks[0].parameters()):>8,}")
    print(f"    LM head:     (weight tied with embedding)")


if __name__ == "__main__":
    demo()
