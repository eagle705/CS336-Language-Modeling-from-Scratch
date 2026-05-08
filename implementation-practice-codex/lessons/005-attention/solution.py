"""
Attention Mechanisms
=====================
Scaled Dot-Product Attention → Multi-Head Attention → Flash Attention 개념

    Q, K, V  (query, key, value)
      |
    Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: Scaled Dot-Product Attention (step by step)
# ============================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    가장 기본적인 attention.

    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)

    수식: softmax(Q @ K^T / sqrt(d_k)) @ V

    왜 sqrt(d_k)로 나누나?
      Q @ K^T의 분산이 d_k에 비례 → 값이 커지면 softmax가 극단적 분포 →
      gradient vanishing → sqrt(d_k)로 나눠서 분산을 1로 정규화.
    """
    d_k = Q.size(-1)

    # (1) Q @ K^T: 각 query와 모든 key의 유사도 계산
    #     shape: (batch, seq_q, d_k) @ (batch, d_k, seq_k) = (batch, seq_q, seq_k)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # (2) mask: causal attention에서 미래 토큰을 못 보게 -inf 처리
    #     mask == 0인 위치는 attention하면 안 되는 위치라 softmax 확률이 0이 되도록 막는다.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # (3) softmax: 유사도를 확률 분포로 변환
    attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_q, seq_k)

    # (4) weighted sum: 확률에 따라 value를 가중합
    output = attn_weights @ V  # (batch, seq_q, d_v)

    return output, attn_weights


# ============================================================
# Part 2: Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    여러 head가 서로 다른 subspace에서 attention 수행.

    전체 흐름:
      1) Q, K, V를 각각 linear projection
      2) num_heads개로 split
      3) 각 head에서 독립적으로 attention
      4) concat → linear projection

    왜 multi-head?
      단일 attention은 하나의 유사도 패턴만 학습.
      multi-head는 여러 관점(문법, 의미, 위치 등)을 동시에 학습.

    head_dim = embed_dim // num_heads
    각 head는 작은 차원에서 동작 → 총 연산량은 single-head와 동일.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)  # (embed, embed) = (embed, heads * head_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        # (1) Linear projection: (B, S, embed) → (B, S, embed)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (2) head 분리: (B, S, embed) → (B, num_heads, S, head_dim)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # (3) 각 head에서 attention: (B, num_heads, S, head_dim)
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # (4) head 합치기: (B, num_heads, S, head_dim) → (B, S, embed)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # (5) 출력 projection
        return self.W_o(attn_out)


# ============================================================
# Part 3: Grouped-Query Attention (GQA)
# ============================================================

class GroupedQueryAttention(nn.Module):
    """
    GQA: K, V head 수를 줄여서 KV cache 메모리 절약.

    MHA:  Q heads = K heads = V heads = num_heads        (예: 32, 32, 32)
    MQA:  Q heads = num_heads, K heads = V heads = 1     (예: 32, 1, 1)
    GQA:  Q heads = num_heads, K heads = V heads = num_kv_heads  (예: 32, 8, 8)

    KV cache 크기: 2 * num_kv_heads * head_dim * seq_len * batch
    → MHA 대비 num_heads/num_kv_heads 배 절약
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # 몇 개의 Q head가 KV를 공유
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.W_k = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # KV를 Q head 수만큼 repeat: (B, num_kv_heads, S, d) → (B, num_heads, S, d)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(attn_out)


# ============================================================
# Part 4: Flash Attention 개념
# ============================================================
#
# 표준 attention의 문제: S x S attention matrix를 메모리에 전부 저장
#   메모리: O(S^2)  → seq_len이 길면 OOM
#
# Flash Attention 핵심 아이디어:
#   Q, K, V를 블록 단위로 쪼개서 처리 (tiling)
#   attention matrix를 한 번에 만들지 않고 블록씩 계산 → 메모리 O(S)
#
# 알고리즘 (simplified):
#   for each Q_block:
#       m_prev = -inf, l_prev = 0, O_prev = 0  (online softmax 상태)
#       for each K_block, V_block:
#           S_block = Q_block @ K_block.T / sqrt(d)
#           m_new = max(m_prev, rowmax(S_block))
#           P_block = exp(S_block - m_new)
#           l_new = exp(m_prev - m_new) * l_prev + rowsum(P_block)
#           O_new = exp(m_prev - m_new) * O_prev + P_block @ V_block
#       output = O_new / l_new
#
# 핵심 트릭: "online softmax"
#   softmax(x) = exp(x - max) / sum(exp(x - max))
#   max와 sum을 블록 단위로 점진적으로 업데이트 가능!
#
# PyTorch 사용법 (가장 쉬운 방법):
#   F.scaled_dot_product_attention(Q, K, V, is_causal=True)
#   → 내부적으로 Flash Attention 2 자동 사용 (CUDA, 조건 충족 시)

def flash_attention_minimal(Q, K, V, block_size=32):
    """
    Flash Attention의 핵심 로직을 순수 Python으로 구현.
    실제로는 CUDA kernel이지만, 알고리즘 이해용.
    """
    B, H, S, D = Q.shape
    O = torch.zeros_like(Q)

    for q_start in range(0, S, block_size):
        q_end = min(q_start + block_size, S)
        Q_block = Q[:, :, q_start:q_end, :]  # (B, H, block, D)

        # online softmax 상태 초기화
        m = torch.full((B, H, q_end - q_start, 1), float('-inf'))  # 현재까지의 max
        l = torch.zeros(B, H, q_end - q_start, 1)                  # 현재까지의 sum(exp)
        O_acc = torch.zeros(B, H, q_end - q_start, D)              # 누적 output

        for k_start in range(0, S, block_size):
            k_end = min(k_start + block_size, S)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            # block attention score
            S_block = Q_block @ K_block.transpose(-2, -1) / math.sqrt(D)

            # online softmax update
            m_new = torch.maximum(m, S_block.max(dim=-1, keepdim=True).values)
            P_block = torch.exp(S_block - m_new)

            # 이전 누적값을 새 max에 맞게 보정
            correction = torch.exp(m - m_new)
            l_new = correction * l + P_block.sum(dim=-1, keepdim=True)
            O_acc = correction * O_acc + P_block @ V_block

            m = m_new
            l = l_new

        O[:, :, q_start:q_end, :] = O_acc / l

    return O


# ============================================================
# Part 5: Demo
# ============================================================

def demo():
    print("=" * 60)
    print("Attention Demo")
    print("=" * 60)

    torch.manual_seed(42)
    B, S, D, H = 2, 8, 64, 4

    # --- Scaled Dot-Product ---
    Q = torch.randn(B, S, D)
    K = torch.randn(B, S, D)
    V = torch.randn(B, S, D)

    # causal mask: 하삼각 행렬 (미래 토큰 차단)
    causal_mask = torch.tril(torch.ones(S, S)).unsqueeze(0)  # (1, S, S)
    out, weights = scaled_dot_product_attention(Q, K, V, causal_mask)
    print(f"\n[Scaled Dot-Product Attention]")
    print(f"  Q,K,V: {Q.shape} → Output: {out.shape}")
    print(f"  Attention weights[0,0,:]: {weights[0, 0, :].tolist()}")
    print(f"  (첫 토큰은 자기 자신에만 attend → [1.0, 0, 0, ...])")

    # --- Multi-Head Attention ---
    mha = MultiHeadAttention(embed_dim=D, num_heads=H)
    x = torch.randn(B, S, D)
    out_mha = mha(x)
    print(f"\n[Multi-Head Attention]")
    print(f"  Input: {x.shape} → Output: {out_mha.shape}")
    print(f"  Params: {sum(p.numel() for p in mha.parameters()):,}")

    # --- GQA ---
    gqa = GroupedQueryAttention(embed_dim=D, num_heads=H, num_kv_heads=2)
    out_gqa = gqa(x)
    print(f"\n[Grouped-Query Attention (GQA)]")
    print(f"  Q heads={H}, KV heads=2 (2 Q heads per KV group)")
    print(f"  Output: {out_gqa.shape}")
    print(f"  Params: {sum(p.numel() for p in gqa.parameters()):,}")
    print(f"  KV cache 절약: {H}÷2 = {H//2}x 감소")

    # --- Flash Attention (검증) ---
    Q = torch.randn(B, H, S, D // H)
    K = torch.randn(B, H, S, D // H)
    V = torch.randn(B, H, S, D // H)

    out_standard, _ = scaled_dot_product_attention(
        Q.view(B * H, S, -1), K.view(B * H, S, -1), V.view(B * H, S, -1)
    )
    out_standard = out_standard.view(B, H, S, -1)
    out_flash = flash_attention_minimal(Q, K, V, block_size=4)

    diff = (out_standard - out_flash).abs().max().item()
    print(f"\n[Flash Attention (minimal)]")
    print(f"  vs standard attention max diff: {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")
    print(f"  메모리: standard O(S^2)={S*S}, flash O(S)={S}")


# ============================================================
# Part 6: Performance Benchmark
# ============================================================

def benchmark():
    """각 attention 구현의 forward 속도 비교."""
    import time
    print("\n" + "=" * 60)
    print("Attention Performance Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = device == "cuda"

    configs = [
        # (B, H, S, Dh, label)
        (2,  4,   64,  32, "Tiny   S=64"),
        (2,  4,  256,  32, "Small  S=256"),
        (2,  4, 1024,  32, "Medium S=1024"),
        (2,  4, 4096,  32, "Large  S=4096"),
    ]

    print(f"\n  Device: {device}")
    print(f"  Warm-up: 3 iters, Measure: 10 iters\n")

    header = f"  {'Config':<18} {'Standard':>10} {'MHA':>10} {'GQA':>10} {'Flash(min)':>11} {'F.sdpa':>10}"
    print(header)
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*10}")

    for B, H, S, Dh, label in configs:
        D = H * Dh
        times = {}

        # --- Standard scaled dot-product ---
        Q = torch.randn(B, S, D, device=device)
        K = torch.randn(B, S, D, device=device)
        V = torch.randn(B, S, D, device=device)

        def bench(fn, n_warmup=3, n_iter=10):
            for _ in range(n_warmup):
                fn()
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn()
            if is_cuda:
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n_iter * 1000  # ms

        # Standard
        times["standard"] = bench(lambda: scaled_dot_product_attention(Q, K, V))

        # MHA
        mha = MultiHeadAttention(D, H).to(device)
        x = torch.randn(B, S, D, device=device)
        times["mha"] = bench(lambda: mha(x))

        # GQA (KV heads = H//2)
        gqa = GroupedQueryAttention(D, H, num_kv_heads=max(1, H // 2)).to(device)
        times["gqa"] = bench(lambda: gqa(x))

        # Flash (minimal, Python) — S>1024이면 너무 느려서 skip
        if S <= 1024:
            Qf = torch.randn(B, H, S, Dh, device=device)
            Kf = torch.randn(B, H, S, Dh, device=device)
            Vf = torch.randn(B, H, S, Dh, device=device)
            bs = min(64, S)
            times["flash_min"] = bench(
                lambda: flash_attention_minimal(Qf, Kf, Vf, block_size=bs),
                n_warmup=1, n_iter=3,
            )
        else:
            times["flash_min"] = float('nan')

        # F.scaled_dot_product_attention (PyTorch native, uses FlashAttn on CUDA)
        Qn = torch.randn(B, H, S, Dh, device=device)
        Kn = torch.randn(B, H, S, Dh, device=device)
        Vn = torch.randn(B, H, S, Dh, device=device)
        times["sdpa"] = bench(lambda: F.scaled_dot_product_attention(Qn, Kn, Vn, is_causal=True))

        # 출력
        def fmt(v):
            if v != v:  # nan
                return "skip"
            return f"{v:.2f}ms"

        print(f"  {label:<18} {fmt(times['standard']):>10} {fmt(times['mha']):>10}"
              f" {fmt(times['gqa']):>10} {fmt(times['flash_min']):>11} {fmt(times['sdpa']):>10}")

    print(f"""
  해석:
    - Standard: naive O(S^2) matmul. 단순하지만 S 커지면 느림.
    - MHA/GQA: Linear projection 포함. GQA는 KV head 적어서 약간 빠름.
    - Flash(min): Python 구현이라 느림. 알고리즘 이해용.
    - F.sdpa: PyTorch 내장. CUDA면 FlashAttention2/cuDNN 자동 선택 → 가장 빠름.
    """)


if __name__ == "__main__":
    demo()
    benchmark()
