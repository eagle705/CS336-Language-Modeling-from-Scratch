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
# ============================================================
# Part 5: Triton 기반 FlashAttention kernel fusion 예시
# ============================================================
#
# 실제 FlashAttention은 Python loop가 아니라 CUDA/Triton kernel 하나 안에서 아래 일을 fuse한다.
#
#   1. Q block load
#   2. K/V block load
#   3. QK^T score 계산
#   4. causal mask 적용
#   5. online softmax의 m/l 업데이트
#   6. P @ V 누적
#   7. output store
#
# 표준 PyTorch식 attention은 보통 중간 tensor를 HBM(global memory)에 만든다.
#
#   scores = Q @ K.T                  # (S, S) huge tensor
#   probs = softmax(scores)           # 또 (S, S)
#   out = probs @ V
#
# FlashAttention/Triton fusion은 scores/probs를 HBM에 저장하지 않고,
# kernel 내부 SRAM/register에 block 단위로만 들고 계산한다.
# 그래서 FLOPs 자체보다 "HBM read/write 횟수"를 줄이는 것이 핵심이다.

TRITON_FLASH_ATTENTION_SKELETON = r"""
# 실제 실행용 full kernel이 아니라 구조를 보여주는 Triton-style skeleton.
# 핵심은 @triton.jit kernel 하나 안에 QK, mask, online softmax, PV를 모두 fuse한다는 점.
#
# import triton
# import triton.language as tl
#
# @triton.jit
# def flash_attn_fwd_kernel(Q, K, V, O, stride_q, stride_k, stride_v, stride_o,
#                           seqlen: tl.constexpr, head_dim: tl.constexpr,
#                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
#                           is_causal: tl.constexpr):
#     pid_m = tl.program_id(0)       # 어떤 Q block인지
#     pid_bh = tl.program_id(1)      # batch * head id
#
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Q row offsets
#     offs_n = tl.arange(0, BLOCK_N)                    # K/V col offsets
#     offs_d = tl.arange(0, head_dim)
#
#     q = tl.load(Q + pid_bh * stride_q + offs_m[:, None] * head_dim + offs_d[None, :])
#
#     # online softmax state per Q row
#     m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)  # running max
#     l_i = tl.zeros((BLOCK_M,), tl.float32)                # running denominator
#     acc = tl.zeros((BLOCK_M, head_dim), tl.float32)       # running output numerator
#
#     for start_n in range(0, seqlen, BLOCK_N):
#         k = tl.load(K + pid_bh * stride_k + (start_n + offs_n)[:, None] * head_dim + offs_d[None, :])
#         v = tl.load(V + pid_bh * stride_v + (start_n + offs_n)[:, None] * head_dim + offs_d[None, :])
#
#         # QK^T score block. 이 block만 SRAM/register에 존재하고 (S,S) 전체는 만들지 않는다.
#         scores = tl.dot(q, tl.trans(k)) / tl.sqrt(head_dim)
#
#         if is_causal:
#             scores = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], scores, -float("inf"))
#
#         # online softmax update:
#         # 새 block의 row max를 반영하면서 이전 acc/l_i를 같은 scale로 보정한다.
#         m_new = tl.maximum(m_i, tl.max(scores, axis=1))
#         p = tl.exp(scores - m_new[:, None])
#         alpha = tl.exp(m_i - m_new)
#         l_i = l_i * alpha + tl.sum(p, axis=1)
#         acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
#         m_i = m_new
#
#     out = acc / l_i[:, None]
#     tl.store(O + pid_bh * stride_o + offs_m[:, None] * head_dim + offs_d[None, :], out)
#
# 실제 production kernel은 boundary check, dtype, dropout, backward, warp/stage tuning 등이 훨씬 복잡하다.
"""


# ============================================================
# Part 6: Linear Attention / Gated Delta Network (GDN)
# ============================================================
#
# 표준 attention:
#   output_t = softmax(q_t K_1:t^T) V_1:t
#   → 모든 과거 token의 K/V를 보거나 cache해야 하므로 decode KV cache가 O(seq_len).
#
# Linear attention 계열:
#   softmax attention을 정확히 만들기보다, recurrent state를 유지해서
#   token t마다 O(1) state update/read로 처리하려는 계열.
#
#   예: state_t = state_{t-1} + phi(k_t)^T v_t
#       output_t = phi(q_t) state_t
#
# GDN(Gated Delta Network):
#   Gated Delta Networks는 Mamba2/DeltaNet 계열의 linear attention류로 볼 수 있다.
#   고정 크기 state에 key -> value association을 저장하고,
#   delta rule로 "기존 기억에서 예측한 값"과 실제 value의 차이만 업데이트한다.
#
#   단순화한 update:
#     old_value = k_t @ state
#     delta = v_t - old_value
#     state = alpha_t * state + beta_t * outer(k_t, delta)
#     y_t = q_t @ state
#
#   alpha_t: forget/decay gate. 오래된 state를 얼마나 지울지 조절.
#   beta_t: update gate. 이번 token의 delta를 얼마나 강하게 쓸지 조절.
#
#   장점:
#     - attention matrix를 만들지 않음: O(S^2) 대신 O(S * state_size)
#     - decode 시 KV cache가 sequence 길이에 비례해서 커지지 않고 fixed recurrent state를 유지
#
#   trade-off:
#     - softmax attention과 완전히 같은 연산은 아님.
#     - long-range retrieval, in-context learning 성능은 architecture와 학습에 크게 의존.
#     - 실제 GDN/FLA kernel은 chunking/parallel scan/Triton kernel로 recurrent update를 병렬화한다.

class GatedDeltaNetwork(nn.Module):
    """
    GDN 느낌을 보여주는 최소 PyTorch 구현.

    이 코드는 논문/production kernel의 모든 세부사항을 재현하려는 목적이 아니라,
    "linear attention류는 K/V cache를 계속 쌓는 대신 fixed-size state를 업데이트한다"는
    감각을 보여주는 학습용 예시다.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_beta = nn.Linear(embed_dim, num_heads)
        self.W_alpha = nn.Linear(embed_dim, num_heads)
        self.W_gate = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        B, S, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.W_q(x).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        k = self.W_k(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(x).view(B, S, H, Dh).transpose(1, 2)

        # GDN/linear attention류는 k/q scale이 state update 안정성에 중요하다.
        # 여기서는 가장 단순하게 L2 normalize해서 outer-product update가 폭주하지 않게 한다.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        beta = torch.sigmoid(self.W_beta(x)).transpose(1, 2)   # (B, H, S)
        alpha = torch.sigmoid(self.W_alpha(x)).transpose(1, 2) # (B, H, S)
        out_gate = torch.sigmoid(self.W_gate(x)).view(B, S, H, Dh).transpose(1, 2)

        # state는 각 head마다 (key_dim, value_dim) matrix 하나.
        # decode 시에는 이 state만 들고 다음 token으로 넘어갈 수 있으므로 seq_len에 대해 O(1) cache.
        state = x.new_zeros(B, H, Dh, Dh)
        outputs = []

        for t in range(S):
            q_t = q[:, :, t, :]          # (B, H, Dh)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)    # (B, H, 1, 1)
            alpha_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)

            # 현재 state가 k_t에 대해 예측하는 value.
            old_value = torch.einsum("bhd,bhdv->bhv", k_t, state)
            delta = v_t - old_value

            # delta rule: 이미 state가 잘 기억하고 있으면 update가 작고,
            # 틀리게 예측한 부분만 outer(k_t, delta)로 보정한다.
            state = alpha_t * state + beta_t * torch.einsum("bhd,bhv->bhdv", k_t, delta)

            y_t = torch.einsum("bhd,bhdv->bhv", q_t, state)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=2)  # (B, H, S, Dh)
        y = y * out_gate
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(y)


# ============================================================
# Part 7: Demo
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

    # --- Linear Attention / GDN ---
    gdn = GatedDeltaNetwork(embed_dim=D, num_heads=H)
    out_gdn = gdn(x)
    print(f"\n[Gated Delta Network (linear attention류)]")
    print(f"  Input: {x.shape} → Output: {out_gdn.shape}")
    print(f"  State per layer/head: head_dim x head_dim = {D//H} x {D//H}")
    print(f"  Decode cache: KV 전체를 쌓는 대신 fixed recurrent state를 유지")

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
    print(f"  Triton fusion skeleton: TRITON_FLASH_ATTENTION_SKELETON 변수 참고")


# ============================================================
# Part 8: Performance Benchmark
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

    header = f"  {'Config':<18} {'Standard':>10} {'MHA':>10} {'GQA':>10} {'GDN':>10} {'Flash(min)':>11} {'F.sdpa':>10}"
    print(header)
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*10}")

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

        # GDN linear attention류. 이 예시는 Python recurrent loop라 production kernel보다 느릴 수 있다.
        # 실제 GDN/linear attention 구현은 chunking/parallel scan/Triton kernel로 병렬화한다.
        if S <= 1024:
            gdn = GatedDeltaNetwork(D, H).to(device)
            times["gdn"] = bench(lambda: gdn(x), n_warmup=1, n_iter=3)
        else:
            times["gdn"] = float('nan')

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
              f" {fmt(times['gqa']):>10} {fmt(times['gdn']):>10}"
              f" {fmt(times['flash_min']):>11} {fmt(times['sdpa']):>10}")

    print(f"""
  해석:
    - Standard: naive O(S^2) matmul. 단순하지만 S 커지면 느림.
    - MHA/GQA: Linear projection 포함. GQA는 KV head 적어서 약간 빠름.
    - GDN: linear attention류. 여기 구현은 Python loop라 느릴 수 있지만 cache/state는 O(1).
    - Flash(min): Python 구현이라 느림. 알고리즘 이해용.
    - F.sdpa: PyTorch 내장. CUDA면 FlashAttention2/cuDNN 자동 선택 → 가장 빠름.
    """)


if __name__ == "__main__":
    demo()
    benchmark()
