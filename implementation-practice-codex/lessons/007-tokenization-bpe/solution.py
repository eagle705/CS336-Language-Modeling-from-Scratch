"""
Long Context Techniques
========================
긴 시퀀스를 효율적으로 처리하는 방법들.

문제: 표준 attention은 O(S^2) 메모리/연산 → S가 길면 불가능
  S=2048:   4M attention entries
  S=128K:  16B attention entries (4000x 증가!)

해결 방법들:
  1. RoPE + NTK-aware scaling (위치 인코딩 외삽)
  2. Sliding Window Attention (Mistral)
  3. Ring Attention (시퀀스를 GPU에 분산)
  4. KV Cache 최적화 (GQA, MQA, quantized KV cache)
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# Part 1: RoPE Scaling (학습 길이를 넘어서 추론)
# ============================================================

def rope_scaling_demo():
    """
    RoPE 외삽 문제와 해결법.

    문제: RoPE는 학습 시 본 위치까지만 잘 동작.
          학습: 4K → 추론: 32K 하면 성능 저하.

    해결 1: Position Interpolation (Linear Scaling)
      theta'_i = theta_i / scale_factor
      → 위치를 압축해서 학습 범위 안에 매핑
      → 단점: 가까운 토큰의 구별력 감소

    해결 2: NTK-aware Scaling (YaRN 등)
      base' = base * scale_factor^(dim/(dim-2))
      → 고주파(가까운 위치)는 유지, 저주파(먼 위치)만 외삽
      → 더 좋은 성능

    해결 3: Dynamic NTK
      seq_len이 학습 길이를 넘으면 동적으로 base 조정
    """
    print("=" * 60)
    print("RoPE Scaling Methods")
    print("=" * 60)

    head_dim = 16
    train_len = 4096
    target_len = 32768
    scale = target_len / train_len  # 8x

    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # (1) Original RoPE
    positions = torch.arange(target_len).float()
    original_angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

    # (2) Linear Scaling: 위치를 scale로 나눔
    linear_angles = (positions / scale).unsqueeze(1) * freqs.unsqueeze(0)

    # (3) NTK-aware: base를 조정
    ntk_base = base * scale ** (head_dim / (head_dim - 2))
    ntk_freqs = 1.0 / (ntk_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ntk_angles = positions.unsqueeze(1) * ntk_freqs.unsqueeze(0)

    print(f"\n  Train length: {train_len}, Target: {target_len}, Scale: {scale}x")
    print(f"\n  Original base: {base}")
    print(f"  NTK base:      {ntk_base:.0f}")
    print(f"\n  Frequency comparison (dim 0, dim -1):")
    print(f"    Original: [{freqs[0]:.6f}, ..., {freqs[-1]:.6f}]")
    print(f"    NTK:      [{ntk_freqs[0]:.6f}, ..., {ntk_freqs[-1]:.6f}]")
    print(f"    → 고주파(dim 0) 거의 유지, 저주파(dim -1)만 감소")


# ============================================================
# Part 2: Sliding Window Attention
# ============================================================

def sliding_window_attention(Q, K, V, window_size):
    """
    Sliding Window Attention (Mistral, Longformer 등).

    각 토큰이 앞의 window_size개 토큰만 attend.
    메모리: O(S * W) instead of O(S^2)

    여러 layer를 쌓으면 receptive field가 넓어짐:
      Layer 1: 각 토큰이 W개 토큰 봄
      Layer 2: 각 토큰이 2W개 토큰 봄 (layer 1의 W 토큰이 각각 W개를 봤으므로)
      Layer L: 각 토큰이 L*W개 토큰 봄
    """
    B, S, D = Q.shape

    # 표준 attention scores
    scores = Q @ K.transpose(-2, -1) / math.sqrt(D)

    # sliding window mask: |i - j| > window_size 인 위치를 -inf
    positions = torch.arange(S)
    mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= window_size
    # causal도 적용 (j > i 차단)
    causal = positions.unsqueeze(0) >= positions.unsqueeze(1)
    mask = mask & causal

    scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn


# ============================================================
# Part 3: Ring Attention (시퀀스 병렬)
# ============================================================
#
# 긴 시퀀스를 GPU에 나눠서 처리.
#
# 아이디어: 시퀀스를 N개 chunk로 나눠 각 GPU에 배치.
#           각 GPU가 자기 Q chunk에 대해 모든 KV chunk를 순회하며 attention 계산.
#           KV chunk를 ring 형태로 GPU간 전달.
#
#   GPU 0: Q0  ←→ KV0 → KV1 → KV2 → KV3 (ring으로 순회)
#   GPU 1: Q1  ←→ KV1 → KV2 → KV3 → KV0
#   GPU 2: Q2  ←→ KV2 → KV3 → KV0 → KV1
#   GPU 3: Q3  ←→ KV3 → KV0 → KV1 → KV2
#
# 장점: 시퀀스 길이에 비례하여 GPU 추가 가능
# 통신: KV chunk를 ring으로 전달 → 계산과 통신 overlap 가능
#
# 구현 핵심: online softmax (Flash Attention과 동일 트릭)
#   KV chunk가 하나씩 올 때마다 running max, running sum 업데이트

def simulate_ring_attention():
    """Ring Attention의 chunk별 처리를 시뮬레이션."""
    print("\n" + "=" * 60)
    print("Ring Attention Simulation")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    total_seq = 16
    chunk_size = total_seq // num_gpus  # 4
    D = 8

    Q_full = torch.randn(total_seq, D)
    K_full = torch.randn(total_seq, D)
    V_full = torch.randn(total_seq, D)

    # 표준 attention (정답)
    scores_full = Q_full @ K_full.T / math.sqrt(D)
    attn_full = F.softmax(scores_full, dim=-1)
    out_full = attn_full @ V_full

    # Ring attention 시뮬레이션
    Q_chunks = Q_full.chunk(num_gpus)
    K_chunks = K_full.chunk(num_gpus)
    V_chunks = V_full.chunk(num_gpus)

    outputs = []
    for gpu_id in range(num_gpus):
        Q_local = Q_chunks[gpu_id]  # 이 GPU의 Q
        # online softmax 상태
        m = torch.full((chunk_size, 1), float('-inf'))
        l = torch.zeros(chunk_size, 1)
        O_acc = torch.zeros(chunk_size, D)

        # ring으로 KV chunk 순회
        for step in range(num_gpus):
            kv_idx = (gpu_id + step) % num_gpus
            K_block = K_chunks[kv_idx]
            V_block = V_chunks[kv_idx]

            # block attention score
            S_block = Q_local @ K_block.T / math.sqrt(D)

            # online softmax update (Flash Attention과 동일)
            m_new = torch.maximum(m, S_block.max(dim=-1, keepdim=True).values)
            P_block = torch.exp(S_block - m_new)
            correction = torch.exp(m - m_new)
            l_new = correction * l + P_block.sum(dim=-1, keepdim=True)
            O_acc = correction * O_acc + P_block @ V_block
            m, l = m_new, l_new

        outputs.append(O_acc / l)

    out_ring = torch.cat(outputs, dim=0)
    diff = (out_full - out_ring).abs().max().item()

    print(f"  Total seq: {total_seq}, GPUs: {num_gpus}, Chunk: {chunk_size}")
    print(f"  Standard vs Ring diff: {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")
    print(f"\n  메모리 per GPU: O(chunk^2) = O((S/N)^2) vs 전체 O(S^2)")


# ============================================================
# Part 4: KV Cache 최적화
# ============================================================

def kv_cache_analysis():
    """KV cache 메모리 분석."""
    print("\n" + "=" * 60)
    print("KV Cache Memory Analysis")
    print("=" * 60)

    configs = [
        ("LLaMA-7B (MHA)",  32, 32, 128, 4096),
        ("LLaMA-70B (GQA)", 64, 8,  128, 4096),
        ("Mistral-7B (GQA)", 32, 8,  128, 32768),
    ]

    for name, num_heads, num_kv_heads, head_dim, max_seq in configs:
        # KV cache: 2 (K+V) * num_kv_heads * head_dim * seq_len * num_layers * batch
        num_layers = 32
        batch = 1
        kv_per_token = 2 * num_kv_heads * head_dim * num_layers * 2  # 2 bytes (BF16)
        kv_total = kv_per_token * max_seq * batch / 1e9

        mha_kv_per_token = 2 * num_heads * head_dim * num_layers * 2
        mha_total = mha_kv_per_token * max_seq * batch / 1e9

        print(f"\n  {name}:")
        print(f"    KV heads: {num_kv_heads} (vs Q heads: {num_heads})")
        print(f"    KV cache: {kv_total:.2f} GB (seq={max_seq}, batch=1)")
        if num_kv_heads != num_heads:
            print(f"    MHA라면:  {mha_total:.2f} GB ({num_heads/num_kv_heads:.0f}x 더 필요)")


if __name__ == "__main__":
    rope_scaling_demo()

    print("\n" + "=" * 60)
    print("Sliding Window Attention Demo")
    print("=" * 60)
    torch.manual_seed(42)
    Q = torch.randn(1, 8, 16)
    K = torch.randn(1, 8, 16)
    V = torch.randn(1, 8, 16)
    out, attn = sliding_window_attention(Q, K, V, window_size=3)
    print(f"  Input: (1, 8, 16), Window: 3")
    print(f"  Attention pattern (token 5 attends to):")
    nonzero = (attn[0, 5] > 0.01).nonzero().flatten().tolist()
    print(f"    tokens {nonzero} (window_size=3 → positions 2-5)")

    simulate_ring_attention()
    kv_cache_analysis()
