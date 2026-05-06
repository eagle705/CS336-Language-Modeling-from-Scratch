"""
Context Parallelism (CP)
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
"""

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
        - head 수가 cp_size로 나눠져야 함
    """
    P2P = "p2p"
    ALL_GATHER = "all_gather"
    A2A = "a2a"


# ============================================================
# Part 1: Ring Attention 시뮬레이션 (P2P / All-Gather / A2A)
# ============================================================

def ring_attention(Q_chunks, K_chunks, V_chunks, causal=False,
                   comm_type=CPCommType.P2P):
    """
    Context Parallelism 시뮬레이션 — 통신 유형별 구현.

    각 GPU가 Q chunk 하나를 들고, KV를 통신 유형에 따라 교환하며
    attention을 계산.

    Q_chunks: list of (chunk_seq, head_dim) per GPU
    K_chunks: list of (chunk_seq, head_dim) per GPU
    V_chunks: list of (chunk_seq, head_dim) per GPU
    comm_type: CPCommType — P2P, ALL_GATHER, A2A
    """
    if comm_type == CPCommType.P2P:
        return _cp_p2p(Q_chunks, K_chunks, V_chunks, causal)
    elif comm_type == CPCommType.ALL_GATHER:
        return _cp_all_gather(Q_chunks, K_chunks, V_chunks, causal)
    elif comm_type == CPCommType.A2A:
        return _cp_a2a(Q_chunks, K_chunks, V_chunks, causal)
    else:
        raise ValueError(f"Unknown comm_type: {comm_type}")


# -------------------------------------------------------
# (1) P2P: Ring Attention
# -------------------------------------------------------
# KV를 ring 형태로 이웃 GPU에 순차 전달.
# 한 번에 KV chunk 1개만 보유 → 메모리 효율 최고.
# 통신과 연산을 overlap할 수 있어 실전에서 가장 많이 사용.
#
# Timeline (GPU 0 기준):
#   Step 0: compute(Q0, KV0)  |  send KV0→GPU1, recv KV3←GPU3
#   Step 1: compute(Q0, KV3)  |  send KV3→GPU1, recv KV2←GPU3
#   Step 2: compute(Q0, KV2)  |  send KV2→GPU1, recv KV1←GPU3
#   Step 3: compute(Q0, KV1)  |  (마지막, 통신 없음)
# -------------------------------------------------------

def _cp_p2p(Q_chunks, K_chunks, V_chunks, causal):
    """P2P Ring Attention: KV를 ring으로 돌리며 online softmax."""
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    D = Q_chunks[0].shape[-1]
    outputs = []

    for gpu_id in range(num_gpus):
        Q_local = Q_chunks[gpu_id]

        # Online softmax 상태 초기화
        m = torch.full((chunk_size, 1), float('-inf'))  # running max
        l = torch.zeros(chunk_size, 1)                   # running sum(exp)
        O_acc = torch.zeros(chunk_size, D)                # running weighted sum

        # Ring으로 KV chunk 순회 (num_gpus steps)
        for step in range(num_gpus):
            # 실제 분산 환경:
            #   kv_buf = local_kv if step==0 else recv_from(prev_gpu)
            #   if step < num_gpus-1: async_send(kv_buf, next_gpu)
            kv_idx = (gpu_id + step) % num_gpus

            K_block = K_chunks[kv_idx]
            V_block = V_chunks[kv_idx]

            # --- Causal mask 최적화 ---
            if causal and kv_idx > gpu_id:
                continue

            S_block = Q_local @ K_block.T / math.sqrt(D)

            if causal:
                q_positions = torch.arange(gpu_id * chunk_size,
                                           (gpu_id + 1) * chunk_size)
                k_positions = torch.arange(kv_idx * chunk_size,
                                           (kv_idx + 1) * chunk_size)
                causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
                S_block = S_block.masked_fill(~causal_mask, float('-inf'))

            # --- Online softmax update ---
            m_block = S_block.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_block)
            correction = torch.exp(m - m_new)
            P_block = torch.exp(S_block - m_new)
            l_new = correction * l + P_block.sum(dim=-1, keepdim=True)
            O_acc = correction * O_acc + P_block @ V_block
            m = m_new
            l = l_new

        outputs.append(O_acc / l)

    return outputs


# -------------------------------------------------------
# (2) ALL_GATHER: 전체 KV를 한 번에 모은 뒤 local attention
# -------------------------------------------------------
# 모든 GPU가 all_gather로 전체 K, V를 받음.
# 그 후 각 GPU가 자기 Q chunk에 대해 전체 KV로 표준 attention.
#
# 장점: 구현이 매우 단순, 한 번의 collective로 끝
# 단점: 전체 KV를 들고 있어야 해서 메모리 cp_size배
#       gather 완료 후에야 연산 시작 → overlap 불가
#
# 실제 코드 (torch.distributed):
#   K_full = [torch.empty_like(K_local)] * cp_size
#   dist.all_gather(K_full, K_local, group=cp_group)
#   K_full = torch.cat(K_full, dim=0)  # (full_seq, D)
# -------------------------------------------------------

def _cp_all_gather(Q_chunks, K_chunks, V_chunks, causal):
    """All-Gather: 전체 KV를 모은 뒤 각 GPU에서 local attention."""
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    total_seq = chunk_size * num_gpus
    outputs = []

    # all_gather 시뮬레이션: 모든 GPU가 전체 K, V를 갖게 됨
    K_full = torch.cat(K_chunks, dim=0)  # (total_seq, D)
    V_full = torch.cat(V_chunks, dim=0)  # (total_seq, D)

    for gpu_id in range(num_gpus):
        Q_local = Q_chunks[gpu_id]  # (chunk_size, D)

        # 표준 attention: Q_local @ K_full^T
        scores = Q_local @ K_full.T / math.sqrt(Q_local.shape[-1])

        if causal:
            q_positions = torch.arange(gpu_id * chunk_size,
                                       (gpu_id + 1) * chunk_size)
            k_positions = torch.arange(total_seq)
            causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V_full
        outputs.append(out)

    return outputs


# -------------------------------------------------------
# (3) A2A: All-to-All (DeepSpeed-Ulysses 스타일)
# -------------------------------------------------------
# 핵심 아이디어: 시퀀스 분할 → 헤드 분할로 layout 변환.
#
# Before all-to-all:
#   GPU 0: Q[0:S/N, :], K[0:S/N, :], V[0:S/N, :]  (시퀀스 분할)
#   GPU 1: Q[S/N:2S/N, :], K[S/N:2S/N, :], V[S/N:2S/N, :]
#   ...
#
# After all-to-all:
#   GPU 0: Q[:, 0:H/N], K[:, 0:H/N], V[:, 0:H/N]  (헤드 분할)
#   GPU 1: Q[:, H/N:2H/N], K[:, H/N:2H/N], V[:, H/N:2H/N]
#   ...
#
# 각 GPU가 전체 시퀀스의 일부 헤드에 대해 완전한 attention 계산.
# → attention 내부에서 추가 통신 불필요!
# → 결과를 다시 all-to-all로 시퀀스 분할로 되돌림.
#
# 단, single-head 시뮬레이션에서는 헤드 차원이 없으므로
# head_dim을 가상의 헤드로 나눠서 시뮬레이션.
# -------------------------------------------------------

def _cp_a2a(Q_chunks, K_chunks, V_chunks, causal):
    """
    All-to-All (DeepSpeed-Ulysses): 시퀀스 분할 ↔ 헤드 분할 변환.

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
      Q_head_local = torch.cat(output_list, dim=0)  # seq 방향으로 합침
    """
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    num_heads = Q_chunks[0].shape[1]
    head_dim = Q_chunks[0].shape[2]
    total_seq = chunk_size * num_gpus

    assert num_heads % num_gpus == 0, \
        f"num_heads({num_heads}) must be divisible by num_gpus({num_gpus})"
    heads_per_gpu = num_heads // num_gpus

    # ===== Forward All-to-All: seq-split → head-split =====
    Q_full = torch.cat(Q_chunks, dim=0)  # (total_seq, num_heads, head_dim)
    K_full = torch.cat(K_chunks, dim=0)
    V_full = torch.cat(V_chunks, dim=0)

    all_head_outputs = []
    for gpu_id in range(num_gpus):
        h_start = gpu_id * heads_per_gpu
        h_end = h_start + heads_per_gpu

        # 이 GPU가 담당하는 head들 (전체 시퀀스)
        Q_h = Q_full[:, h_start:h_end, :]  # (total_seq, heads_per_gpu, head_dim)
        K_h = K_full[:, h_start:h_end, :]
        V_h = V_full[:, h_start:h_end, :]

        # 각 head에 대해 독립적으로 attention 계산
        head_outs = []
        for h in range(heads_per_gpu):
            q = Q_h[:, h, :]  # (total_seq, head_dim)
            k = K_h[:, h, :]
            v = V_h[:, h, :]

            scores = q @ k.T / math.sqrt(head_dim)
            if causal:
                mask = torch.tril(torch.ones(total_seq, total_seq))
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            head_outs.append(attn @ v)  # (total_seq, head_dim)

        all_head_outputs.append(torch.stack(head_outs, dim=1))  # (total_seq, heads_per_gpu, head_dim)

    # ===== Backward All-to-All: head-split → seq-split =====
    full_output = torch.cat(all_head_outputs, dim=1)  # (total_seq, num_heads, head_dim)
    return list(full_output.chunk(num_gpus, dim=0))  # list of (chunk_seq, num_heads, head_dim)


# ============================================================
# Part 2: 정확성 검증
# ============================================================

def verify_ring_attention():
    """모든 CP 통신 유형이 표준 attention과 동일한 결과를 내는지 검증."""
    print("=" * 60)
    print("Context Parallelism Verification (P2P / All-Gather / A2A)")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    total_seq = 16
    chunk_size = total_seq // num_gpus
    D = 8  # head_dim (num_gpus로 나눠떨어져야 A2A 가능)

    Q = torch.randn(total_seq, D)
    K = torch.randn(total_seq, D)
    V = torch.randn(total_seq, D)

    # --- 표준 attention (정답) ---
    scores = Q @ K.T / math.sqrt(D)
    out_standard = F.softmax(scores, dim=-1) @ V

    # --- 표준 causal attention (정답) ---
    causal_mask = torch.tril(torch.ones(total_seq, total_seq))
    scores_causal = scores.masked_fill(causal_mask == 0, float('-inf'))
    out_causal = F.softmax(scores_causal, dim=-1) @ V

    # chunk 분할
    Q_chunks = list(Q.chunk(num_gpus))
    K_chunks = list(K.chunk(num_gpus))
    V_chunks = list(V.chunk(num_gpus))

    # --- P2P, All-Gather 검증 (single-head) ---
    for comm_type in [CPCommType.P2P, CPCommType.ALL_GATHER]:
        print(f"\n  ── {comm_type.value.upper()} ──")

        # Bidirectional
        cp_outputs = ring_attention(Q_chunks, K_chunks, V_chunks,
                                    causal=False, comm_type=comm_type)
        out_cp = torch.cat(cp_outputs, dim=0)
        diff = (out_standard - out_cp).abs().max().item()
        print(f"    [Bidirectional] diff: {diff:.2e}  {'PASSED' if diff < 1e-5 else 'FAILED'}")

        # Causal
        cp_causal = ring_attention(Q_chunks, K_chunks, V_chunks,
                                   causal=True, comm_type=comm_type)
        out_cp_causal = torch.cat(cp_causal, dim=0)
        diff_c = (out_causal - out_cp_causal).abs().max().item()
        print(f"    [Causal]        diff: {diff_c:.2e}  {'PASSED' if diff_c < 1e-5 else 'FAILED'}")

    # --- A2A 검증 (multi-head, head 분할이므로 num_heads >= num_gpus 필요) ---
    print(f"\n  ── A2A (multi-head) ──")
    num_heads = 8  # num_gpus로 나눠떨어져야 함
    head_dim = 4

    Q_mh = torch.randn(total_seq, num_heads, head_dim)
    K_mh = torch.randn(total_seq, num_heads, head_dim)
    V_mh = torch.randn(total_seq, num_heads, head_dim)

    # 표준 multi-head attention (정답)
    out_mh_standard = []
    out_mh_causal = []
    for h in range(num_heads):
        s = Q_mh[:, h] @ K_mh[:, h].T / math.sqrt(head_dim)
        out_mh_standard.append(F.softmax(s, dim=-1) @ V_mh[:, h])
        s_c = s.masked_fill(torch.tril(torch.ones(total_seq, total_seq)) == 0, float('-inf'))
        out_mh_causal.append(F.softmax(s_c, dim=-1) @ V_mh[:, h])
    out_mh_standard = torch.stack(out_mh_standard, dim=1)  # (total_seq, num_heads, head_dim)
    out_mh_causal = torch.stack(out_mh_causal, dim=1)

    # A2A: seq 분할 → head 분할 → attention → head 분할 → seq 분할
    Q_mh_chunks = list(Q_mh.chunk(num_gpus, dim=0))
    K_mh_chunks = list(K_mh.chunk(num_gpus, dim=0))
    V_mh_chunks = list(V_mh.chunk(num_gpus, dim=0))

    a2a_out = _cp_a2a(Q_mh_chunks, K_mh_chunks, V_mh_chunks, causal=False)
    out_a2a = torch.cat(a2a_out, dim=0)
    diff_a2a = (out_mh_standard - out_a2a).abs().max().item()
    print(f"    [Bidirectional] diff: {diff_a2a:.2e}  {'PASSED' if diff_a2a < 1e-5 else 'FAILED'}")

    a2a_causal = _cp_a2a(Q_mh_chunks, K_mh_chunks, V_mh_chunks, causal=True)
    out_a2a_c = torch.cat(a2a_causal, dim=0)
    diff_a2a_c = (out_mh_causal - out_a2a_c).abs().max().item()
    print(f"    [Causal]        diff: {diff_a2a_c:.2e}  {'PASSED' if diff_a2a_c < 1e-5 else 'FAILED'}")
    print(f"    (num_heads={num_heads}, heads_per_gpu={num_heads//num_gpus})")

    # Causal에서 P2P의 skip 분석
    total_blocks = num_gpus * num_gpus
    skipped = 0
    for gpu_id in range(num_gpus):
        for step in range(num_gpus):
            kv_idx = (gpu_id + step) % num_gpus
            if kv_idx > gpu_id:
                skipped += 1
    print(f"\n  P2P Causal mask 최적화:")
    print(f"    전체 QK blocks: {total_blocks}")
    print(f"    Skip된 blocks:  {skipped} ({skipped/total_blocks*100:.0f}%)")
    print(f"    → causal mask로 거의 절반의 연산 절약!")


# ============================================================
# Part 3: 통신 분석
# ============================================================

def communication_analysis():
    """CP의 통신량과 overlap 분석."""
    print("\n" + "=" * 60)
    print("Context Parallelism Communication Analysis")
    print("=" * 60)

    cp_size = 4
    seq_len = 131072   # 128K
    num_heads = 32
    head_dim = 128
    num_layers = 32
    batch = 1
    bf16 = 2

    chunk_seq = seq_len // cp_size  # 32K per GPU

    # KV chunk size: (chunk_seq, num_heads, head_dim) × 2 (K+V)
    kv_chunk_bytes = chunk_seq * num_heads * head_dim * bf16 * 2
    # Ring에서 cp_size-1번 전송
    kv_per_layer = kv_chunk_bytes * (cp_size - 1)
    kv_total = kv_per_layer * num_layers

    # Attention scores per GPU: (chunk_seq, chunk_seq) per head per KV step
    attn_mem_standard = seq_len * seq_len * num_heads * bf16 / 1e9  # 전체
    attn_mem_cp = chunk_seq * chunk_seq * num_heads * bf16 / 1e9    # CP 시

    print(f"\n  Config: seq={seq_len}, CP={cp_size}, heads={num_heads}")
    print(f"  Chunk per GPU: {chunk_seq} tokens")

    print(f"\n  Attention memory (per layer, single head):")
    print(f"    No CP:  {seq_len}×{seq_len} = {seq_len**2/1e6:.0f}M entries"
          f" ({attn_mem_standard:.1f} GB total)")
    print(f"    CP={cp_size}: {chunk_seq}×{chunk_seq} = {chunk_seq**2/1e6:.0f}M entries"
          f" ({attn_mem_cp:.2f} GB per GPU)")
    print(f"    절약: {attn_mem_standard / attn_mem_cp:.0f}x (= CP_size^2)")

    print(f"\n  Communication (per layer):")
    print(f"    KV chunk: {kv_chunk_bytes / 1e6:.1f} MB")
    print(f"    Ring steps: {cp_size - 1}")
    print(f"    Total: {kv_per_layer / 1e6:.1f} MB per layer")
    print(f"    All layers: {kv_total / 1e9:.2f} GB per step")

    print(f"\n  Overlap 전략:")
    print( "    Step i:  GPU에서 KV_i로 attention 계산")
    print( "             동시에 KV_{i+1}을 다음 GPU로부터 recv")
    print( "    → 통신이 연산에 완전히 숨겨짐 (연산 > 통신이면)")

    # 연산 vs 통신 비교
    # attention 연산: 2 * chunk_seq * chunk_seq * head_dim * num_heads
    compute_flops = 2 * chunk_seq * chunk_seq * head_dim * num_heads * batch
    compute_tflops = compute_flops / 1e12
    transfer_gb = kv_chunk_bytes / 1e9

    print(f"\n  연산 vs 통신 (per ring step, per layer):")
    print(f"    Compute: {compute_tflops:.2f} TFLOP")
    print(f"    Transfer: {transfer_gb:.3f} GB")
    print(f"    H100 기준: {compute_tflops*1000/990:.1f}ms compute"
          f" vs {transfer_gb*1000/900:.1f}ms transfer (NVLink)")
    print(f"    → {'Compute-bound (overlap 가능!)' if compute_tflops/990 > transfer_gb/900 else 'Transfer-bound'}")

    # ========== 통신 유형별 비교 ==========
    print(f"\n  {'─' * 56}")
    print(f"  CP 통신 유형별 비교 (seq={seq_len}, CP={cp_size})")
    print(f"  {'─' * 56}")

    # QKV 한 세트: (chunk_seq, num_heads, head_dim) × 3
    qkv_chunk_bytes = chunk_seq * num_heads * head_dim * bf16 * 3

    # --- P2P (Ring) ---
    p2p_per_step = kv_chunk_bytes  # send KV chunk to neighbor
    p2p_total_per_layer = p2p_per_step * (cp_size - 1)
    p2p_peak_mem = kv_chunk_bytes  # KV chunk 1개만 추가 보유

    # --- All-Gather ---
    ag_total_per_layer = kv_chunk_bytes * (cp_size - 1)  # all_gather = 각자 chunk 보냄
    ag_peak_mem = kv_chunk_bytes * cp_size  # 전체 KV 보유

    # --- A2A (Ulysses) ---
    # forward: Q,K,V 각각 all-to-all (seq→head), backward: output all-to-all (head→seq)
    a2a_per_call = qkv_chunk_bytes * (cp_size - 1) / cp_size  # all-to-all 전송량
    a2a_total_per_layer = a2a_per_call * 2  # forward + backward all-to-all
    a2a_peak_mem = chunk_seq * num_heads * head_dim * bf16  # 추가 메모리 적음

    print(f"""
  ┌──────────────┬───────────────┬───────────────┬──────────────────────┐
  │ 유형          │ 통신량/layer   │ Peak 추가 메모리│ Overlap 가능 여부     │
  ├──────────────┼───────────────┼───────────────┼──────────────────────┤
  │ P2P (Ring)   │ {p2p_total_per_layer/1e6:>8.1f} MB │ {p2p_peak_mem/1e6:>8.1f} MB │ Yes (compute+comm)  │
  │ All-Gather   │ {ag_total_per_layer/1e6:>8.1f} MB │ {ag_peak_mem/1e6:>8.1f} MB │ No (gather→compute) │
  │ A2A (Ulysses)│ {a2a_total_per_layer/1e6:>8.1f} MB │ {a2a_peak_mem/1e6:>8.1f} MB │ Partial             │
  └──────────────┴───────────────┴───────────────┴──────────────────────┘

  선택 가이드:
    P2P (Ring):    CP_size가 크고, 긴 시퀀스에서 overlap 효과 극대화
                   Megatron-Core, PyTorch FSDP2 기본 방식
    All-Gather:    CP_size 작고 (2~4), 단순 구현 원할 때
                   메모리 여유 있으면 가장 쉬운 선택
    A2A (Ulysses): head 수가 충분하고, 노드 내 NVLink에서 효율적
                   DeepSpeed-Ulysses, Megatron-Core (--cp-comm-type=a2a)""")


# ============================================================
# Part 4: 4D Parallelism (TP × CP × PP × DP)
# ============================================================

def parallelism_4d():
    """4D parallelism 구성."""
    print("\n" + "=" * 60)
    print("4D Parallelism: TP × CP × PP × DP")
    print("=" * 60)

    print("""
  최신 대규모 학습은 4D parallelism 사용:

    Total GPUs = TP × CP × PP × DP

  예: LLaMA-3 405B on 16384 GPUs
    TP = 8    (노드 내 NVLink, layer 내부 weight split)
    CP = 2    (시퀀스를 2등분, ring attention)
    PP = 16   (32 layers를 16 stage로)
    DP = 64   (데이터 병렬)
    → 8 × 2 × 16 × 64 = 16,384 GPUs

  각 parallelism이 나누는 차원:
    ┌──────────┬────────────┬──────────────────────────┐
    │          │ 나누는 것   │ 통신                      │
    ├──────────┼────────────┼──────────────────────────┤
    │ TP       │ hidden dim │ all-reduce (NVLink)       │
    │ CP       │ seq dim    │ ring send/recv (NVLink)   │
    │ PP       │ layers     │ point-to-point (IB 가능)  │
    │ DP       │ data       │ all-reduce (IB)           │
    └──────────┴────────────┴──────────────────────────┘

  DeviceMesh 구성:
    mesh = init_device_mesh("cuda",
        (dp_size, pp_size, cp_size, tp_size),
        mesh_dim_names=("dp", "pp", "cp", "tp"),
    )

  Megatron-Core에서:
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=8,
        context_parallel_size=2,
        pipeline_model_parallel_size=16,
        # DP는 자동 계산
    )
    """)

    # GPU 배치 예시
    tp, cp, pp, dp = 2, 2, 2, 2
    total = tp * cp * pp * dp
    print(f"  예시: {tp}TP × {cp}CP × {pp}PP × {dp}DP = {total} GPUs\n")

    print(f"  {'GPU':>4} {'TP':>4} {'CP':>4} {'PP':>4} {'DP':>4}   역할")
    print(f"  {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}   {'-'*30}")

    for rank in range(total):
        tp_r = rank % tp
        cp_r = (rank // tp) % cp
        pp_r = (rank // (tp * cp)) % pp
        dp_r = rank // (tp * cp * pp)

        roles = []
        if tp_r == 0 and cp_r == 0:
            roles.append(f"PP stage {pp_r}")
        if tp_r == 0:
            roles.append(f"seq chunk {cp_r}")

        role = ", ".join(roles) if roles else ""
        print(f"  {rank:>4} {tp_r:>4} {cp_r:>4} {pp_r:>4} {dp_r:>4}   {role}")


# ============================================================
# Part 5: CP vs 다른 접근법 비교
# ============================================================

def cp_comparison():
    print("\n" + "=" * 60)
    print("Context Parallelism vs Alternatives")
    print("=" * 60)

    print("""
  긴 시퀀스를 처리하는 방법들:

  ┌───────────────────┬────────────┬────────────┬──────────────────┐
  │ 방법               │ 메모리     │ 정확도     │ 구현 복잡도       │
  ├───────────────────┼────────────┼────────────┼──────────────────┤
  │ Flash Attention    │ O(S)      │ 정확       │ 낮음 (라이브러리) │
  │ Sliding Window     │ O(S×W)    │ 근사       │ 낮음             │
  │ Gradient Ckpt      │ 절반      │ 정확       │ 낮음             │
  │ Context Parallel   │ O(S²/N²)  │ 정확       │ 높음             │
  │ Ring Attention     │ O(S²/N²)  │ 정확       │ 높음             │
  └───────────────────┴────────────┴────────────┴──────────────────┘

  조합해서 사용:
    Flash Attention + CP가 가장 효과적
      Flash: 단일 GPU 내에서 O(S) 메모리
      CP:    GPU 간에 시퀀스 분산 → 각 GPU의 S가 S/N으로 감소

  언제 CP를 쓰나?
    - seq_len > 32K 이상일 때 (activation 메모리 문제)
    - Flash Attention만으로 부족할 때
    - 예: 128K context → CP=4 → 각 GPU 32K → Flash로 충분히 처리
    """)


if __name__ == "__main__":
    verify_ring_attention()
    communication_analysis()
    parallelism_4d()
    cp_comparison()
