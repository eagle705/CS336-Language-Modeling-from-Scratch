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


# ============================================================
# Part 1: Ring Attention 시뮬레이션
# ============================================================

def ring_attention(Q_chunks, K_chunks, V_chunks, causal=False):
    """
    Ring Attention을 시뮬레이션.

    각 GPU가 Q chunk 하나를 들고, KV chunk를 ring으로 돌리며
    online softmax로 attention을 점진적으로 계산.

    Q_chunks: list of (chunk_seq, head_dim) per GPU
    K_chunks: list of (chunk_seq, head_dim) per GPU
    V_chunks: list of (chunk_seq, head_dim) per GPU
    """
    num_gpus = len(Q_chunks)
    chunk_size = Q_chunks[0].shape[0]
    D = Q_chunks[0].shape[-1]
    outputs = []

    for gpu_id in range(num_gpus):
        Q_local = Q_chunks[gpu_id]  # 이 GPU의 Q

        # Online softmax 상태 초기화
        m = torch.full((chunk_size, 1), float('-inf'))  # running max
        l = torch.zeros(chunk_size, 1)                   # running sum(exp)
        O_acc = torch.zeros(chunk_size, D)                # running weighted sum

        # Ring으로 KV chunk 순회 (num_gpus steps)
        for step in range(num_gpus):
            # step번째에 받는 KV chunk의 인덱스
            kv_idx = (gpu_id + step) % num_gpus

            K_block = K_chunks[kv_idx]
            V_block = V_chunks[kv_idx]

            # --- Causal mask 최적화 ---
            # Q chunk의 위치: [gpu_id * chunk_size, (gpu_id+1) * chunk_size)
            # KV chunk의 위치: [kv_idx * chunk_size, (kv_idx+1) * chunk_size)
            # Q의 모든 토큰이 KV보다 앞에 있으면 → 이 KV chunk 전부 skip!
            if causal and kv_idx > gpu_id:
                # 이 KV chunk는 Q보다 미래 → 완전히 skip (연산 절약!)
                continue

            # Block attention score: Q_local @ K_block^T / sqrt(D)
            S_block = Q_local @ K_block.T / math.sqrt(D)  # (chunk, chunk)

            # Causal mask: Q 위치 < K 위치인 곳을 -inf
            if causal:
                q_positions = torch.arange(gpu_id * chunk_size,
                                           (gpu_id + 1) * chunk_size)
                k_positions = torch.arange(kv_idx * chunk_size,
                                           (kv_idx + 1) * chunk_size)
                causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
                S_block = S_block.masked_fill(~causal_mask, float('-inf'))

            # --- Online softmax update ---
            # (Flash Attention과 동일한 트릭)
            #
            # 새 block의 max:
            m_block = S_block.max(dim=-1, keepdim=True).values  # (chunk, 1)
            m_new = torch.maximum(m, m_block)

            # 기존 누적값을 새 max에 맞게 보정
            # exp(m_old - m_new) < 1 → 기존 값을 줄임
            correction = torch.exp(m - m_new)

            # 새 block의 exp(score - max)
            P_block = torch.exp(S_block - m_new)

            # Running sum 업데이트
            l_new = correction * l + P_block.sum(dim=-1, keepdim=True)

            # Running weighted sum 업데이트
            O_acc = correction * O_acc + P_block @ V_block

            m = m_new
            l = l_new

        # 최종 정규화: O / l
        outputs.append(O_acc / l)

    return outputs


# ============================================================
# Part 2: 정확성 검증
# ============================================================

def verify_ring_attention():
    """Ring Attention이 표준 attention과 동일한 결과를 내는지 검증."""
    print("=" * 60)
    print("Ring Attention Verification")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    total_seq = 16
    chunk_size = total_seq // num_gpus
    D = 8

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

    # --- Ring Attention (bidirectional) ---
    ring_outputs = ring_attention(Q_chunks, K_chunks, V_chunks, causal=False)
    out_ring = torch.cat(ring_outputs, dim=0)
    diff = (out_standard - out_ring).abs().max().item()
    print(f"\n  [Bidirectional] seq={total_seq}, GPUs={num_gpus}, chunk={chunk_size}")
    print(f"    Standard vs Ring diff: {diff:.2e}")
    print(f"    Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")

    # --- Ring Attention (causal) ---
    ring_causal = ring_attention(Q_chunks, K_chunks, V_chunks, causal=True)
    out_ring_causal = torch.cat(ring_causal, dim=0)
    diff_causal = (out_causal - out_ring_causal).abs().max().item()
    print(f"\n  [Causal] seq={total_seq}, GPUs={num_gpus}, chunk={chunk_size}")
    print(f"    Standard vs Ring diff: {diff_causal:.2e}")
    print(f"    Result: {'PASSED' if diff_causal < 1e-5 else 'FAILED'}")

    # Causal에서 skip된 연산량
    total_blocks = num_gpus * num_gpus
    skipped = 0
    for gpu_id in range(num_gpus):
        for step in range(num_gpus):
            kv_idx = (gpu_id + step) % num_gpus
            if kv_idx > gpu_id:
                skipped += 1
    print(f"\n  Causal mask 최적화:")
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
