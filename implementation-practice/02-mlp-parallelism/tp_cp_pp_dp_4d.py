"""
4D Parallelism: TP × CP × PP × DP
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
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# Part 1: 4D Process Group 구성
# ============================================================

def build_4d_process_groups(tp_size, cp_size, pp_size, dp_size):
    """
    4D parallelism process group을 구성.

    rank 배치: [DP][PP][CP][TP]
    rank = dp*(pp*cp*tp) + pp*(cp*tp) + cp*tp + tp_rank
    """
    world_size = tp_size * cp_size * pp_size * dp_size

    rank_info = {}
    for rank in range(world_size):
        tp_r = rank % tp_size
        cp_r = (rank // tp_size) % cp_size
        pp_r = (rank // (tp_size * cp_size)) % pp_size
        dp_r = rank // (tp_size * cp_size * pp_size)
        rank_info[rank] = {"tp": tp_r, "cp": cp_r, "pp": pp_r, "dp": dp_r}

    groups = {"tp": [], "cp": [], "pp": [], "dp": []}

    # TP group: 같은 dp, pp, cp 내에서 tp 다른 GPU들
    for dp in range(dp_size):
        for pp in range(pp_size):
            for cp in range(cp_size):
                ranks = [dp * (pp_size * cp_size * tp_size)
                         + pp * (cp_size * tp_size)
                         + cp * tp_size + tp
                         for tp in range(tp_size)]
                groups["tp"].append(ranks)

    # CP group: 같은 dp, pp, tp_rank 내에서 cp 다른 GPU들
    for dp in range(dp_size):
        for pp in range(pp_size):
            for tp in range(tp_size):
                ranks = [dp * (pp_size * cp_size * tp_size)
                         + pp * (cp_size * tp_size)
                         + cp * tp_size + tp
                         for cp in range(cp_size)]
                groups["cp"].append(ranks)

    # PP group: 같은 dp, cp, tp_rank 내에서 pp 다른 GPU들
    for dp in range(dp_size):
        for cp in range(cp_size):
            for tp in range(tp_size):
                ranks = [dp * (pp_size * cp_size * tp_size)
                         + pp * (cp_size * tp_size)
                         + cp * tp_size + tp
                         for pp in range(pp_size)]
                groups["pp"].append(ranks)

    # DP group: 같은 pp, cp, tp_rank 내에서 dp 다른 GPU들
    for pp in range(pp_size):
        for cp in range(cp_size):
            for tp in range(tp_size):
                ranks = [dp * (pp_size * cp_size * tp_size)
                         + pp * (cp_size * tp_size)
                         + cp * tp_size + tp
                         for dp in range(dp_size)]
                groups["dp"].append(ranks)

    return groups, rank_info


# ============================================================
# Part 2: 단일 Attention layer의 4D 시뮬레이션
# ============================================================

def ring_attention_with_tp(Q_full, K_full, V_full, cp_size, tp_size):
    """
    TP + CP를 동시에 적용한 attention 시뮬레이션.

    CP: seq를 cp_size 등분 → 각 CP rank가 Q chunk 담당
    TP: head를 tp_size 등분 → 각 TP rank가 head subset 담당
    Ring Attention: CP rank 간 KV chunk를 ring으로 순회

    Q_full: (seq, num_heads, head_dim)
    """
    seq_len, num_heads, head_dim = Q_full.shape
    chunk_seq = seq_len // cp_size
    heads_per_tp = num_heads // tp_size

    # seq 분할 (CP)
    Q_cp = [Q_full[i * chunk_seq:(i + 1) * chunk_seq] for i in range(cp_size)]
    K_cp = [K_full[i * chunk_seq:(i + 1) * chunk_seq] for i in range(cp_size)]
    V_cp = [V_full[i * chunk_seq:(i + 1) * chunk_seq] for i in range(cp_size)]

    all_outputs = []

    for cp_rank in range(cp_size):
        # 이 CP rank의 head를 TP로 분할
        tp_outputs = []

        for tp_rank in range(tp_size):
            # 이 TP rank가 담당하는 head subset
            h_start = tp_rank * heads_per_tp
            h_end = h_start + heads_per_tp

            Q_local = Q_cp[cp_rank][:, h_start:h_end, :]   # (chunk, heads/tp, d)
            # KV도 동일 head subset만
            K_chunks = [K_cp[i][:, h_start:h_end, :] for i in range(cp_size)]
            V_chunks = [V_cp[i][:, h_start:h_end, :] for i in range(cp_size)]

            # Ring Attention (causal): 이 CP rank의 Q에 대해 모든 KV 순회
            CS, H, D = Q_local.shape
            m = torch.full((CS, H, 1), float('-inf'))
            l = torch.zeros(CS, H, 1)
            O_acc = torch.zeros(CS, H, D)

            for step in range(cp_size):
                kv_idx = (cp_rank + step) % cp_size

                # Causal: 미래 KV chunk skip
                if kv_idx > cp_rank:
                    continue

                K_block = K_chunks[kv_idx]
                V_block = V_chunks[kv_idx]

                # attention score: (chunk, heads, chunk)
                # Q(CS,H,D) @ K(CS,H,D).T → (CS,H,CS) via einsum
                S_block = torch.einsum('qhd,khd->hqk', Q_local, K_block) / math.sqrt(D)
                # → (H, CS_q, CS_k) → transpose to (CS_q, H, CS_k)
                S_block = S_block.permute(1, 0, 2)  # (CS_q, H, CS_k)

                # Causal mask
                q_pos = torch.arange(cp_rank * chunk_seq, (cp_rank + 1) * chunk_seq)
                k_pos = torch.arange(kv_idx * chunk_seq, (kv_idx + 1) * chunk_seq)
                mask = q_pos.unsqueeze(-1) >= k_pos.unsqueeze(0)  # (CS_q, CS_k)
                S_block = S_block.masked_fill(~mask.unsqueeze(1), float('-inf'))

                # Online softmax
                m_block = S_block.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m, m_block)
                correction = torch.exp(m - m_new)
                P_block = torch.exp(S_block - m_new)
                l = correction * l + P_block.sum(dim=-1, keepdim=True)
                # P_block(CS_q,H,CS_k) @ V_block(CS_k,H,D) → need einsum
                O_acc = correction * O_acc + torch.einsum('qhk,khd->qhd', P_block, V_block)
                m = m_new

            tp_outputs.append(O_acc / l)  # (chunk_seq, heads_per_tp, head_dim)

        # TP: head subset을 concat (실제로는 각 TP rank에 분산)
        # 여기서는 시뮬레이션이므로 concat
        cp_output = torch.cat(tp_outputs, dim=1)  # (chunk_seq, num_heads, head_dim)
        all_outputs.append(cp_output)

    # CP: seq chunk를 concat
    return torch.cat(all_outputs, dim=0)  # (seq, num_heads, head_dim)


# ============================================================
# Part 3: FFN의 TP 시뮬레이션
# ============================================================

def ffn_with_tp(x, W1, W2, tp_size):
    """
    FFN에 TP를 적용. (CP 내 각 seq chunk에 독립 적용)

    x: (seq_chunk, embed_dim)
    W1: (embed_dim, ffn_hidden) → column split
    W2: (ffn_hidden, embed_dim) → row split
    """
    ffn_hidden = W1.shape[1]
    half = ffn_hidden // tp_size

    partial_outputs = []
    for tp_rank in range(tp_size):
        W1_shard = W1[:, tp_rank * half:(tp_rank + 1) * half]
        W2_shard = W2[tp_rank * half:(tp_rank + 1) * half, :]
        a = F.gelu(x @ W1_shard)
        partial_outputs.append(a @ W2_shard)

    # TP all-reduce
    return sum(partial_outputs)


# ============================================================
# Part 4: 전체 4D Forward 시뮬레이션
# ============================================================

def simulate_4d_forward():
    """TP + CP + PP + DP를 모두 적용한 forward pass 시뮬레이션."""
    print("=" * 70)
    print("4D Parallelism Forward Simulation: TP × CP × PP × DP")
    print("=" * 70)

    torch.manual_seed(42)

    # --- Config ---
    tp_size, cp_size, pp_size, dp_size = 2, 2, 2, 2
    world_size = tp_size * cp_size * pp_size * dp_size

    embed_dim = 16
    num_heads = 4
    head_dim = embed_dim // num_heads
    ffn_hidden = 32
    num_layers = 4
    layers_per_stage = num_layers // pp_size
    seq_len = 16
    batch = 1  # 간소화

    print(f"\n  Config: {tp_size}TP × {cp_size}CP × {pp_size}PP × {dp_size}DP"
          f" = {world_size} GPUs")
    print(f"  Model: {num_layers}L, embed={embed_dim}, heads={num_heads},"
          f" ffn={ffn_hidden}, seq={seq_len}")

    # --- Process groups ---
    groups, rank_info = build_4d_process_groups(tp_size, cp_size, pp_size, dp_size)

    print(f"\n  Process Groups (총 {world_size} GPUs):")
    print(f"    TP groups ({len(groups['tp'])}): {groups['tp'][:4]} ...")
    print(f"    CP groups ({len(groups['cp'])}): {groups['cp'][:4]} ...")
    print(f"    PP groups ({len(groups['pp'])}): {groups['pp'][:4]} ...")
    print(f"    DP groups ({len(groups['dp'])}): {groups['dp'][:4]} ...")

    # --- GPU 할당 테이블 ---
    print(f"\n  GPU Assignments:")
    print(f"    {'GPU':>4} {'TP':>4} {'CP':>4} {'PP':>4} {'DP':>4}")
    print(f"    {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for rank in range(world_size):
        info = rank_info[rank]
        print(f"    {rank:>4} {info['tp']:>4} {info['cp']:>4}"
              f" {info['pp']:>4} {info['dp']:>4}")

    # --- 모델 weights ---
    # Attention: Wq, Wk, Wv, Wo per layer
    Wq = [torch.randn(embed_dim, embed_dim) * 0.1 for _ in range(num_layers)]
    Wk = [torch.randn(embed_dim, embed_dim) * 0.1 for _ in range(num_layers)]
    Wv = [torch.randn(embed_dim, embed_dim) * 0.1 for _ in range(num_layers)]
    Wo = [torch.randn(embed_dim, embed_dim) * 0.1 for _ in range(num_layers)]
    # FFN: W1, W2 per layer
    W1 = [torch.randn(embed_dim, ffn_hidden) * 0.1 for _ in range(num_layers)]
    W2 = [torch.randn(ffn_hidden, embed_dim) * 0.1 for _ in range(num_layers)]

    # --- 단일 GPU 기준 (causal attention + FFN) ---
    x_ref = torch.randn(seq_len, embed_dim)
    x_single = x_ref.clone()

    for layer_idx in range(num_layers):
        # Self-attention (단일 GPU)
        Q = (x_single @ Wq[layer_idx]).view(seq_len, num_heads, head_dim)
        K = (x_single @ Wk[layer_idx]).view(seq_len, num_heads, head_dim)
        V = (x_single @ Wv[layer_idx]).view(seq_len, num_heads, head_dim)

        scores = torch.einsum('qhd,khd->hqk', Q, K) / math.sqrt(head_dim)
        causal = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        scores = scores.masked_fill(causal == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.einsum('hqk,khd->qhd', attn, V)
        attn_out = attn_out.reshape(seq_len, embed_dim) @ Wo[layer_idx]
        x_single = x_single + attn_out

        # FFN
        ffn_out = F.gelu(x_single @ W1[layer_idx]) @ W2[layer_idx]
        x_single = x_single + ffn_out

    # --- 4D Parallelism Forward ---
    print(f"\n  === Forward Pass (DP group 0) ===")
    x = x_ref.clone()

    for stage in range(pp_size):
        start_l = stage * layers_per_stage
        end_l = start_l + layers_per_stage
        print(f"\n    PP Stage {stage} (layers {start_l}-{end_l - 1}):")

        for layer_idx in range(start_l, end_l):
            # --- Attention with TP + CP ---
            Q = (x @ Wq[layer_idx]).view(seq_len, num_heads, head_dim)
            K = (x @ Wk[layer_idx]).view(seq_len, num_heads, head_dim)
            V = (x @ Wv[layer_idx]).view(seq_len, num_heads, head_dim)

            # CP + TP ring attention
            attn_out = ring_attention_with_tp(Q, K, V, cp_size, tp_size)
            attn_out = attn_out.reshape(seq_len, embed_dim) @ Wo[layer_idx]
            x = x + attn_out

            # --- FFN with TP (CP chunks 독립) ---
            ffn_out = ffn_with_tp(x, W1[layer_idx], W2[layer_idx], tp_size)
            x = x + ffn_out

            print(f"      Layer {layer_idx}:"
                  f" Attn(CP={cp_size} ring × TP={tp_size} heads)"
                  f" → FFN(TP={tp_size} column/row)")

        if stage < pp_size - 1:
            print(f"      → PP send activation to stage {stage + 1}")

    # --- 검증 ---
    diff = (x_single - x).abs().max().item()
    print(f"\n  Single GPU vs 4D Parallel diff: {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-4 else 'FAILED'}")

    return diff


# ============================================================
# Part 5: 통신 패턴 요약
# ============================================================

def communication_summary():
    print("\n" + "=" * 70)
    print("4D Communication Pattern Summary")
    print("=" * 70)

    print("""
  Forward (각 layer):
    ┌──────────────────────────────────────────────────────────┐
    │  1. Attention                                            │
    │     CP: Ring Attention                                   │
    │         각 ring step에서 KV chunk를 이웃 CP rank에 전달    │
    │         (send/recv, cp_size-1 steps)                     │
    │     TP: head별 독립 계산 → all-reduce 불필요 (concat)     │
    │         Wo projection 후 TP all-reduce                   │
    │                                                          │
    │  2. FFN                                                  │
    │     TP: ColumnParallel(fc1) → GELU → RowParallel(fc2)   │
    │         fc2 끝에서 TP all-reduce 1회                     │
    │     CP: 각 seq chunk가 독립적으로 FFN 계산 (통신 없음)     │
    └──────────────────────────────────────────────────────────┘

  PP stage 경계:
    activation send/recv (point-to-point)

  Backward 후:
    DP all-reduce (gradient 동기화)

  통신 횟수 (per layer):
    TP: all-reduce 2회 (attn output + FFN output)
    CP: send/recv (cp_size-1) × 2 (KV fwd + grad bwd)
    PP: send/recv 1회 per stage boundary
    DP: all-reduce 1회 (전체 step 끝)
    """)


# ============================================================
# Part 6: 메모리 & 통신량 분석
# ============================================================

def detailed_analysis():
    print("=" * 70)
    print("4D Parallelism: Memory & Communication")
    print("=" * 70)

    # LLaMA-3.1 405B 규모
    tp, cp, pp, dp = 8, 2, 16, 64
    total = tp * cp * pp * dp
    P = 405  # B params
    D = 16384
    H = 128  # num_heads
    Dh = 128 # head_dim
    FF = 53248
    L = 126
    S = 131072  # 128K context
    B = 1
    bf16 = 2

    layers_per_stage = L // pp

    print(f"\n  Model: {P}B params, {L} layers, D={D}, heads={H}")
    print(f"  Parallelism: {tp}TP × {cp}CP × {pp}PP × {dp}DP = {total:,} GPUs")
    print(f"  Sequence: {S:,} tokens (CP → {S // cp:,} per GPU)")

    # 메모리
    P_per_gpu = P / (tp * pp)
    params_gb = P_per_gpu * bf16
    opt_gb = P_per_gpu * 4 * 2  # Adam FP32 m,v
    master_gb = P_per_gpu * 4
    grad_gb = P_per_gpu * bf16

    print(f"\n  Memory per GPU:")
    print(f"    Parameters (BF16):  {params_gb:.2f} GB")
    print(f"    Gradients (BF16):   {grad_gb:.2f} GB")
    print(f"    Optimizer (FP32):   {opt_gb:.2f} GB")
    print(f"    Master wt (FP32):   {master_gb:.2f} GB")
    print(f"    Total model state:  {params_gb + grad_gb + opt_gb + master_gb:.2f} GB")

    # Activation (per micro-batch, per stage)
    chunk_seq = S // cp
    act_per_layer = B * chunk_seq * D * bf16 * 4 / 1e9  # ~4 tensors
    attn_score = B * (H // tp) * chunk_seq * chunk_seq * bf16 / 1e9
    act_total = (act_per_layer + attn_score) * layers_per_stage
    # Flash Attention 적용 시 attn S^2 → O(S) 로 줄어듦
    attn_score_flash = B * (H // tp) * chunk_seq * bf16 / 1e9  # Flash: O(S)
    act_total_flash = (act_per_layer + attn_score_flash) * layers_per_stage

    print(f"\n    Activations per stage:")
    print(f"      Per layer (hidden):   {act_per_layer * 1000:.1f} MB")
    print(f"      Per layer (attn):     {attn_score * 1000:.1f} MB (naive S^2)")
    print(f"      Per layer (flash):    {attn_score_flash * 1000:.2f} MB (Flash Attn O(S))")
    print(f"      Total naive ({layers_per_stage} layers): {act_total:.2f} GB")
    print(f"      Total flash ({layers_per_stage} layers): {act_total_flash:.2f} GB ← 실전")

    # 통신량
    print(f"\n  Communication per step:")

    tp_per_layer = B * chunk_seq * D * bf16 * 2  # fwd + bwd, 2 all-reduces
    tp_actual = tp_per_layer * 2 * (tp - 1) / tp
    tp_total = tp_actual * layers_per_stage
    print(f"    TP: {tp_total / 1e9:.2f} GB (all-reduce × {layers_per_stage} layers)")

    kv_chunk = B * chunk_seq * (H // tp) * Dh * bf16 * 2  # K+V
    cp_per_layer = kv_chunk * (cp - 1) * 2  # ring steps, fwd+bwd
    cp_total = cp_per_layer * layers_per_stage
    print(f"    CP: {cp_total / 1e9:.2f} GB (ring KV × {layers_per_stage} layers)")

    pp_per_mb = B * chunk_seq * D * bf16
    num_mb = 8  # micro-batches
    pp_total = pp_per_mb * num_mb * 2  # fwd + bwd
    print(f"    PP: {pp_total / 1e9:.3f} GB (activation × {num_mb} micro-batches)")

    dp_total = P_per_gpu * 1e9 * bf16 * 2 * (dp - 1) / dp
    print(f"    DP: {dp_total / 1e9:.2f} GB (gradient all-reduce)")


# ============================================================
# Part 7: DeviceMesh / Megatron-Core 설정
# ============================================================

def setup_guide():
    print("\n" + "=" * 70)
    print("4D Setup Guide")
    print("=" * 70)

    print("""
  --- PyTorch DeviceMesh ---

    mesh = init_device_mesh("cuda",
        (dp_size, pp_size, cp_size, tp_size),
        mesh_dim_names=("dp", "pp", "cp", "tp"),
    )

    # TP 적용
    parallelize_module(block.ffn, mesh["tp"], {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    })

    # CP: 현재 PyTorch native에서는 직접 Ring Attention 구현 필요
    # 또는 Megatron-Core의 context_parallel 사용

    # PP 적용
    pipe = pipeline(model, mb_args=(...),
                    split_spec={"layers.N": SplitPoint.BEGINNING})

    # DP (FSDP2)
    fully_shard(block, mesh=mesh["dp"])

  --- Megatron-Core ---

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=8,
        context_parallel_size=2,
        pipeline_model_parallel_size=16,
        # DP = world_size / (8 * 2 * 16) = 자동 계산
    )

    config = TransformerConfig(
        tensor_model_parallel_size=8,
        context_parallel_size=2,
        pipeline_model_parallel_size=16,
        sequence_parallel=True,
        ...
    )
    # Megatron-Core가 내부적으로 Ring Attention 구현 포함

  --- torchrun 실행 ---

    # 단일 노드 (8 GPU)
    torchrun --nproc_per_node=8 train.py

    # 멀티 노드 (예: 256 GPU = 32 nodes × 8 GPUs)
    srun --nodes=32 --ntasks-per-node=8 --gpus-per-node=8 \\
        torchrun --nproc_per_node=8 train.py
    """)


if __name__ == "__main__":
    diff = simulate_4d_forward()
    communication_summary()
    detailed_analysis()
    setup_guide()
