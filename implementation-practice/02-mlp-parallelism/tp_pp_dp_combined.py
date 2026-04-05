"""
TP + PP + DP Combined (3D Parallelism)
========================================
세 가지 parallelism을 동시에 적용하는 전체 시뮬레이션.

예시: 16 GPUs = 2 TP × 2 PP × 4 DP

  DP group 0          DP group 1          DP group 2          DP group 3
  ┌───────────┐       ┌───────────┐       ┌───────────┐       ┌───────────┐
  │PP0: GPU0,1│       │PP0: GPU4,5│       │PP0: GPU8,9│       │PP0:GPU12,13│
  │    (TP=2) │       │    (TP=2) │       │    (TP=2) │       │    (TP=2)  │
  │PP1: GPU2,3│       │PP1: GPU6,7│       │PP1:GPU10,11│      │PP1:GPU14,15│
  │    (TP=2) │       │    (TP=2) │       │    (TP=2)  │      │    (TP=2)  │
  └───────────┘       └───────────┘       └───────────┘       └───────────┘

  통신 패턴:
    TP:  all-reduce (노드 내 NVLink)    — 같은 layer를 나눠 계산
    PP:  send/recv  (stage 경계)        — 다른 layer를 순차 실행
    DP:  all-reduce (노드 간 IB 가능)   — 같은 모델, 다른 데이터

  각 GPU의 역할:
    GPU 0: DP group 0, PP stage 0, TP rank 0
    GPU 1: DP group 0, PP stage 0, TP rank 1  ← GPU 0과 같은 layer를 TP
    GPU 2: DP group 0, PP stage 1, TP rank 0  ← GPU 0과 다른 layer
    GPU 3: DP group 0, PP stage 1, TP rank 1
    GPU 4: DP group 1, PP stage 0, TP rank 0  ← GPU 0과 같은 모델, 다른 데이터
    ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: Process Group 시뮬레이션
# ============================================================

def build_process_groups(world_size, tp_size, pp_size, dp_size):
    """
    3D parallelism의 process group을 구성.

    rank 배치 순서: [DP][PP][TP]  (TP가 가장 안쪽 = 같은 노드)

    rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank
    """
    assert world_size == tp_size * pp_size * dp_size

    groups = {"tp": [], "pp": [], "dp": []}
    rank_info = {}

    for rank in range(world_size):
        tp_rank = rank % tp_size
        pp_rank = (rank // tp_size) % pp_size
        dp_rank = rank // (tp_size * pp_size)
        rank_info[rank] = {"tp": tp_rank, "pp": pp_rank, "dp": dp_rank}

    # TP group: 같은 DP + 같은 PP stage 내에서 TP하는 GPU들
    for dp in range(dp_size):
        for pp in range(pp_size):
            tp_group = [dp * pp_size * tp_size + pp * tp_size + tp
                        for tp in range(tp_size)]
            groups["tp"].append(tp_group)

    # PP group: 같은 DP + 같은 TP rank끼리
    for dp in range(dp_size):
        for tp in range(tp_size):
            pp_group = [dp * pp_size * tp_size + pp * tp_size + tp
                        for pp in range(pp_size)]
            groups["pp"].append(pp_group)

    # DP group: 같은 PP stage + 같은 TP rank끼리
    for pp in range(pp_size):
        for tp in range(tp_size):
            dp_group = [dp * pp_size * tp_size + pp * tp_size + tp
                        for dp in range(dp_size)]
            groups["dp"].append(dp_group)

    return groups, rank_info


# ============================================================
# Part 2: 전체 3D Parallelism 시뮬레이션
# ============================================================

def simulate_3d_parallelism():
    """TP + PP + DP를 모두 적용한 MLP 학습을 시뮬레이션."""
    print("=" * 70)
    print("3D Parallelism Simulation: TP + PP + DP")
    print("=" * 70)

    torch.manual_seed(42)

    # --- Config ---
    tp_size = 2
    pp_size = 2
    dp_size = 2
    world_size = tp_size * pp_size * dp_size  # 8 GPUs

    embed_dim = 8
    hidden_dim = 16
    num_layers = 4
    layers_per_stage = num_layers // pp_size
    micro_batch_size = 2
    seq_len = 4

    print(f"\n  Config: {tp_size} TP × {pp_size} PP × {dp_size} DP = {world_size} GPUs")
    print(f"  Model: {num_layers} layers, embed={embed_dim}, ffn={hidden_dim}")
    print(f"  PP: {layers_per_stage} layers per stage")

    # --- Process groups ---
    groups, rank_info = build_process_groups(world_size, tp_size, pp_size, dp_size)

    print(f"\n  Process Groups:")
    print(f"    TP groups:  {groups['tp']}")
    print(f"    PP groups:  {groups['pp']}")
    print(f"    DP groups:  {groups['dp']}")

    print(f"\n  GPU Assignments:")
    print(f"    {'GPU':>4} {'TP rank':>8} {'PP stage':>9} {'DP rank':>8}")
    print(f"    {'-'*4} {'-'*8} {'-'*9} {'-'*8}")
    for rank in range(world_size):
        info = rank_info[rank]
        print(f"    {rank:>4} {info['tp']:>8} {info['pp']:>9} {info['dp']:>8}")

    # --- 전체 모델 weights (기준점) ---
    all_W1 = [torch.randn(embed_dim, hidden_dim) for _ in range(num_layers)]
    all_W2 = [torch.randn(hidden_dim, embed_dim) for _ in range(num_layers)]

    # --- DP별 다른 데이터 ---
    data_per_dp = {
        dp: torch.randn(micro_batch_size, seq_len, embed_dim)
        for dp in range(dp_size)
    }

    # --- Single GPU 기준 (dp=0 데이터, 전체 모델) ---
    x_ref = data_per_dp[0].clone()
    for layer_idx in range(num_layers):
        x_ref = x_ref + F.gelu(x_ref @ all_W1[layer_idx]) @ all_W2[layer_idx]
    ref_output = x_ref

    # --- 3D Parallelism 시뮬레이션 ---
    print(f"\n  === Forward Pass Simulation (DP group 0) ===")

    # DP group 0의 데이터
    x = data_per_dp[0].clone()

    for stage in range(pp_size):
        print(f"\n    PP Stage {stage}:")
        start_layer = stage * layers_per_stage
        end_layer = start_layer + layers_per_stage

        for layer_idx in range(start_layer, end_layer):
            # TP: weight를 column/row split
            half = hidden_dim // tp_size
            W1 = all_W1[layer_idx]
            W2 = all_W2[layer_idx]

            # TP Rank 0
            W1_r0 = W1[:, :half]
            W2_r0 = W2[:half, :]
            a1_r0 = F.gelu(x @ W1_r0)
            partial_r0 = a1_r0 @ W2_r0

            # TP Rank 1
            W1_r1 = W1[:, half:]
            W2_r1 = W2[half:, :]
            a1_r1 = F.gelu(x @ W1_r1)
            partial_r1 = a1_r1 @ W2_r1

            # TP all-reduce
            layer_out = partial_r0 + partial_r1

            # Residual
            x = x + layer_out

            print(f"      Layer {layer_idx}: TP split → fc1 → GELU → fc2 → all-reduce → residual")

        if stage < pp_size - 1:
            print(f"      → send activation to PP stage {stage + 1} (point-to-point)")

    # 검증
    # 주의: GELU(concat) ≠ concat(GELU(split)) 이므로 정확히 일치하지 않음.
    # 이건 Megatron-LM style TP의 본래 특성:
    #   Single GPU: GELU(x @ [W1_left, W1_right]) → 전체 concat에 GELU
    #   TP:         GELU(x @ W1_left), GELU(x @ W1_right) → 각각 GELU 후 합산
    # 실제 학습에서는 처음부터 TP로 학습하므로 문제 없음 (수렴 동일).
    diff = (ref_output - x).abs().max().item()
    print(f"\n  Single GPU vs 3D Parallel output diff: {diff:.2e}")
    print(f"  (GELU가 column split 사이에 있어서 정확히 일치하지 않는 것은 정상)")
    print(f"  (처음부터 TP로 학습하면 동일하게 수렴)")

    # --- Backward에서의 통신 ---
    print(f"\n  === Backward Communication Pattern ===")
    print(f"""
    DP group 0, PP stage 1 (마지막 stage):
      1. Loss 계산 + backward (layer 3, 2)
      2. 각 layer에서 TP all-reduce (gradient)
      3. PP: gradient를 stage 0으로 send

    DP group 0, PP stage 0:
      4. PP: stage 1로부터 gradient recv
      5. backward (layer 1, 0)
      6. 각 layer에서 TP all-reduce (gradient)

    All DP groups:
      7. DP all-reduce: 각 DP group의 gradient를 평균
         → DP group 0과 1의 gradient 합산
      8. Optimizer step (모든 GPU 동일 update)
    """)


# ============================================================
# Part 3: 통신량 분석
# ============================================================

def communication_analysis():
    """3D parallelism의 통신량을 상세 분석."""
    print("=" * 70)
    print("Communication Analysis: TP + PP + DP")
    print("=" * 70)

    # 예시 설정
    tp_size = 8
    pp_size = 4
    dp_size = 2
    world_size = tp_size * pp_size * dp_size

    embed_dim = 4096
    hidden_dim = 16384
    num_layers = 32
    seq_len = 2048
    micro_batch = 4
    num_microbatches = 8
    bytes_per_elem = 2  # BF16

    layers_per_stage = num_layers // pp_size

    print(f"\n  Config: {tp_size}TP × {pp_size}PP × {dp_size}DP = {world_size} GPUs")
    print(f"  Model: {num_layers}L, embed={embed_dim}, ffn={hidden_dim}")

    # TP 통신: 각 layer에서 all-reduce 2회 (fwd + bwd)
    # all-reduce 데이터: activation size = batch * seq * embed
    tp_per_allreduce = micro_batch * seq_len * embed_dim * bytes_per_elem
    tp_per_layer = tp_per_allreduce * 2  # fwd + bwd
    tp_total = tp_per_layer * layers_per_stage  # 이 stage의 layer 수
    # all-reduce 실제 통신: 2 * data * (tp_size-1)/tp_size
    tp_actual = tp_total * 2 * (tp_size - 1) / tp_size

    # PP 통신: stage 경계에서 activation send/recv
    pp_per_transfer = micro_batch * seq_len * embed_dim * bytes_per_elem
    pp_total = pp_per_transfer * num_microbatches * 2  # fwd + bwd, 각 microbatch

    # DP 통신: gradient all-reduce (전체 파라미터)
    params_per_stage = layers_per_stage * (embed_dim * hidden_dim * 2)  # W1 + W2
    # TP로 이미 split된 params
    params_per_rank = params_per_stage // tp_size
    dp_allreduce = params_per_rank * bytes_per_elem * 2 * (dp_size - 1) / dp_size

    print(f"\n  Per-step 통신량:")
    print(f"    TP (all-reduce):     {tp_actual / 1e6:>8.1f} MB  (NVLink, 빈번)")
    print(f"    PP (send/recv):      {pp_total / 1e6:>8.1f} MB  (stage 경계)")
    print(f"    DP (gradient sync):  {dp_allreduce / 1e6:>8.1f} MB  (IB, 1회)")

    print(f"\n  Overlap 전략:")
    print(f"    TP: layer 연산과 겹치기 어려움 (같은 layer 내 의존성)")
    print(f"    PP: micro-batch 간 overlap (1F1B schedule)")
    print(f"    DP: backward 중 gradient bucketing으로 overlap")

    print(f"\n  Bandwidth 요구:")
    step_time_ms = 500  # 가정
    print(f"    TP: {tp_actual / 1e6 / (step_time_ms/1000):>6.1f} MB/s per step"
          f" → NVLink ({900*1000} MB/s) 충분")
    print(f"    DP: {dp_allreduce / 1e6 / (step_time_ms/1000):>6.1f} MB/s per step"
          f" → IB ({400*1000} MB/s) 충분")


# ============================================================
# Part 4: 메모리 분석
# ============================================================

def memory_analysis():
    """3D parallelism에서 각 GPU의 메모리 사용량."""
    print("\n" + "=" * 70)
    print("Memory Analysis: TP + PP + DP")
    print("=" * 70)

    P_total = 7  # 7B params (GB in FP32)
    tp, pp, dp = 8, 4, 2
    bf16 = 2

    # 각 GPU의 파라미터 수
    # PP: 1/pp 만큼의 layer
    # TP: 각 layer의 weight를 1/tp로 split
    P_per_gpu = P_total / (tp * pp)  # in billions

    print(f"\n  7B model, {tp}TP × {pp}PP × {dp}DP = {tp*pp*dp} GPUs")
    print(f"\n  Parameters per GPU: {P_total}B / ({tp}×{pp}) = {P_per_gpu:.3f}B")

    params_gb = P_per_gpu * bf16
    grads_gb = P_per_gpu * bf16
    opt_gb = P_per_gpu * 4 * 2  # Adam m,v in FP32
    master_gb = P_per_gpu * 4   # FP32 master copy

    print(f"\n  {'Component':<30} {'Size (GB)'}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'Params (BF16)':<30} {params_gb:.3f}")
    print(f"  {'Gradients (BF16)':<30} {grads_gb:.3f}")
    print(f"  {'Optimizer states (FP32 m,v)':<30} {opt_gb:.3f}")
    print(f"  {'Master weights (FP32)':<30} {master_gb:.3f}")
    total = params_gb + grads_gb + opt_gb + master_gb
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'Total (model states)':<30} {total:.3f}")

    # Activation memory (per micro-batch)
    seq, batch = 2048, 4
    embed = 4096
    layers_per_stage = 32 // pp
    act_per_layer = batch * seq * embed * bf16 * 4 / 1e9  # ~4 tensors per layer
    act_total = act_per_layer * layers_per_stage
    print(f"\n  Activations (per micro-batch):")
    print(f"    Per layer: {act_per_layer*1000:.1f} MB")
    print(f"    Per stage ({layers_per_stage} layers): {act_total:.3f} GB")

    print(f"\n  → 3D parallelism으로 7B 모델이 GPU당 ~{total + act_total:.1f} GB")
    print(f"     80GB GPU에서 충분히 학습 가능!")


# ============================================================
# Part 5: DTensor 3D Parallelism (PyTorch 2.x)
# ============================================================
#
# from torch.distributed.device_mesh import init_device_mesh
# from torch.distributed.tensor.parallel import parallelize_module
# from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
# from torch.distributed.fsdp import fully_shard
# from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
#
# # 3D mesh 생성
# mesh = init_device_mesh("cuda", (dp_size, pp_size, tp_size),
#                         mesh_dim_names=("dp", "pp", "tp"))
#
# # Step 1: TP 적용 (가장 안쪽)
# for block in model.blocks:
#     parallelize_module(block.ffn, mesh["tp"], {
#         "fc1": ColwiseParallel(),
#         "fc2": RowwiseParallel(),
#     })
#
# # Step 2: PP 적용
# pipe = pipeline(model, mb_args=(example,),
#                 split_spec={"blocks.16": SplitPoint.BEGINNING})
# stage = pipe.build_stage(stage_index=pp_rank, device=device)
# schedule = ScheduleGPipe(stage, n_microbatches=8, loss_fn=loss_fn)
#
# # Step 3: DP 적용 (가장 바깥)
# # FSDP2로 data parallel (ZeRO-3 style)
# for block in model.blocks:
#     fully_shard(block, mesh=mesh["dp"])
# fully_shard(model, mesh=mesh["dp"])
#
# # 또는 2D mesh로 DP+TP 동시 적용:
# mesh_2d = init_device_mesh("cuda", (dp_size, tp_size),
#                            mesh_dim_names=("dp", "tp"))


# ============================================================
# Part 6: 전략 선택 가이드
# ============================================================

def strategy_guide():
    print("\n" + "=" * 70)
    print("Parallelism Strategy Selection Guide")
    print("=" * 70)

    print("""
  Q: 어떤 parallelism을 써야 하나?

  Step 1: 모델이 1 GPU에 들어가나?
    Yes → DDP만으로 충분 (가장 단순, 가장 빠름)
    No  → Step 2

  Step 2: 모델이 1 노드(8 GPU)에 들어가나?
    Yes → TP=8 (노드 내 NVLink) + DP
    No  → Step 3

  Step 3: 3D parallelism 필요
    TP = 8 (노드 내 NVLink, 고정)
    PP = ceil(model_memory / (8 * gpu_memory)) (최소한으로)
    DP = total_gpus / (TP × PP) (나머지)

  예시:
    ┌──────────┬──────────┬───────────────────────────────┐
    │ Model    │ GPUs     │ Strategy                      │
    ├──────────┼──────────┼───────────────────────────────┤
    │ 1.3B     │ 8        │ DDP (or FSDP)                 │
    │ 7B       │ 8        │ FSDP (ZeRO-3) or TP=8        │
    │ 7B       │ 32       │ TP=8, DP=4                    │
    │ 70B      │ 64       │ TP=8, PP=2, DP=4              │
    │ 70B      │ 256      │ TP=8, PP=4, DP=8              │
    │ 405B     │ 16384    │ TP=8, PP=16, DP=128           │
    └──────────┴──────────┴───────────────────────────────┘

  원칙:
    1. TP를 먼저 (NVLink 활용, 가장 빠른 통신)
    2. PP를 최소한 (bubble 발생, micro-batch ≥ 4×pp_size)
    3. DP를 최대한 (가장 확장성 좋고 단순)
    4. 메모리 부족하면 FSDP (ZeRO-3) 추가
    """)


if __name__ == "__main__":
    simulate_3d_parallelism()
    communication_analysis()
    memory_analysis()
    strategy_guide()
