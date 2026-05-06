"""
FSDP (Fully Sharded Data Parallel)
=====================================
PyTorch 네이티브 ZeRO-3 구현. 모든 모델 상태를 GPU에 분산.

FSDP vs DDP:
  DDP:  각 GPU가 전체 모델 복사본 보유 + gradient all-reduce
  FSDP: 모델을 shard로 쪼개서 분산 + 필요할 때만 all-gather

FSDP 동작 (각 FSDP unit = 보통 1개 Transformer block):
  ┌──────────────────────────────────────────────────────┐
  │ Forward:                                             │
  │   all-gather params → forward 계산 → params 해제     │
  │                                                      │
  │ Backward:                                            │
  │   all-gather params → backward 계산 →                │
  │   reduce-scatter grads → params 해제                 │
  │                                                      │
  │ Optimizer step:                                      │
  │   각 GPU가 자기 shard만 update (local operation)      │
  └──────────────────────────────────────────────────────┘

FSDP1 vs FSDP2:
  FSDP1 (torch.distributed.fsdp.FullyShardedDataParallel):
    - FlatParameter: 여러 params를 하나로 flatten → 통신 효율적
    - 단점: flatten 때문에 디버깅 어렵고 유연성 부족

  FSDP2 (torch.distributed.fsdp.fully_shard, PyTorch 2.x):
    - DTensor 기반: 각 param이 독립적인 DTensor
    - per-parameter sharding → 더 유연하고 디버깅 쉬움
    - DeviceMesh와 자연스럽게 통합
"""

import torch
import torch.nn as nn


# ============================================================
# Part 1: FSDP1 사용법 (기존 API)
# ============================================================
#
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
#
# --- ShardingStrategy ---
# FULL_SHARD:     params + grads + opt states 모두 분산 (= ZeRO-3)
# SHARD_GRAD_OP:  grads + opt states만 분산 (= ZeRO-2)
# NO_SHARD:       분산 안 함 (= DDP)
# HYBRID_SHARD:   노드 내 FULL_SHARD + 노드 간 replicate
#
# --- 기본 사용법 ---
#
# model = MyModel()
#
# # 각 Transformer block을 별도 FSDP unit으로 wrap
# for i, block in enumerate(model.blocks):
#     model.blocks[i] = FSDP(block)
#
# # 전체 모델도 FSDP wrap
# model = FSDP(
#     model,
#     sharding_strategy=ShardingStrategy.FULL_SHARD,
#     mixed_precision=MixedPrecision(
#         param_dtype=torch.bfloat16,    # forward에서 params를 BF16으로
#         reduce_dtype=torch.float32,     # gradient reduce는 FP32로
#     ),
# )
#
# # 학습 루프는 일반 PyTorch와 동일
# output = model(input_ids)
# loss = loss_fn(output, targets)
# loss.backward()
# optimizer.step()


# ============================================================
# Part 2: FSDP2 사용법 (fully_shard, PyTorch 2.x)
# ============================================================
#
# from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
# from torch.distributed.device_mesh import init_device_mesh
#
# mesh = init_device_mesh("cuda", (world_size,))
#
# # FSDP2: fully_shard()로 선언적 적용
# # 내부적으로 각 param을 DTensor(Shard(0))로 변환
# mp_policy = MixedPrecisionPolicy(
#     param_dtype=torch.bfloat16,
#     reduce_dtype=torch.float32,
# )
#
# for block in model.blocks:
#     fully_shard(block, mesh=mesh, mp_policy=mp_policy)
# fully_shard(model, mesh=mesh, mp_policy=mp_policy)
#
# # 학습 루프 동일
# output = model(input_ids)
# loss.backward()
# optimizer.step()
#
# --- FSDP1 vs FSDP2 차이 ---
# FSDP1: model을 FSDP()로 wrap → FlatParameter로 변환
# FSDP2: fully_shard()로 적용 → 각 param이 DTensor, 원래 구조 유지


# ============================================================
# Part 3: FSDP 동작 시뮬레이션
# ============================================================

def simulate_fsdp():
    """GPU 없이 FSDP의 shard/gather 동작을 시뮬레이션."""
    print("=" * 60)
    print("FSDP Simulation (4 GPUs)")
    print("=" * 60)

    num_gpus = 4
    num_layers = 2

    # 각 layer의 params (간소화)
    layer_params = [torch.randn(8) for _ in range(num_layers)]  # 8 params per layer

    print(f"\n  전체 모델: {num_layers} layers × 8 params = {num_layers * 8} params")
    print(f"  GPU 수: {num_gpus}")

    # --- Shard: 각 GPU가 전체 params의 1/N만 보관 ---
    shards = {gpu: {} for gpu in range(num_gpus)}
    params_per_gpu = 8 // num_gpus  # 2 params per GPU per layer

    for layer_idx, params in enumerate(layer_params):
        for gpu_id in range(num_gpus):
            start = gpu_id * params_per_gpu
            end = start + params_per_gpu
            shards[gpu_id][layer_idx] = params[start:end].clone()

    print(f"\n  초기 상태 (각 GPU가 보관하는 shard):")
    for gpu_id in range(num_gpus):
        total = sum(s.numel() for s in shards[gpu_id].values())
        print(f"    GPU {gpu_id}: {total} params (전체의 1/{num_gpus})")

    # --- Forward 시뮬레이션 ---
    print(f"\n  Forward 동작:")
    for layer_idx in range(num_layers):
        # all-gather: 모든 GPU의 shard를 모아서 전체 params 복원
        gathered = torch.cat([shards[gpu][layer_idx] for gpu in range(num_gpus)])
        print(f"    Layer {layer_idx}: all-gather ({params_per_gpu}×{num_gpus}={gathered.numel()} params)"
              f" → forward 계산 → 전체 params 해제")

    # --- Backward 시뮬레이션 ---
    print(f"\n  Backward 동작:")
    for layer_idx in reversed(range(num_layers)):
        print(f"    Layer {layer_idx}: all-gather params → backward 계산"
              f" → reduce-scatter grads → params 해제")

    # --- 통신량 ---
    total_params = sum(p.numel() for p in layer_params)
    print(f"\n  통신량:")
    print(f"    Forward:  {num_layers} × all-gather({total_params}) = {num_layers * total_params} elements")
    print(f"    Backward: {num_layers} × (all-gather + reduce-scatter) = {num_layers * total_params * 2} elements")
    print(f"    Total:    3 × model_size (vs DDP 2 × model_size)")


# ============================================================
# Part 4: FSDP + TP 조합 (2D Parallelism)
# ============================================================
#
# 대규모 모델에서는 FSDP와 TP를 함께 사용:
#
#   mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
#
#   # TP 먼저 적용 (intra-layer)
#   parallelize_module(block.ffn, mesh["tp"], {
#       "fc1": ColwiseParallel(),
#       "fc2": RowwiseParallel(),
#   })
#
#   # FSDP 적용 (inter-layer, DP 차원)
#   fully_shard(block, mesh=mesh["dp"])
#
# 예: 32 GPUs = 4 DP × 8 TP
#   - 8 GPUs가 하나의 TP group (layer 내부를 나눔)
#   - 4 TP groups가 FSDP로 data parallel (layer를 나눔)
#
#                     TP group (8 GPUs)
#                  ┌──────────────────┐
#   FSDP group 0: │ GPU0 ... GPU7    │  ← 같은 layer의 weight를 column/row split
#   FSDP group 1: │ GPU8 ... GPU15   │
#   FSDP group 2: │ GPU16 ... GPU23  │
#   FSDP group 3: │ GPU24 ... GPU31  │
#                  └──────────────────┘
#                  각 FSDP group은 동일 데이터의 다른 micro-batch 처리


# ============================================================
# Part 5: 메모리 비교
# ============================================================

def memory_comparison():
    print("\n" + "=" * 60)
    print("Memory per GPU (7B model, 4 GPUs, Adam, BF16)")
    print("=" * 60)

    P = 7  # 7B params
    N = 4  # GPUs
    bf16 = 2  # bytes
    fp32 = 4

    print(f"\n  {'Component':<25} {'DDP':<15} {'FSDP':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")

    ddp_params = P * bf16
    ddp_grads = P * bf16
    ddp_opt = P * fp32 * 2  # m + v in FP32
    ddp_master = P * fp32   # master weights
    ddp_total = ddp_params + ddp_grads + ddp_opt + ddp_master

    fsdp_params = P * bf16 / N
    fsdp_grads = P * bf16 / N
    fsdp_opt = P * fp32 * 2 / N
    fsdp_master = P * fp32 / N
    fsdp_total = fsdp_params + fsdp_grads + fsdp_opt + fsdp_master
    # forward 시 all-gather하면 일시적으로 전체 params 필요
    fsdp_peak = fsdp_total + P * bf16  # shard + 1 layer의 전체 params

    print(f"  {'Parameters (BF16)':<25} {ddp_params:.1f} GB{'':<7} {fsdp_params:.1f} GB")
    print(f"  {'Gradients (BF16)':<25} {ddp_grads:.1f} GB{'':<7} {fsdp_grads:.1f} GB")
    print(f"  {'Optimizer (FP32 m,v)':<25} {ddp_opt:.1f} GB{'':<7} {fsdp_opt:.1f} GB")
    print(f"  {'Master weights (FP32)':<25} {ddp_master:.1f} GB{'':<7} {fsdp_master:.1f} GB")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Total (steady)':<25} {ddp_total:.1f} GB{'':<7} {fsdp_total:.1f} GB")
    print(f"  {'Peak (fwd all-gather)':<25} {'N/A':<15} ~{fsdp_peak:.1f} GB")


if __name__ == "__main__":
    simulate_fsdp()
    memory_comparison()
