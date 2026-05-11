"""
ZeRO (Zero Redundancy Optimizer) Stage 1 / 2 / 3
====================================================
핵심: 모든 GPU가 전체 모델 상태를 중복 저장하는 것을 제거.

DDP의 문제:
  각 GPU가 model weights + optimizer states + gradients를 전부 보유
  → N개 GPU여도 메모리 사용량은 1개 GPU와 동일

ZeRO 해결책: 모델 상태를 GPU들에 분산(partition)

  ┌──────────────────────────────────────────────────────┐
  │               각 GPU당 메모리 (1B params, Adam)       │
  │                                                      │
  │  Component            DDP    ZeRO-1  ZeRO-2  ZeRO-3 │
  │  ─────────────────── ────── ─────── ─────── ─────── │
  │  Optimizer states     8 GB   8/N GB  8/N GB  8/N GB │
  │  Gradients            4 GB   4 GB    4/N GB  4/N GB │
  │  Parameters           4 GB   4 GB    4 GB    4/N GB │
  │  ─────────────────── ────── ─────── ─────── ─────── │
  │  Total (N=4)         16 GB   6 GB    5 GB    4 GB   │
  └──────────────────────────────────────────────────────┘

  ZeRO-1: Optimizer states만 분산
  ZeRO-2: + Gradients도 분산
  ZeRO-3: + Parameters도 분산 (= FSDP와 동일 개념)

TP/PP와 같이 쓸 때 주의:
  TP가 이미 parameter를 쪼개고 있다고 해서 자동으로 ZeRO-3인 것은 아님.
  TP shard는 "model-parallel 계산을 위해 weight를 나눈 것"이고,
  ZeRO shard는 "data-parallel replica 사이의 중복 state를 없애려고 나눈 것"이다.

  예: TP=2, DP=4이면
    TP0 owns W_left, TP1 owns W_right.
    DP0-TP0, DP1-TP0, DP2-TP0, DP3-TP0가 모두 W_left를 들고 있으면
    parameter는 DP 축에서 여전히 중복이므로 ZeRO-3가 아니다.

  Distributed Optimizer는 보통 "각 TP shard 안에서" DP group 방향으로
  optimizer states / main gradients를 shard한다.
    full model param → TP shard → DP optimizer shard

왜 Megatron은 ZeRO-3보다 distributed optimizer(ZeRO-1/2 계열)를 자주 쓰나?
  1. TP/PP가 이미 parameter 메모리를 total_params/(TP×PP)로 크게 줄인다.
  2. BF16 parameter보다 Adam states, FP32 master weights, grad buffers가 더 큰 병목이다.
     mixed precision Adam에서 parameter 1개당 대략:
       BF16 param:         2 bytes
       BF16 grad:          2 bytes
       FP32 master weight: 4 bytes  # update 안정성을 위해 optimizer가 쓰는 FP32 복사본
       Adam m:             4 bytes  # first moment, gradient의 지수이동평균(momentum)
       Adam v:             4 bytes  # second moment, gradient^2의 지수이동평균(variance/RMS)
     즉 실제 메모리 병목은 param 자체보다 master weight + m + v + grad 쪽에서 커진다.
  3. ZeRO-3처럼 parameter까지 DP-shard하면 매 layer마다 TP shard 내부 parameter all-gather가 추가된다.
  4. Megatron은 TP/PP/CP 통신이 이미 많아서 parameter all-gather까지 넣으면 overlap/schedule이 복잡해진다.
  5. 대규모 dense transformer에서는 throughput 예측성과 통신 overlap이 중요하므로,
     parameter는 TP/PP shard로 두고 optimizer state/main grad 중복을 DP 축에서 줄이는 선택이 실용적이다.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


# ============================================================
# Part 0: 실제 torch.distributed ZeRO helper
# ============================================================

def _dist_ready(group=None):
    """torchrun 등으로 process group이 초기화되어 있으면 실제 dist API를 사용할 수 있다."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def _partition_range(numel, group=None):
    """현재 rank가 담당하는 flat parameter shard 범위 [start:end)를 계산."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    assert numel % world_size == 0, "이 교육용 예시는 균등 shard만 다룹니다."
    shard_size = numel // world_size
    start = rank * shard_size
    end = start + shard_size
    return start, end, shard_size


def _adam_update_(param_shard, grad_shard, m_shard, v_shard,
                  lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """local shard에 Adam update를 in-place로 적용."""
    m_shard.mul_(beta1).add_(grad_shard, alpha=1 - beta1)
    v_shard.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1 - beta2)
    param_shard.addcdiv_(m_shard, v_shard.sqrt().add(eps), value=-lr)
    return param_shard


def _all_gather_flat_shards(shard, group=None):
    """
    모든 rank의 shard를 rank 순서대로 모아 full flat tensor를 만든다.

    dist.all_gather(output_list, input_tensor):
      output_list = 모든 rank의 tensor를 받을 빈 리스트
      input_tensor = 현재 rank가 보낼 shard
    """
    world_size = dist.get_world_size(group)
    parts = [torch.empty_like(shard) for _ in range(world_size)]
    dist.all_gather(parts, shard.contiguous(), group=group)
    return torch.cat(parts, dim=0)


def zero1_adam_step_dist(params_full, grads_full, m_shard, v_shard, group=None):
    """
    ZeRO-1 실제 dist 버전: optimizer states만 shard.

    각 rank가 보유:
      params_full: 전체 parameter flat tensor
      grads_full:  전체 gradient flat tensor
      m_shard/v_shard: Adam state 중 자기 담당 parameter 구간만

    통신:
      1. dist.all_reduce(grads_full): DDP처럼 모든 rank의 gradient를 합쳐 full grad 동기화
      2. 각 rank가 자기 param shard만 Adam update
      3. dist.all_gather(param_shard): 업데이트된 shard들을 모아 모든 rank의 full params 갱신
    """
    if not _dist_ready(group):
        raise RuntimeError("dist.init_process_group() 이후에 호출해야 합니다.")

    world_size = dist.get_world_size(group)
    start, end, _ = _partition_range(params_full.numel(), group)

    # ZeRO-1은 gradient는 DDP와 동일하게 full all-reduce한다.
    # all_reduce(SUM) 후 grads_full은 모든 rank gradient의 합:
    #   grads_full = g_rank0 + g_rank1 + ... + g_rankN
    dist.all_reduce(grads_full, op=dist.ReduceOp.SUM, group=group)
    # 보통 optimizer step에는 평균 gradient를 쓰므로 rank 수로 나눈다.
    # div_의 trailing underscore는 in-place 연산이라는 PyTorch convention.
    #   grads_full = grads_full / world_size 를 새 tensor 없이 직접 수행한다.
    grads_full.div_(world_size)

    param_shard = params_full[start:end].contiguous()
    grad_shard = grads_full[start:end].contiguous()
    _adam_update_(param_shard, grad_shard, m_shard, v_shard)

    # 각 rank가 업데이트한 param shard를 다시 모아 full params를 복원한다.
    params_full.copy_(_all_gather_flat_shards(param_shard, group))
    return params_full


def zero2_adam_step_dist(params_full, grads_full, m_shard, v_shard, group=None):
    """
    ZeRO-2 실제 dist 버전: optimizer states + gradients shard.

    ZeRO-1과 핵심 차이:
      full gradient all-reduce 대신 reduce-scatter를 사용한다.

    통신:
      1. dist.reduce_scatter(grad_shard, grad_chunks):
         모든 rank의 grads_full을 합치되, 현재 rank 담당 chunk만 받는다.
      2. local Adam update
      3. dist.all_gather(param_shard): 업데이트된 params를 모든 rank에 복제

    Megatron distributed optimizer와의 관계:
      실제 Megatron에서 params_full은 "전체 모델 parameter"가 아니라
      현재 TP/PP rank가 원래 담당하는 local parameter shard를 flatten한 buffer라고 보면 된다.
      Distributed optimizer는 그 local TP/PP shard buffer를 DP group 안에서 다시 나눠
      optimizer state와 main grad 중복을 줄인다.
      즉 TP shard를 full model로 합쳤다가 다시 자르는 것이 아니다.
    """
    if not _dist_ready(group):
        raise RuntimeError("dist.init_process_group() 이후에 호출해야 합니다.")

    world_size = dist.get_world_size(group)
    start, end, shard_size = _partition_range(params_full.numel(), group)

    # input_list는 이 rank의 full grad를 rank별 shard로 쪼갠 리스트.
    # reduce_scatter 후 grad_shard에는 "all-reduce 결과 중 내 shard"만 남는다.
    grad_chunks = [chunk.contiguous() for chunk in grads_full.chunk(world_size, dim=0)]
    grad_shard = torch.empty(shard_size, dtype=grads_full.dtype, device=grads_full.device)
    dist.reduce_scatter(grad_shard, grad_chunks, op=dist.ReduceOp.SUM, group=group)
    grad_shard.div_(world_size)

    param_shard = params_full[start:end].contiguous()
    _adam_update_(param_shard, grad_shard, m_shard, v_shard)

    params_full.copy_(_all_gather_flat_shards(param_shard, group))
    return params_full, grad_shard


def zero3_all_gather_params_dist(param_shard, group=None):
    """
    ZeRO-3/FSDP forward 직전: parameter shard들을 all-gather해서 full param을 임시 복원.

    각 rank는 평소 params shard만 들고 있다가, layer 계산이 필요할 때만 full param을 만든다.
    계산이 끝나면 full param temporary는 버릴 수 있다.
    """
    if not _dist_ready(group):
        raise RuntimeError("dist.init_process_group() 이후에 호출해야 합니다.")
    return _all_gather_flat_shards(param_shard, group)


def zero3_reduce_scatter_grads_dist(grad_full, group=None):
    """
    ZeRO-3/FSDP backward 후: full gradient를 reduce-scatter해서 grad shard만 남긴다.

    all-reduce처럼 모든 rank의 gradient를 합치지만,
    결과 전체를 복제하지 않고 현재 rank가 담당하는 shard만 받는다.
    """
    if not _dist_ready(group):
        raise RuntimeError("dist.init_process_group() 이후에 호출해야 합니다.")

    world_size = dist.get_world_size(group)
    _, _, shard_size = _partition_range(grad_full.numel(), group)
    grad_chunks = [chunk.contiguous() for chunk in grad_full.chunk(world_size, dim=0)]
    grad_shard = torch.empty(shard_size, dtype=grad_full.dtype, device=grad_full.device)
    dist.reduce_scatter(grad_shard, grad_chunks, op=dist.ReduceOp.SUM, group=group)
    grad_shard.div_(world_size)
    return grad_shard


def zero3_adam_step_dist(param_shard, grad_shard, m_shard, v_shard):
    """
    ZeRO-3 optimizer step: params/grads/optimizer states가 모두 local shard.

    이 단계 자체에는 통신이 없다. 통신은 계산 전 all-gather(params),
    backward 후 reduce-scatter(grads)에서 이미 끝났다.
    """
    return _adam_update_(param_shard, grad_shard, m_shard, v_shard)


# ============================================================
# Part 1: ZeRO Stage 1 시뮬레이션 (Optimizer State Partitioning)
# ============================================================

def simulate_zero_stage1():
    """
    ZeRO-1: Optimizer states를 GPU에 분산.

    동작:
    1. Forward/Backward: 일반 DDP와 동일 (all-reduce gradients)
    2. Optimizer step: 각 GPU가 자기 담당 파라미터만 update
    3. All-gather: 업데이트된 파라미터를 모든 GPU에 broadcast

    통신량: DDP와 동일 (all-reduce gradients)
    메모리 절약: optimizer states만 1/N
    """
    print("=" * 60)
    print("ZeRO Stage 1: Optimizer State Partitioning")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    num_params = 8  # 예시: 8개 파라미터

    # 전체 파라미터와 gradient
    params = torch.randn(num_params)
    grads = torch.randn(num_params)

    # Adam optimizer states: m (momentum), v (variance)
    m = torch.zeros(num_params)
    v = torch.zeros(num_params)

    # --- DDP: 모든 GPU가 전체 optimizer state 보유 ---
    ddp_mem_per_gpu = num_params * 3  # params + m + v
    print(f"\n  DDP: 각 GPU optimizer 메모리 = {ddp_mem_per_gpu} values")

    # --- ZeRO-1: optimizer state를 GPU별로 분할 ---
    params_per_gpu = num_params // num_gpus
    zero1_mem_per_gpu = num_params + params_per_gpu * 2  # full params + partitioned (m, v)
    print(f"  ZeRO-1: 각 GPU optimizer 메모리 = {zero1_mem_per_gpu} values")
    print(f"  절약: {(1 - zero1_mem_per_gpu / ddp_mem_per_gpu) * 100:.0f}%")

    # 시뮬레이션: GPU 0은 params[0:2]만 담당
    print(f"\n  GPU 0 담당: params[0:{params_per_gpu}]")
    print(f"  GPU 1 담당: params[{params_per_gpu}:{2*params_per_gpu}]")
    print(f"  ...")

    # Step 1: All-reduce gradients (DDP와 동일)
    print(f"\n  Step 1: All-reduce gradients → 모든 GPU가 전체 gradient 보유")

    # Step 2: 각 GPU가 담당 파라미터만 Adam update
    for gpu_id in range(num_gpus):
        start = gpu_id * params_per_gpu
        end = start + params_per_gpu
        # 이 GPU는 m[start:end], v[start:end]만 보유
        m_local = m[start:end]
        v_local = v[start:end]
        g_local = grads[start:end]

        # Adam update (간소화)
        beta1, beta2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
        m_local = beta1 * m_local + (1 - beta1) * g_local
        v_local = beta2 * v_local + (1 - beta2) * g_local ** 2
        params[start:end] -= lr * m_local / (v_local.sqrt() + eps)

    # Step 3: All-gather updated parameters
    print(f"  Step 2: 각 GPU가 담당 params만 Adam update")
    print(f"  Step 3: All-gather → 모든 GPU가 업데이트된 전체 params 보유")


# ============================================================
# Part 2: ZeRO Stage 2 (+ Gradient Partitioning)
# ============================================================

def simulate_zero_stage2():
    """
    ZeRO-2: Optimizer states + Gradients 분산.

    ZeRO-1과 차이:
    - All-reduce 대신 Reduce-scatter 사용
    - 각 GPU가 담당 파라미터의 gradient만 보유 (나머지 버림)

    통신량: DDP와 동일 (reduce-scatter = all-reduce의 절반 + all-gather의 절반)
    메모리 절약: optimizer states + gradients 모두 1/N
    """
    print("\n" + "=" * 60)
    print("ZeRO Stage 2: + Gradient Partitioning")
    print("=" * 60)

    num_gpus = 4
    num_params = 8
    params_per_gpu = num_params // num_gpus

    print(f"\n  통신 비교:")
    print(f"  DDP:    all-reduce(gradients)  → 각 GPU가 전체 gradient 보유")
    print(f"  ZeRO-2: reduce-scatter(grads)  → 각 GPU가 담당 gradient만 보유")

    print(f"\n  메모리 비교 (per GPU):")
    print(f"  DDP:    params({num_params}) + grads({num_params}) + opt({num_params*2}) = {num_params*4}")
    zero2_mem = num_params + params_per_gpu + params_per_gpu * 2
    print(f"  ZeRO-2: params({num_params}) + grads({params_per_gpu}) + opt({params_per_gpu*2}) = {zero2_mem}")

    print(f"\n  동작 순서:")
    print(f"  1. Forward:        모든 GPU에 전체 params 있음 (동일 연산)")
    print(f"  2. Backward:       gradient 계산")
    print(f"  3. Reduce-scatter: 각 GPU가 자기 담당 gradient의 합만 받음")
    print(f"     (= all-reduce 하되, 자기 담당 아닌 부분은 바로 버림)")
    print(f"  4. Optimizer step: 담당 params만 update")
    print(f"  5. All-gather:     업데이트된 params를 모든 GPU에 broadcast")


# ============================================================
# Part 3: ZeRO Stage 3 (+ Parameter Partitioning)
# ============================================================

def simulate_zero_stage3():
    """
    ZeRO-3: 모든 것(params + grads + optimizer)을 분산. = FSDP와 동일 개념.

    핵심: forward/backward 시에도 파라미터를 필요할 때만 all-gather로 모음.

    동작:
    1. Forward의 각 layer:
       - all-gather로 해당 layer params 수집
       - forward 계산
       - 사용 끝난 params 버림 (메모리 해제)
    2. Backward의 각 layer:
       - all-gather로 해당 layer params 수집 (다시!)
       - backward 계산
       - reduce-scatter로 gradient 분산
       - params 다시 버림

    통신량: forward에 all-gather 추가 (DDP 대비 1.5x 통신)
    메모리 절약: 모든 것이 1/N → 모델 크기에 비례하여 GPU 추가 가능
    """
    print("\n" + "=" * 60)
    print("ZeRO Stage 3: Full Partitioning (= FSDP)")
    print("=" * 60)

    num_gpus = 4
    num_params = 8
    params_per_gpu = num_params // num_gpus

    print(f"\n  메모리 (per GPU):")
    zero3_mem = params_per_gpu * 4  # params/N + grads/N + m/N + v/N
    print(f"  ZeRO-3: params({params_per_gpu}) + grads({params_per_gpu}) + opt({params_per_gpu*2}) = {zero3_mem}")
    print(f"  DDP:    params({num_params}) + grads({num_params}) + opt({num_params*2}) = {num_params*4}")
    print(f"  절약: {(1 - zero3_mem / (num_params*4)) * 100:.0f}%")

    print(f"\n  Forward 동작 (layer별):")
    for layer in range(2):
        print(f"    Layer {layer}: all-gather params → forward → 사용 끝난 params 해제")

    print(f"\n  Backward 동작 (역순):")
    for layer in [1, 0]:
        print(f"    Layer {layer}: all-gather params → backward → reduce-scatter grads → params 해제")

    print(f"\n  통신량 비교:")
    print(f"    DDP:    2 * model_size (all-reduce = reduce-scatter + all-gather)")
    print(f"    ZeRO-3: 3 * model_size (fwd all-gather + bwd all-gather + reduce-scatter)")
    print(f"    → 통신 1.5x 증가, but 메모리 1/N으로 감소!")


# ============================================================
# Part 4: DeepSpeed ZeRO 사용법
# ============================================================
#
# DeepSpeed에서 ZeRO 사용 (ds_config.json):
#
# Stage 1:
#   {"zero_optimization": {"stage": 1}}
#
# Stage 2:
#   {"zero_optimization": {
#       "stage": 2,
#       "contiguous_gradients": true,  # gradient 메모리 연속 배치
#       "overlap_comm": true           # 통신과 연산 겹치기
#   }}
#
# Stage 3:
#   {"zero_optimization": {
#       "stage": 3,
#       "param_persistence_threshold": 1e6,  # 작은 params는 분산 안 함
#       "prefetch_bucket_size": 5e7           # 다음 layer params 미리 가져오기
#   }}
#
# Python 코드:
#   import deepspeed
#   model, optimizer, _, _ = deepspeed.initialize(
#       model=model, config=ds_config
#   )
#   output = model(input)
#   model.backward(loss)
#   model.step()
#
# 이 파일의 torch.distributed helper 사용 흐름:
#
#   # torchrun --nproc_per_node=4 train.py
#   dist.init_process_group("nccl")
#
#   # ZeRO-1: full params/full grads + sharded optimizer states
#   params_full = zero1_adam_step_dist(params_full, grads_full, m_shard, v_shard)
#
#   # ZeRO-2: full params + reduce-scatter로 sharded grads 획득
#   # TP/PP와 같이 쓰는 Megatron식 distributed optimizer라면,
#   # 여기서 params_full은 진짜 full model이 아니라 "이 TP/PP rank가 들고 있는 local shard를 flat하게 편 buffer"에 가깝다.
#   # DP group 안에서만 이 local shard buffer를 reduce-scatter/all-gather한다.
#   params_full, grad_shard = zero2_adam_step_dist(
#       params_full, grads_full, m_shard, v_shard
#   )
#
#   # ZeRO-3: params도 shard. layer 계산 직전에만 full params 임시 복원
#   params_full_tmp = zero3_all_gather_params_dist(param_shard)
#   # ... forward/backward with params_full_tmp ...
#   grad_shard = zero3_reduce_scatter_grads_dist(grad_full)
#   zero3_adam_step_dist(param_shard, grad_shard, m_shard, v_shard)


# ============================================================
# Part 5: 비교 요약
# ============================================================

def comparison_table():
    print("\n" + "=" * 60)
    print("ZeRO Stages Comparison (N GPUs, P params, Adam)")
    print("=" * 60)

    N = 8
    P_gb = 4.0  # 1B params = 4GB in FP32

    headers = ["", "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    rows = [
        ["Params/GPU",       f"{P_gb:.1f}",  f"{P_gb:.1f}",    f"{P_gb:.1f}",    f"{P_gb/N:.2f}"],
        ["Grads/GPU",        f"{P_gb:.1f}",  f"{P_gb:.1f}",    f"{P_gb/N:.2f}",  f"{P_gb/N:.2f}"],
        ["Opt states/GPU",   f"{P_gb*2:.1f}", f"{P_gb*2/N:.2f}", f"{P_gb*2/N:.2f}", f"{P_gb*2/N:.2f}"],
        ["Total/GPU",        f"{P_gb*4:.1f}", f"{P_gb*2+P_gb*2/N:.2f}", f"{P_gb+P_gb/N+P_gb*2/N:.2f}", f"{P_gb*4/N:.2f}"],
        ["Communication",    "2P",           "2P",              "2P",              "3P"],
        ["분산 대상",         "없음",         "opt states",      "+gradients",      "+parameters"],
    ]

    # 출력
    widths = [20, 8, 8, 8, 8]
    print("  " + "".join(h.center(w) for h, w in zip(headers, widths)))
    print("  " + "-" * sum(widths))
    for row in rows:
        print("  " + "".join(v.center(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    simulate_zero_stage1()
    simulate_zero_stage2()
    simulate_zero_stage3()
    comparison_table()
