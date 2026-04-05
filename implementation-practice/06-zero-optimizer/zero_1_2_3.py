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
"""

import torch
import torch.nn as nn
import numpy as np


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
