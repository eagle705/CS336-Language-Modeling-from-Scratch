"""
Distributed Optimizer
=======================
Optimizer states를 DP ranks에 분산하여 메모리 절약.

문제: Adam optimizer는 param당 2개 추가 state (m, v) 보유 → FP32 기준 8x overhead
  7B model: params 14GB (BF16) + optimizer 56GB (FP32 m,v) + master 28GB (FP32)
  → optimizer가 메모리의 대부분!

해결: optimizer states를 DP group에 분산 (= ZeRO Stage 1)

일반 DDP:
  모든 GPU: [full params] + [full grads] + [full opt states (m, v)]

Distributed Optimizer (ZeRO-1):
  GPU 0: [full params] + [full grads] + [opt states for params 0-24%]
  GPU 1: [full params] + [full grads] + [opt states for params 25-49%]
  GPU 2: [full params] + [full grads] + [opt states for params 50-74%]
  GPU 3: [full params] + [full grads] + [opt states for params 75-100%]

Megatron Distributed Optimizer (ZeRO-1 + 최적화):
  DDP의 gradient all-reduce 대신:
  1. reduce-scatter: 각 GPU가 담당 파라미터의 gradient만 받음
  2. 담당 파라미터만 optimizer step (FP32)
  3. all-gather: 업데이트된 파라미터를 전체 GPU에 broadcast

  일반 DDP:           all-reduce(grads) → full optimizer step
  Dist Optimizer:     reduce-scatter(grads) → partial step → all-gather(params)
  통신량: 동일! (reduce-scatter + all-gather = all-reduce)
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# Part 1: 일반 Adam vs Distributed Adam 시뮬레이션
# ============================================================

def simulate_distributed_optimizer():
    """Distributed Optimizer의 동작을 시뮬레이션."""
    print("=" * 60)
    print("Distributed Optimizer Simulation")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    num_params = 16  # 총 파라미터 수
    params_per_gpu = num_params // num_gpus

    # 모델 파라미터 (모든 GPU 동일)
    params = torch.randn(num_params)

    # 각 GPU의 gradient (다른 데이터 → 다른 gradient)
    gpu_grads = [torch.randn(num_params) for _ in range(num_gpus)]

    print(f"\n  Config: {num_gpus} GPUs, {num_params} params")
    print(f"  Each GPU owns: {params_per_gpu} params' optimizer states")

    # === 일반 DDP: all-reduce → full optimizer step ===
    print(f"\n  [일반 DDP Adam]")

    # All-reduce: gradient 평균
    avg_grad = sum(gpu_grads) / num_gpus

    # 모든 GPU에서 동일한 full Adam step
    m_full = torch.zeros(num_params)  # 모든 GPU가 전체 m 보유
    v_full = torch.zeros(num_params)
    beta1, beta2, lr, eps = 0.9, 0.999, 1e-3, 1e-8

    m_full = beta1 * m_full + (1 - beta1) * avg_grad
    v_full = beta2 * v_full + (1 - beta2) * avg_grad ** 2
    params_ddp = params - lr * m_full / (v_full.sqrt() + eps)

    ddp_opt_memory = num_params * 2  # m + v (전체)
    print(f"    Optimizer memory per GPU: {ddp_opt_memory} values (full m + v)")

    # === Distributed Optimizer: reduce-scatter → partial step → all-gather ===
    print(f"\n  [Distributed Optimizer]")

    params_dist = params.clone()
    dist_opt_memory = params_per_gpu * 2  # m + v (부분)

    # Step 1: reduce-scatter (각 GPU가 담당 params의 gradient만 받음)
    for gpu_id in range(num_gpus):
        start = gpu_id * params_per_gpu
        end = start + params_per_gpu

        # 이 GPU가 담당하는 params의 gradient만 reduce
        local_grad = sum(g[start:end] for g in gpu_grads) / num_gpus

        # Step 2: 담당 params만 Adam step (FP32)
        m_local = torch.zeros(params_per_gpu)
        v_local = torch.zeros(params_per_gpu)

        m_local = beta1 * m_local + (1 - beta1) * local_grad
        v_local = beta2 * v_local + (1 - beta2) * local_grad ** 2
        params_dist[start:end] = params[start:end] - lr * m_local / (v_local.sqrt() + eps)

    # Step 3: all-gather (업데이트된 params를 모든 GPU에 broadcast)
    # (시뮬레이션에서는 params_dist가 이미 전체를 가짐)

    diff = (params_ddp - params_dist).abs().max().item()
    print(f"    Optimizer memory per GPU: {dist_opt_memory} values (partial m + v)")
    print(f"    Memory 절약: {(1 - dist_opt_memory / ddp_opt_memory) * 100:.0f}%")
    print(f"\n  DDP vs Distributed Optimizer diff: {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-6 else 'FAILED'}")


# ============================================================
# Part 2: 통신 패턴 비교
# ============================================================

def communication_comparison():
    """DDP vs Distributed Optimizer의 통신 패턴."""
    print("\n" + "=" * 60)
    print("Communication: DDP vs Distributed Optimizer")
    print("=" * 60)

    print("""
  일반 DDP:
    backward 중: all-reduce(gradients)
    → reduce-scatter + all-gather = 2 × model_size

    [backward] ──→ [all-reduce grads] ──→ [optimizer step] ──→ [next fwd]
                    ↑ 2 × P bytes          ↑ 전체 params

  Distributed Optimizer:
    backward 후: reduce-scatter(grads) → partial step → all-gather(params)
    → reduce-scatter + all-gather = 2 × model_size (동일!)

    [backward] ──→ [reduce-scatter] ──→ [partial step] ──→ [all-gather] ──→ [next fwd]
                    ↑ P bytes             ↑ 담당만          ↑ P bytes

  통신량: 완전히 동일!
  차이: optimizer memory가 1/DP_size로 감소

  그런데 왜 Megatron은 Distributed Optimizer를 따로 구현?
    1. DDP의 all-reduce를 reduce-scatter + all-gather로 분리
    2. reduce-scatter → optimizer step → all-gather 사이에
       FP32 master weights를 사용한 정밀한 update 가능
    3. gradient를 받자마자 바로 step → optimizer step과 통신 overlap
    """)


# ============================================================
# Part 3: Megatron Distributed Optimizer 상세
# ============================================================

def megatron_dist_optimizer():
    """Megatron-Core의 Distributed Optimizer 구현 상세."""
    print("\n" + "=" * 60)
    print("Megatron Distributed Optimizer Details")
    print("=" * 60)

    print("""
  Megatron의 DistributedOptimizer는 일반 ZeRO-1보다 더 최적화:

  1. Contiguous gradient buffer:
     모든 params의 gradient를 하나의 연속 buffer에 배치
     → reduce-scatter가 한 번의 NCCL call로 가능

     일반:     [grad_param0] [grad_param1] ... (분산된 메모리)
     Megatron: [grad_param0 | grad_param1 | ...] (연속 buffer)

  2. FP32 master weights + BF16 gradient:
     reduce-scatter: BF16 gradient (통신량 절약)
     optimizer step: FP32 master weights (정밀도 유지)
     all-gather: BF16 params (통신량 절약)

     GPU 0의 담당 params:
       BF16 grad (받음) → FP32로 변환 → FP32 m,v 업데이트
       → FP32 master weights 업데이트 → BF16로 변환 → all-gather

  3. Overlap:
     reduce-scatter와 backward의 나머지 부분을 겹침
     all-gather와 다음 forward의 앞부분을 겹침

  코드 (Megatron-Core):
    megatron/core/optimizer/distrib_optimizer.py
    핵심 클래스: DistributedOptimizer

    # 사용법
    from megatron.core.optimizer import get_megatron_optimizer

    optimizer = get_megatron_optimizer(
        config,
        model,
        no_weight_decay_cond=...,
        scale_lr_cond=...,
    )
    # config에 use_distributed_optimizer=True면 자동으로 DistributedOptimizer 사용
    """)


# ============================================================
# Part 4: 메모리 분석
# ============================================================

def memory_analysis():
    """Distributed Optimizer의 메모리 절약 효과."""
    print("\n" + "=" * 60)
    print("Memory Analysis")
    print("=" * 60)

    P_gb = 7.0  # 7B params in FP32 = 28GB, BF16 = 14GB
    bf16 = 2
    fp32 = 4

    for dp_size in [1, 4, 8, 64]:
        print(f"\n  7B model, DP={dp_size}:")

        # 일반 DDP
        ddp_params = P_gb * bf16
        ddp_grads = P_gb * bf16
        ddp_opt = P_gb * fp32 * 2  # m + v
        ddp_master = P_gb * fp32
        ddp_total = ddp_params + ddp_grads + ddp_opt + ddp_master

        # Distributed Optimizer
        dist_params = P_gb * bf16
        dist_grads = P_gb * bf16  # full grads (reduce-scatter 전)
        dist_opt = P_gb * fp32 * 2 / dp_size  # m + v 분산
        dist_master = P_gb * fp32 / dp_size   # master weights 분산
        dist_total = dist_params + dist_grads + dist_opt + dist_master

        print(f"    {'Component':<25} {'DDP':>8} {'Dist Opt':>10}")
        print(f"    {'-'*25} {'-'*8} {'-'*10}")
        print(f"    {'Params (BF16)':<25} {ddp_params:>7.1f}G {dist_params:>9.1f}G")
        print(f"    {'Gradients (BF16)':<25} {ddp_grads:>7.1f}G {dist_grads:>9.1f}G")
        print(f"    {'Opt states (FP32 m,v)':<25} {ddp_opt:>7.1f}G {dist_opt:>9.1f}G")
        print(f"    {'Master weights (FP32)':<25} {ddp_master:>7.1f}G {dist_master:>9.1f}G")
        print(f"    {'-'*25} {'-'*8} {'-'*10}")
        print(f"    {'Total':<25} {ddp_total:>7.1f}G {dist_total:>9.1f}G"
              f"  ({(1-dist_total/ddp_total)*100:.0f}% 절약)")


# ============================================================
# Part 5: ZeRO Stage 1/2/3 vs Megatron Dist Optimizer
# ============================================================

def comparison_with_zero():
    print("\n" + "=" * 60)
    print("Megatron Dist Optimizer vs ZeRO Stages")
    print("=" * 60)

    print("""
  ┌──────────────────────┬──────────────┬──────────────┬───────────────┐
  │                      │ Megatron     │ ZeRO-1       │ ZeRO-2        │
  │                      │ Dist Opt     │ (DeepSpeed)  │ (DeepSpeed)   │
  ├──────────────────────┼──────────────┼──────────────┼───────────────┤
  │ Opt states 분산       │ ✓ (1/DP)     │ ✓ (1/DP)     │ ✓ (1/DP)      │
  │ Gradients 분산        │ ✗            │ ✗            │ ✓ (1/DP)      │
  │ Params 분산           │ ✗            │ ✗            │ ✗             │
  ├──────────────────────┼──────────────┼──────────────┼───────────────┤
  │ Contiguous buffer    │ ✓            │ ✗            │ ✗             │
  │ FP32 master weights  │ ✓ (분산)      │ ✓ (분산)      │ ✓ (분산)       │
  │ Overlap comm/compute │ ✓ (최적화)    │ 부분적       │ 부분적        │
  │ TP와 통합             │ ✓ (native)   │ 별도 설정    │ 별도 설정     │
  └──────────────────────┴──────────────┴──────────────┴───────────────┘

  Megatron Dist Optimizer의 장점:
    1. TP/PP와 네이티브 통합 → 추가 설정 없이 자동 호환
    2. Contiguous buffer → 한 번의 NCCL reduce-scatter call
    3. gradient bucket과 optimizer shard가 정렬되어 효율적
    4. Megatron의 1F1B schedule과 최적화된 overlap

  ZeRO의 장점:
    1. Stage 2, 3으로 더 많은 메모리 절약 가능
    2. DeepSpeed 생태계와 통합
    3. CPU offloading, NVMe offloading 지원

  실전:
    Megatron-Core 사용 시 → Distributed Optimizer (기본)
    DeepSpeed 사용 시 → ZeRO Stage 1 또는 2
    FSDP 사용 시 → FSDP가 자체적으로 ZeRO-3 방식 사용
    """)


if __name__ == "__main__":
    simulate_distributed_optimizer()
    communication_comparison()
    megatron_dist_optimizer()
    memory_analysis()
    comparison_with_zero()
