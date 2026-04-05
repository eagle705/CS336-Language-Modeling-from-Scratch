"""
DL Systems Concepts
=====================
대규모 모델 학습에 필요한 시스템 레벨 지식.

GPU, 네트워크, throughput 분석 등.
"""

import torch


# ============================================================
# Part 1: GPU Architecture Basics
# ============================================================

def gpu_specs():
    """주요 GPU 스펙 비교 및 bottleneck 분석."""
    print("=" * 60)
    print("GPU Specs & Bottleneck Analysis")
    print("=" * 60)

    gpus = [
        # (name, FP16 TFLOPS, HBM GB, BW GB/s)
        ("A100 80GB",   312,  80, 2039),
        ("H100 80GB",   990,  80, 3352),
        ("H200 141GB", 990, 141, 4800),
        ("B200 192GB", 2250, 192, 8000),
    ]

    print(f"\n  {'GPU':<16} {'FP16 TFLOPS':>12} {'HBM (GB)':>10} {'BW (GB/s)':>10} {'AI (F/B)':>10}")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, tflops, mem, bw in gpus:
        # Arithmetic Intensity = FLOPS / Bandwidth
        # 연산이 이 값보다 높으면 compute-bound, 낮으면 memory-bound
        ai = tflops * 1e12 / (bw * 1e9)
        print(f"  {name:<16} {tflops:>12} {mem:>10} {bw:>10} {ai:>10.0f}")

    print(f"\n  Arithmetic Intensity (AI) = FLOPS / Bytes")
    print(f"  AI > GPU의 F/B ratio → Compute-bound (좋음, GPU 활용 높음)")
    print(f"  AI < GPU의 F/B ratio → Memory-bound (BW가 bottleneck)")
    print(f"\n  Matmul은 보통 compute-bound (높은 AI)")
    print(f"  Element-wise ops (ReLU, LayerNorm)는 memory-bound (낮은 AI)")
    print(f"  → operator fusion으로 memory-bound ops를 합쳐서 BW 절약")


# ============================================================
# Part 2: Interconnect
# ============================================================

def interconnect_specs():
    """GPU 간 통신 대역폭."""
    print("\n" + "=" * 60)
    print("Interconnect Bandwidth")
    print("=" * 60)

    links = [
        # (name, BW GB/s, 특징)
        ("PCIe Gen4 x16",    32,  "CPU-GPU, 느림"),
        ("PCIe Gen5 x16",    64,  "CPU-GPU, 느림"),
        ("NVLink 3 (A100)",  600, "GPU-GPU, 노드 내"),
        ("NVLink 4 (H100)",  900, "GPU-GPU, 노드 내"),
        ("NVLink 5 (B200)", 1800, "GPU-GPU, 노드 내"),
        ("InfiniBand HDR",    25, "노드 간, 1포트"),
        ("InfiniBand NDR",    50, "노드 간, 1포트"),
        ("InfiniBand NDR x8", 400, "노드 간, 8포트 (H100 DGX)"),
    ]

    print(f"\n  {'Link':<22} {'BW (GB/s)':>10} {'Note'}")
    print(f"  {'-'*22} {'-'*10} {'-'*30}")
    for name, bw, note in links:
        print(f"  {name:<22} {bw:>10} {note}")

    print(f"\n  핵심 포인트:")
    print(f"    NVLink >> IB → TP는 노드 내 (NVLink), DP는 노드 간 (IB) 허용")
    print(f"    IB 대역폭을 높이려면 멀티포트 (Rail-optimized) 필요")


# ============================================================
# Part 3: Throughput / MFU 분석
# ============================================================

def throughput_analysis():
    """Model FLOPS Utilization (MFU) 계산."""
    print("\n" + "=" * 60)
    print("Throughput / MFU Analysis")
    print("=" * 60)

    # 예: LLaMA-7B on 8x H100
    P = 7e9         # parameters
    B = 4            # micro batch per GPU
    S = 2048         # seq length
    num_gpus = 8
    gpu_tflops = 990  # H100 FP16

    # Transformer FLOPs per token ≈ 6 * P (forward + backward ≈ 3x forward)
    # forward만: 2 * P per token
    flops_per_token = 6 * P

    # Tokens per step
    tokens_per_step = B * S * num_gpus

    # Total FLOPs per step
    total_flops = flops_per_token * tokens_per_step

    # 가정: step time
    step_time_ms = 500  # 예시

    # Achieved TFLOPS
    achieved_tflops = total_flops / (step_time_ms / 1000) / 1e12

    # MFU = achieved / theoretical peak
    peak_tflops = gpu_tflops * num_gpus
    mfu = achieved_tflops / peak_tflops

    # Tokens per second
    tokens_per_sec = tokens_per_step / (step_time_ms / 1000)

    print(f"\n  Model: 7B params")
    print(f"  Hardware: {num_gpus}x H100 ({gpu_tflops} TFLOPS each)")
    print(f"  Batch: {B} × {S} tokens × {num_gpus} GPUs = {tokens_per_step:,} tokens/step")

    print(f"\n  FLOPs per token: 6P = {flops_per_token/1e9:.0f}B")
    print(f"  FLOPs per step:  {total_flops/1e12:.1f} TFLOP")
    print(f"  Step time:       {step_time_ms}ms (assumed)")

    print(f"\n  Achieved:        {achieved_tflops:.1f} TFLOPS")
    print(f"  Peak:            {peak_tflops:,} TFLOPS")
    print(f"  MFU:             {mfu*100:.1f}%")
    print(f"  Tokens/sec:      {tokens_per_sec:,.0f}")

    print(f"\n  좋은 MFU 기준:")
    print(f"    > 50%: 우수 (대부분의 최적화 적용)")
    print(f"    30-50%: 보통 (통신/메모리 bottleneck)")
    print(f"    < 30%: 개선 필요")


# ============================================================
# Part 4: Training Time / Cost 추정
# ============================================================

def training_cost_estimate():
    """학습 시간과 비용 추정."""
    print("\n" + "=" * 60)
    print("Training Cost Estimation")
    print("=" * 60)

    # Chinchilla optimal: tokens ≈ 20 * params
    configs = [
        ("7B",   7e9,   140e9,   8,  990),
        ("70B",  70e9,  1.4e12,  256, 990),
        ("405B", 405e9, 15e12,   16384, 990),
    ]

    print(f"\n  {'Model':<8} {'Params':>8} {'Tokens':>10} {'GPUs':>6} {'Days':>8} {'GPU-hours':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*10}")

    for name, params, tokens, gpus, gpu_tflops in configs:
        # Total FLOPs: 6 * P * T
        total_flops = 6 * params * tokens

        # Assume 40% MFU
        mfu = 0.40
        effective_tflops = gpu_tflops * gpus * mfu

        # Time in seconds
        time_sec = total_flops / (effective_tflops * 1e12)
        time_days = time_sec / 86400
        gpu_hours = time_sec * gpus / 3600

        print(f"  {name:<8} {params/1e9:>7.0f}B {tokens/1e9:>8.0f}B {gpus:>6} "
              f"{time_days:>7.1f}d {gpu_hours:>10,.0f}")

    print(f"\n  Cost (H100 @ $2/GPU-hour):")
    for name, params, tokens, gpus, _ in configs:
        total_flops = 6 * params * tokens
        time_sec = total_flops / (990 * gpus * 0.4 * 1e12)
        gpu_hours = time_sec * gpus / 3600
        cost = gpu_hours * 2
        print(f"    {name}: ${cost:,.0f}")


# ============================================================
# Part 5: Profiling
# ============================================================
#
# PyTorch Profiler 사용법:
#
# from torch.profiler import profile, schedule, tensorboard_trace_handler
#
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
#     on_trace_ready=tensorboard_trace_handler("./log"),
#     record_shapes=True,
#     with_stack=True,
# ) as prof:
#     for step, batch in enumerate(dataloader):
#         output = model(batch)
#         loss.backward()
#         optimizer.step()
#         prof.step()
#
# # 결과 확인
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#
# # TensorBoard에서 시각화:
# # tensorboard --logdir=./log
#
# 핵심 지표:
#   - GPU Utilization: GPU가 실제 연산하는 시간 비율
#   - SM Efficiency: GPU SM(Streaming Multiprocessor) 활용률
#   - Memory throughput: HBM 대역폭 활용률
#   - Kernel launch overhead: 작은 kernel이 많으면 launch 비용 큼
#     → torch.compile로 fusion하여 해결


if __name__ == "__main__":
    gpu_specs()
    interconnect_specs()
    throughput_analysis()
    training_cost_estimate()
