"""
Virtual Pipeline Parallelism (VPP)
====================================
일반 PP의 bubble을 줄이는 Megatron-LM의 핵심 기법.

일반 PP의 문제: Pipeline bubble
  PP=4, M=8 micro-batches일 때:
  Bubble ratio = (PP-1)/M = 3/8 = 37.5%

VPP 아이디어: 각 GPU에 연속된 layer 대신 "비연속 layer 묶음"을 배치.
  virtual_pipeline_model_parallel_size (= V) = 각 GPU가 담당하는 chunk 수

일반 PP (V=1): 각 GPU가 연속된 layer 묶음 하나
  GPU 0: [Layer 0-3]
  GPU 1: [Layer 4-7]
  GPU 2: [Layer 8-11]
  GPU 3: [Layer 12-15]

VPP (V=2): 각 GPU가 2개의 비연속 chunk
  GPU 0: [Layer 0-1] + [Layer 8-9]     ← 2 chunks
  GPU 1: [Layer 2-3] + [Layer 10-11]
  GPU 2: [Layer 4-5] + [Layer 12-13]
  GPU 3: [Layer 6-7] + [Layer 14-15]

  Forward: 0→1→2→3→0→1→2→3 (2 round trips)
           chunk0        chunk1

왜 bubble이 줄어드나?
  - 일반 PP: forward가 stage 0→3까지 가야 backward 시작 (3 time slots 대기)
  - VPP: chunk가 2개이므로 chunk 0의 forward가 빨리 끝남 → 더 빨리 backward 시작

  Bubble ratio = (PP-1) / (M × V)
  V=1: (4-1)/(8×1) = 37.5%
  V=2: (4-1)/(8×2) = 18.75%  ← 절반!
  V=4: (4-1)/(8×4) = 9.375%

  대가: 통신 증가 (round trip이 V번)
"""

import torch
import torch.nn as nn


# ============================================================
# Part 1: 일반 PP vs VPP Schedule 시각화
# ============================================================

def visualize_schedules():
    """일반 PP와 VPP의 schedule을 나란히 비교."""
    print("=" * 70)
    print("Pipeline Schedule: Regular PP vs VPP")
    print("=" * 70)

    PP = 4
    M = 8

    # === 일반 PP (V=1): 1F1B ===
    print(f"\n  [Regular PP] V=1, Bubble ratio = {(PP-1)/(M*1)*100:.1f}%")
    print(f"  {'-'*60}")

    for s in range(PP):
        timeline = []
        f_done = b_done = 0

        # Delay
        for _ in range(s):
            timeline.append("   ")

        # Warmup: PP - s forwards
        for _ in range(PP - s):
            if f_done < M:
                timeline.append(f"F{f_done} ")
                f_done += 1

        # Steady: 1B + 1F
        while f_done < M or b_done < M:
            if b_done < M:
                timeline.append(f"B{b_done} ")
                b_done += 1
            if f_done < M:
                timeline.append(f"F{f_done} ")
                f_done += 1

        # Cooldown
        while b_done < M:
            timeline.append(f"B{b_done} ")
            b_done += 1

        max_len = 2 * M + PP - 1
        while len(timeline) < max_len:
            timeline.append("   ")

        print(f"    GPU {s}: {'|'.join(timeline[:max_len])}")

    # === VPP (V=2): Interleaved 1F1B ===
    V = 2
    print(f"\n  [VPP] V={V}, Bubble ratio = {(PP-1)/(M*V)*100:.1f}%")
    print(f"  {'-'*70}")
    print(f"  (F0.0 = micro-batch 0의 virtual chunk 0)")

    for s in range(PP):
        timeline = []
        # VPP에서 각 micro-batch는 V번 forward (chunk 0, 1, ...)
        # 총 forward units = M × V
        total_units = M * V
        f_done = b_done = 0

        # Delay
        for _ in range(s):
            timeline.append("     ")

        # Warmup
        warmup = PP - s
        for _ in range(warmup):
            if f_done < total_units:
                mb = f_done // V
                chunk = f_done % V
                timeline.append(f"F{mb}.{chunk} ")
                f_done += 1

        # Steady: 1B + 1F
        while f_done < total_units or b_done < total_units:
            if b_done < total_units:
                mb = b_done // V
                chunk = b_done % V
                timeline.append(f"B{mb}.{chunk} ")
                b_done += 1
            if f_done < total_units:
                mb = f_done // V
                chunk = f_done % V
                timeline.append(f"F{mb}.{chunk} ")
                f_done += 1

        # Cooldown
        while b_done < total_units:
            mb = b_done // V
            chunk = b_done % V
            timeline.append(f"B{mb}.{chunk} ")
            b_done += 1

        # Truncate for readability
        display = '|'.join(timeline[:20])
        if len(timeline) > 20:
            display += f"|... ({len(timeline)} total)"
        print(f"    GPU {s}: {display}")


# ============================================================
# Part 2: VPP Layer 배치 시뮬레이션
# ============================================================

def simulate_vpp_layer_assignment():
    """VPP에서 layer가 GPU에 어떻게 배치되는지 시뮬레이션."""
    print("\n" + "=" * 70)
    print("VPP Layer Assignment")
    print("=" * 70)

    configs = [
        (16, 4, 1, "Regular PP"),
        (16, 4, 2, "VPP V=2"),
        (16, 4, 4, "VPP V=4"),
        (32, 4, 2, "32 layers, VPP V=2"),
    ]

    for num_layers, pp_size, V, label in configs:
        layers_per_chunk = num_layers // (pp_size * V)
        print(f"\n  [{label}] {num_layers} layers, PP={pp_size}, V={V}"
              f" → {layers_per_chunk} layers/chunk")

        for gpu in range(pp_size):
            chunks = []
            for v in range(V):
                # chunk v의 시작 layer
                # Megatron-LM 배치: gpu가 담당하는 v번째 chunk
                start = (v * pp_size + gpu) * layers_per_chunk
                end = start + layers_per_chunk
                chunks.append(f"L{start}-{end-1}")

            print(f"    GPU {gpu}: {' + '.join(chunks)}")

        # Forward 순서
        forward_order = []
        for v in range(V):
            for gpu in range(pp_size):
                start = (v * pp_size + gpu) * layers_per_chunk
                forward_order.append(f"GPU{gpu}(L{start}-{start + layers_per_chunk - 1})")
        print(f"    Forward: {' → '.join(forward_order)}")


# ============================================================
# Part 3: Forward Pass 시뮬레이션
# ============================================================

def simulate_vpp_forward():
    """VPP forward pass를 실제 데이터로 시뮬레이션."""
    print("\n" + "=" * 70)
    print("VPP Forward Simulation")
    print("=" * 70)

    torch.manual_seed(42)

    num_layers = 8
    pp_size = 2
    V = 2
    embed_dim = 8
    ffn_hidden = 16

    layers_per_chunk = num_layers // (pp_size * V)  # 2

    # Weight
    W1 = [torch.randn(embed_dim, ffn_hidden) * 0.1 for _ in range(num_layers)]
    W2 = [torch.randn(ffn_hidden, embed_dim) * 0.1 for _ in range(num_layers)]

    x = torch.randn(4, embed_dim)

    # --- Single GPU (baseline) ---
    x_ref = x.clone()
    for i in range(num_layers):
        x_ref = x_ref + torch.nn.functional.gelu(x_ref @ W1[i]) @ W2[i]

    # --- VPP Forward ---
    x_vpp = x.clone()

    # Layer 배치:
    # GPU 0: chunk0=[L0,L1], chunk1=[L4,L5]
    # GPU 1: chunk0=[L2,L3], chunk1=[L6,L7]
    gpu_layers = {0: [], 1: []}
    for v in range(V):
        for gpu in range(pp_size):
            start = (v * pp_size + gpu) * layers_per_chunk
            for offset in range(layers_per_chunk):
                gpu_layers[gpu].append(start + offset)

    print(f"\n  Config: {num_layers}L, PP={pp_size}, V={V}")
    print(f"  GPU 0 layers: {gpu_layers[0]}")
    print(f"  GPU 1 layers: {gpu_layers[1]}")

    # Forward: chunk 0 (GPU0 → GPU1), chunk 1 (GPU0 → GPU1)
    print(f"\n  Forward execution order:")
    for v in range(V):
        print(f"    Virtual chunk {v}:")
        for gpu in range(pp_size):
            start = (v * pp_size + gpu) * layers_per_chunk
            for offset in range(layers_per_chunk):
                layer_idx = start + offset
                x_vpp = x_vpp + torch.nn.functional.gelu(x_vpp @ W1[layer_idx]) @ W2[layer_idx]
                print(f"      GPU {gpu}: Layer {layer_idx}")

        if v < V - 1:
            print(f"    (back to GPU 0 for chunk {v+1})")

    diff = (x_ref - x_vpp).abs().max().item()
    print(f"\n  Single GPU vs VPP diff: {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")


# ============================================================
# Part 4: Bubble 비율 분석
# ============================================================

def bubble_analysis():
    """다양한 설정에서 bubble 비율 비교."""
    print("\n" + "=" * 70)
    print("Bubble Ratio Analysis")
    print("=" * 70)

    print(f"\n  Bubble ratio = (PP - 1) / (M × V)")
    print(f"\n  {'PP':>4} {'V':>4} {'M':>4} {'Bubble':>10} {'통신 round trips':>20}")
    print(f"  {'-'*4} {'-'*4} {'-'*4} {'-'*10} {'-'*20}")

    for PP in [4, 8]:
        for V in [1, 2, 4]:
            for M in [8, 16, 32]:
                bubble = (PP - 1) / (M * V) * 100
                trips = V  # forward가 V번 pipeline 통과
                print(f"  {PP:>4} {V:>4} {M:>4} {bubble:>9.1f}% {trips:>20}")
        print()

    print("""
  Trade-off:
    V↑ → bubble↓ (좋음) + 통신 round trips↑ (나쁨)
    M↑ → bubble↓ (좋음) + memory↑ (나쁨, gradient accumulation)

  실전 권장:
    V=2가 대부분 sweet spot (bubble 절반, 통신 2배 = 감당 가능)
    V=4 이상은 통신 overhead가 bubble 감소보다 클 수 있음
    """)


# ============================================================
# Part 5: Megatron-Core VPP 설정
# ============================================================
#
# config = TransformerConfig(
#     pipeline_model_parallel_size=4,
#     virtual_pipeline_model_parallel_size=2,  # ← V=2
#     num_layers=32,
#     ...
# )
#
# # Megatron-Core 내부 동작:
# # 1. Layer 배치
# #    num_layers_per_virtual_stage = num_layers / (PP * V) = 32 / (4*2) = 4
# #    GPU 0: virtual stage 0 (L0-3) + virtual stage 4 (L16-19)
# #    GPU 1: virtual stage 1 (L4-7) + virtual stage 5 (L20-23)
# #    ...
# #
# # 2. Schedule
# #    forward_backward_pipelining_with_interleaving()
# #    schedules.py에서 구현
# #    각 micro-batch가 V번 pipeline을 통과
# #
# # 3. 통신
# #    virtual stage 경계에서도 send/recv 발생
# #    같은 GPU 내 virtual stage 간에는 통신 불필요 (local)
#
# # torchrun 실행:
# # torchrun --nproc_per_node=8 pretrain_gpt.py \
# #     --pipeline-model-parallel-size 4 \
# #     --virtual-pipeline-model-parallel-size 2 \
# #     --num-layers 32


# ============================================================
# Part 6: Memory 영향
# ============================================================

def memory_impact():
    print("\n" + "=" * 70)
    print("VPP Memory Impact")
    print("=" * 70)

    print("""
  VPP는 bubble을 줄이지만 메모리에 영향:

  Parameters:
    동일! 각 GPU의 총 layer 수는 같음 (배치만 다름)
    Regular: GPU 0 = [L0-3]     → 4 layers
    VPP V=2: GPU 0 = [L0-1, L8-9] → 4 layers

  Activations:
    VPP는 각 virtual chunk의 activation을 따로 보관해야 함.
    Regular: 1 set of activations (4 layers)
    VPP V=2: 2 sets (각 2 layers), but 동시에 보관할 수 있음
    → activation memory는 비슷하거나 약간 증가

  Communication buffers:
    VPP는 round trip이 많아서 send/recv buffer가 더 필요
    → 약간의 메모리 overhead

  정리:
    ┌────────────┬─────────────────┬───────────────────┐
    │            │ Regular PP (V=1) │ VPP (V=2)         │
    ├────────────┼─────────────────┼───────────────────┤
    │ Bubble     │ (PP-1)/M        │ (PP-1)/(M×V)      │
    │ Parameters │ L/PP per GPU    │ L/PP per GPU (동일)│
    │ Activation │ baseline        │ ~동일 or 약간 증가 │
    │ P2P 통신   │ V=1 round trip  │ V round trips     │
    │ 구현 복잡도 │ 낮음            │ 높음 (interleaved) │
    └────────────┴─────────────────┴───────────────────┘
    """)


if __name__ == "__main__":
    visualize_schedules()
    simulate_vpp_layer_assignment()
    simulate_vpp_forward()
    bubble_analysis()
    memory_impact()
