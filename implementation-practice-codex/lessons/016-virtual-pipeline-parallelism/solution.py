"""
Virtual Pipeline Parallelism (VPP)
====================================
일반 PP의 bubble을 줄이는 Megatron-LM의 핵심 기법.

일반 PP의 문제: Pipeline bubble
  PP=4, M=8 micro-batches일 때:
  Bubble ratio = (PP-1)/M = 3/8 = 37.5%
  - 왜 ratio인가?
    pipeline이 steady state로 M개의 micro-batch를 처리하는 동안,
    앞/뒤에서 대략 PP-1개 worth의 빈 시간 슬롯이 추가로 생긴다고 보는 근사다.
    그래서 "실제 일한 micro-batch 슬롯 M" 대비 "비어 있던 슬롯 PP-1"의 비율로 bubble을 잡는다.
  - PP-1인 이유: 첫 micro-batch가 마지막 pipeline stage까지 도달하려면
    나머지 stage 수만큼 warmup 빈 slot이 생긴다. PP=4면 stage 0→1→2→3이라 3칸.
  - M은 micro-batch "개수"다. micro-batch size(각 micro-batch 안의 sample 수)가 아님.
    global batch를 몇 조각으로 쪼개 pipeline에 흘려보내는지에 해당한다.

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
  - VPP: micro-batch 하나는 chunk 1의 마지막 layer까지 forward해야 backward 가능.
    즉 chunk 0만 빨리 끝났다고 바로 backward하는 것은 아님.
    대신 각 GPU의 layer를 더 작은 virtual chunk로 나눠 interleave하므로,
    같은 M개 micro-batch 안에서 pipeline slot을 더 촘촘히 채워 bubble 비율이 줄어든다.
    interleave = micro-batch와 virtual chunk 작업을 번갈아 끼워 넣어 실행한다는 뜻.
    각 device가 한 번에 forward하는 layer 수가 줄어 작업 단위가 작아지고,
    scheduler가 다른 micro-batch의 chunk0 또는 이전 micro-batch의 chunk1을 더 빨리 배치할 수 있다.

  Bubble ratio = (PP-1) / (M × V)
  - 왜 M×V인가?
    VPP는 각 micro-batch를 V개의 virtual chunk 단위로 더 잘게 흘려보내므로,
    pipeline scheduler 입장에서는 M개 micro-batch가 M×V개의 작은 작업 단위처럼 보인다.
    같은 PP-1개의 bubble slot을 더 많은 작업 단위가 나눠 가지므로 bubble ratio가 줄어든다.
    단, 이 수식은 직관용 근사이며 실제 utilization은 warmup/cooldown과 통신 비용에 따라 달라진다.
  - 글/논문의 bubble time 식과의 관계:
    bubble time = (PP-1) × (t_f + t_b) / V
    여기서 t_f, t_b는 VPP로 쪼개기 전 한 physical stage의 forward/backward 시간.
    V개의 virtual chunk로 나누면 chunk 하나의 시간이 t_f/V, t_b/V가 되므로 bubble time도 1/V로 줄어든다.
    이를 M개 micro-batch의 유효 작업 시간 M × (t_f + t_b)로 나누면
    bubble ratio ≈ [(PP-1) × (t_f+t_b) / V] / [M × (t_f+t_b)] = (PP-1)/(M×V).
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

    # === VPP (V=2): Interleaved 1F1B 예시 schedule ===
    # VPP와 1F1B는 다른 개념이다:
    #   - VPP: layer를 virtual chunk로 쪼개서 physical stage 안에 interleave 배치하는 방법
    #   - 1F1B: forward 1개와 backward 1개를 번갈아 실행하는 pipeline schedule
    # Megatron의 VPP는 보통 interleaved 1F1B schedule과 함께 쓰이지만,
    # VPP 자체가 반드시 1F1B를 의미하는 것은 아니다.
    # 실제 실행도 warmup은 forward 위주, steady state는 1F1B, cooldown은 backward 위주로 나뉜다.
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

        # 바깥 루프는 physical pipeline stage(GPU)를 하나씩 돈다.
        # 각 GPU가 어떤 virtual chunks(v=0..V-1)를 맡는지 출력하기 위한 루프.
        # 예: pp_size=4이면 GPU0, GPU1, GPU2, GPU3의 layer 배치를 각각 계산한다.
        for gpu in range(pp_size):
            chunks = []
            for v in range(V):
                # 전체 layer를 pp_size * V개의 virtual chunk로 먼저 일렬 배치한다.
                # 예: num_layers=16, pp_size=4, V=2이면 총 8 chunks, 2 layers/chunk:
                #   global chunk 0: L0-1   -> GPU0, v=0
                #   global chunk 1: L2-3   -> GPU1, v=0
                #   global chunk 2: L4-5   -> GPU2, v=0
                #   global chunk 3: L6-7   -> GPU3, v=0
                #   global chunk 4: L8-9   -> GPU0, v=1
                #   global chunk 5: L10-11 -> GPU1, v=1
                #   ...
                # v * pp_size + gpu = 이 GPU가 맡는 v번째 virtual chunk의 global chunk index.
                # 쉽게 외우기:
                #   v번째 virtual round의 시작 chunk index = v * pp_size
                #   그 round 안에서 gpu번째 칸으로 이동 = + gpu
                #   따라서 global_chunk = v * pp_size + gpu
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
