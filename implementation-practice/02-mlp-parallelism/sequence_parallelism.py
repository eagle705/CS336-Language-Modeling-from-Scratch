"""
Sequence Parallelism (SP)
===========================
TP와 결합하여 LayerNorm, Dropout 등의 activation도 시퀀스 차원으로 분산.

문제: TP만 쓰면 LayerNorm, Dropout은 여전히 전체 시퀀스를 각 GPU가 중복 보유.
     activation 메모리의 상당 부분이 이 non-TP 영역에서 발생.

TP만 적용 (SP 없음):
    모든 GPU가 동일한 전체 activation 보유 (중복!)

    GPU 0: [LayerNorm(전체 seq)] → [TP fc1(절반)] → [TP fc2(절반)] → [LayerNorm(전체 seq)]
    GPU 1: [LayerNorm(전체 seq)] → [TP fc1(절반)] → [TP fc2(절반)] → [LayerNorm(전체 seq)]
                  ↑ 중복!                                                    ↑ 중복!

TP + SP (Megatron-LM):
    non-TP 영역은 seq 차원으로 split, TP 영역에서만 전체 seq 복원.

    GPU 0: [LN(seq 앞절반)] → gather → [TP fc1] → [TP fc2] → scatter → [LN(seq 앞절반)]
    GPU 1: [LN(seq 뒷절반)] → gather → [TP fc1] → [TP fc2] → scatter → [LN(seq 뒷절반)]
                                 ↑ all-gather          ↑ reduce-scatter

통신 변화:
    TP only:  all-reduce = reduce-scatter + all-gather (2번)
    TP + SP:  reduce-scatter 1번 + all-gather 1번 (총량 동일! 위치만 다름)
    → 통신 총량은 같지만, activation 메모리가 1/TP로 감소!

    ┌──────────────────────────────────────────────────────────────────┐
    │  TP only (forward):                                             │
    │                                                                 │
    │  [LN] ──→ [ColumnParallel] ──→ [RowParallel] ──all-reduce──→   │
    │   ↑ 전체 seq                                    ↑ 전체 seq      │
    │                                                                 │
    │  TP + SP (forward):                                             │
    │                                                                 │
    │  [LN] ─all-gather─→ [ColParallel] ─→ [RowParallel] ─r-scatter→ │
    │   ↑ seq/TP                                           ↑ seq/TP   │
    │                                                                 │
    │  통신 총량 동일, activation 메모리 1/TP 절약!                     │
    └──────────────────────────────────────────────────────────────────┘

인터뷰 포인트:
  1. SP는 TP의 all-reduce를 (all-gather + reduce-scatter)로 분리
  2. 통신량은 동일하지만, non-TP 영역의 activation이 1/TP로 감소
  3. Megatron-Core에서 sequence_parallel=True 한 줄로 활성화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: 통신 primitive (SP용)
# ============================================================
#
# SP에서 사용하는 통신:
#
# (1) all-gather: seq split → 전체 seq 복원 (ColumnParallel 앞)
#     GPU 0: [seq_0]  ─┐
#     GPU 1: [seq_1]  ─┼─ all-gather → 모든 GPU가 [seq_0, seq_1] 보유
#
# (2) reduce-scatter: 전체 seq의 gradient/output → seq split (RowParallel 뒤)
#     GPU 0: [full_0] ─┐
#     GPU 1: [full_1] ─┼─ reduce-scatter → GPU 0: sum[:half], GPU 1: sum[half:]
#
# 핵심: all-reduce = reduce-scatter + all-gather
#   TP only:  한 곳에서 all-reduce
#   TP + SP:  all-gather를 앞으로, reduce-scatter를 뒤로 분리
#   → 총 통신량 동일!

class _AllGatherFromSP(torch.autograd.Function):
    """
    Forward: all-gather (seq/TP → 전체 seq)
    Backward: reduce-scatter (gradient를 seq/TP로 분산)
    """

    @staticmethod
    def forward(ctx, x, tp_size):
        # 시뮬레이션: seq 차원으로 concat
        ctx.tp_size = tp_size
        # 실제: dist.all_gather → concat
        return x  # 시뮬에서는 이미 전체 seq

    @staticmethod
    def backward(ctx, grad):
        # 실제: dist.reduce_scatter
        return grad, None


class _ReduceScatterToSP(torch.autograd.Function):
    """
    Forward: reduce-scatter (전체 seq → seq/TP)
    Backward: all-gather (seq/TP gradient → 전체 seq gradient)
    """

    @staticmethod
    def forward(ctx, x, tp_size):
        ctx.tp_size = tp_size
        # 실제: dist.reduce_scatter
        return x

    @staticmethod
    def backward(ctx, grad):
        # 실제: dist.all_gather
        return grad, None


# ============================================================
# Part 2: SP 시뮬레이션
# ============================================================

def simulate_sequence_parallelism():
    """TP + SP의 activation 분산을 시뮬레이션."""
    print("=" * 60)
    print("Sequence Parallelism Simulation")
    print("=" * 60)

    torch.manual_seed(42)
    tp_size = 2
    seq_len = 8
    batch = 2
    embed_dim = 16
    ffn_hidden = 32

    # 전체 weight
    W1 = torch.randn(embed_dim, ffn_hidden)
    W2 = torch.randn(ffn_hidden, embed_dim)
    ln_weight = torch.ones(embed_dim)
    ln_bias = torch.zeros(embed_dim)

    x = torch.randn(seq_len, batch, embed_dim)

    # === TP only (SP 없음) ===
    print(f"\n  [TP only] 각 GPU가 전체 seq 보유")

    # LayerNorm: 전체 seq (중복!)
    x_ln = F.layer_norm(x, [embed_dim], ln_weight, ln_bias)
    print(f"    LayerNorm input:  {list(x_ln.shape)} (전체 seq, 모든 GPU 동일)")

    # ColumnParallel: hidden split
    half = ffn_hidden // tp_size
    h_r0 = F.gelu(x_ln @ W1[:, :half])
    h_r1 = F.gelu(x_ln @ W1[:, half:])

    # RowParallel + all-reduce
    out_r0 = h_r0 @ W2[:half, :]
    out_r1 = h_r1 @ W2[half:, :]
    out_tp = out_r0 + out_r1  # all-reduce

    print(f"    FFN hidden (r0):  {list(h_r0.shape)} (hidden/TP)")
    print(f"    FFN output:       {list(out_tp.shape)} (all-reduce 후, 전체 seq)")

    tp_activation_elems = (
        x_ln.numel()       # LN output: 전체 seq (중복)
        + h_r0.numel()     # FFN hidden: hidden/TP
        + out_tp.numel()   # FFN output: 전체 seq (중복)
    )

    # === TP + SP ===
    print(f"\n  [TP + SP] non-TP 영역은 seq split")

    chunk = seq_len // tp_size

    # LayerNorm: seq split (각 GPU가 seq/TP만 보유)
    x_sp_r0 = x[:chunk]   # GPU 0: seq 앞절반
    x_sp_r1 = x[chunk:]   # GPU 1: seq 뒷절반
    x_ln_r0 = F.layer_norm(x_sp_r0, [embed_dim], ln_weight, ln_bias)
    x_ln_r1 = F.layer_norm(x_sp_r1, [embed_dim], ln_weight, ln_bias)
    print(f"    LayerNorm input:  {list(x_ln_r0.shape)} (seq/TP, 각 GPU 다름)")

    # all-gather: seq/TP → 전체 seq (ColumnParallel 진입 전)
    x_ln_full_r0 = torch.cat([x_ln_r0, x_ln_r1], dim=0)  # all-gather 시뮬
    x_ln_full_r1 = torch.cat([x_ln_r0, x_ln_r1], dim=0)
    print(f"    After all-gather: {list(x_ln_full_r0.shape)} (전체 seq)")

    # ColumnParallel + RowParallel (TP와 동일)
    h_sp_r0 = F.gelu(x_ln_full_r0 @ W1[:, :half])
    h_sp_r1 = F.gelu(x_ln_full_r1 @ W1[:, half:])
    partial_r0 = h_sp_r0 @ W2[:half, :]
    partial_r1 = h_sp_r1 @ W2[half:, :]

    # reduce-scatter: 전체 seq → seq/TP (다음 LayerNorm 진입 전)
    full_sum = partial_r0 + partial_r1  # reduce
    out_sp_r0 = full_sum[:chunk]        # scatter: GPU 0은 앞절반
    out_sp_r1 = full_sum[chunk:]        # scatter: GPU 1은 뒷절반
    print(f"    After r-scatter: {list(out_sp_r0.shape)} (seq/TP)")

    sp_activation_elems = (
        x_ln_r0.numel()    # LN output: seq/TP ← 절약!
        + h_sp_r0.numel()  # FFN hidden: hidden/TP (동일)
        + out_sp_r0.numel() # FFN output: seq/TP ← 절약!
    )

    # 검증
    out_sp_full = torch.cat([out_sp_r0, out_sp_r1], dim=0)
    diff = (out_tp - out_sp_full).abs().max().item()

    print(f"\n  검증:")
    print(f"    TP vs TP+SP diff: {diff:.2e}")
    print(f"    Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")

    # 메모리 비교
    print(f"\n  Activation 메모리 비교 (per GPU):")
    print(f"    TP only: {tp_activation_elems:>6} elements")
    print(f"    TP + SP: {sp_activation_elems:>6} elements")
    print(f"    절약:    {(1 - sp_activation_elems / tp_activation_elems) * 100:.0f}%")


# ============================================================
# Part 3: 메모리 절약 분석 (실제 규모)
# ============================================================

def memory_analysis():
    """실제 모델 규모에서 SP의 메모리 절약 효과."""
    print("\n" + "=" * 60)
    print("SP Memory Savings (real scale)")
    print("=" * 60)

    configs = [
        ("7B",  4096, 16384, 32, 8,  2048),
        ("70B", 8192, 28672, 80, 8,  4096),
        ("405B", 16384, 53248, 126, 8, 8192),
    ]

    for name, D, FF, L, TP, S in configs:
        B = 1
        bf16 = 2
        layers_per_gpu = L  # PP=1 가정

        # TP only: non-TP 영역이 전체 seq
        # LayerNorm input/output: B * S * D (전체 seq, 2번: attn + ffn)
        # Dropout mask: B * S * D
        non_tp_per_layer = B * S * D * bf16 * 4  # LN_in, LN_out, dropout × 2

        # TP 영역: hidden/TP
        tp_per_layer = B * S * (FF // TP) * bf16 * 2  # fc1_out, activation

        total_tp_only = (non_tp_per_layer + tp_per_layer) * layers_per_gpu

        # TP + SP: non-TP 영역이 seq/TP
        non_tp_sp = non_tp_per_layer // TP
        total_tp_sp = (non_tp_sp + tp_per_layer) * layers_per_gpu

        savings = (1 - total_tp_sp / total_tp_only) * 100

        print(f"\n  {name} (D={D}, TP={TP}, S={S}):")
        print(f"    TP only activation: {total_tp_only / 1e9:.1f} GB")
        print(f"    TP + SP activation: {total_tp_sp / 1e9:.1f} GB")
        print(f"    절약: {savings:.0f}%")


# ============================================================
# Part 4: Megatron-Core에서 SP 활성화
# ============================================================
#
# config = TransformerConfig(
#     tensor_model_parallel_size=8,
#     sequence_parallel=True,        # ← 이 한 줄!
#     ...
# )
#
# 내부 동작 (Megatron-Core 코드):
#   tensor_parallel/mappings.py:
#     - _AllGatherFromSequenceParallelRegion
#         forward: all-gather (seq/TP → full seq)
#         backward: reduce-scatter
#     - _ReduceScatterToSequenceParallelRegion
#         forward: reduce-scatter (full seq → seq/TP)
#         backward: all-gather
#
#   transformer/transformer_layer.py:
#     if config.sequence_parallel:
#         # LayerNorm 출력: (seq/TP, batch, hidden)
#         # all-gather → ColumnParallel
#         # RowParallel → reduce-scatter
#         # 다음 LayerNorm 입력: (seq/TP, batch, hidden)
#
# 통신 함수 (layers.py):
#   ColumnParallelLinear:
#     if sequence_parallel:
#         input = all_gather_from_sp(input)   # (seq/TP → seq)
#
#   RowParallelLinear:
#     if sequence_parallel:
#         output = reduce_scatter_to_sp(output)  # (seq → seq/TP)


# ============================================================
# Part 5: SP + TP 통신 비교
# ============================================================

def communication_comparison():
    print("\n" + "=" * 60)
    print("Communication: TP only vs TP + SP")
    print("=" * 60)

    print("""
  Per transformer layer (forward):

  ┌──────────────────────────────────────────────────────────────┐
  │  TP only:                                                    │
  │    LN(全seq) → ColParallel → RowParallel → all-reduce → LN  │
  │                                              ↑               │
  │                                    2 × S×B×D elements        │
  │                                    (reduce-scatter + gather) │
  │                                                              │
  │  TP + SP:                                                    │
  │    LN(seq/TP) → all-gather → ColP → RowP → reduce-scatter   │
  │                  ↑ S×B×D                     ↑ S×B×D         │
  │                  (한쪽)                       (한쪽)          │
  │                                                              │
  │  총 통신량: 동일! (2 × S×B×D × (TP-1)/TP)                    │
  │  차이: 통신 위치가 바뀜 → non-TP 영역 activation 1/TP 절약   │
  └──────────────────────────────────────────────────────────────┘

  왜 통신량이 같은가?
    all-reduce = reduce-scatter + all-gather
    TP only:  all-reduce를 한 곳에서 수행
    TP + SP:  all-gather와 reduce-scatter를 분리해서 양쪽에 배치
    → 총 데이터량은 동일, 위치만 재배치!
    """)


if __name__ == "__main__":
    simulate_sequence_parallelism()
    memory_analysis()
    communication_comparison()
