"""
Memory Optimization
====================
GPU 메모리 사용을 최적화하는 기법들.

GPU 메모리 구성 (학습 시):
  ┌─────────────────────────────────┐
  │ Model Parameters (weights)      │  → FSDP/ZeRO로 분산
  │ Optimizer States (Adam m, v)    │  → ZeRO-1+로 분산
  │ Gradients                       │  → ZeRO-2+로 분산
  │ Activations (forward 중간값)     │  → 이 파일에서 다루는 핵심!
  │ Temporary buffers               │
  └─────────────────────────────────┘

Activation 메모리가 왜 문제?
  - batch_size, seq_len에 비례하여 증가
  - 각 layer의 중간 결과를 backward까지 보관해야 함
  - 예: 7B model, seq=2048, batch=32 → activation ~50GB
"""

import torch
import torch.nn as nn


# ============================================================
# Part 1: Activation 메모리 계산
# ============================================================

def activation_memory_analysis():
    """Transformer layer별 activation 메모리 상세 분석."""
    print("=" * 60)
    print("Activation Memory Analysis")
    print("=" * 60)

    # GPT-like config
    B = 8        # batch size
    S = 2048     # seq len
    D = 4096     # embed dim
    H = 32       # num heads
    FF = 16384   # FFN hidden dim (4x D)
    L = 32       # num layers
    bytes_per = 2  # BF16

    print(f"\n  Config: B={B}, S={S}, D={D}, H={H}, FF={FF}, L={L}")
    print(f"\n  Per-layer activation breakdown:")

    # Attention block
    # Q, K, V projection 입력: (B, S, D)
    qkv_input = B * S * D * bytes_per
    # Attention scores: (B, H, S, S) → 이게 제일 큼!
    attn_scores = B * H * S * S * bytes_per
    # Attention output: (B, S, D)
    attn_output = B * S * D * bytes_per
    # Softmax output 저장 (dropout mask도)
    softmax_out = B * H * S * S * bytes_per

    print(f"    QKV input:       {qkv_input / 1e6:>8.1f} MB")
    print(f"    Attention scores:{attn_scores / 1e6:>8.1f} MB  ← O(S^2)!")
    print(f"    Softmax output:  {softmax_out / 1e6:>8.1f} MB  ← O(S^2)!")
    print(f"    Attention out:   {attn_output / 1e6:>8.1f} MB")

    # FFN block
    ffn_input = B * S * D * bytes_per
    ffn_hidden = B * S * FF * bytes_per
    print(f"    FFN input:       {ffn_input / 1e6:>8.1f} MB")
    print(f"    FFN hidden:      {ffn_hidden / 1e6:>8.1f} MB")

    per_layer = qkv_input + attn_scores + softmax_out + attn_output + ffn_input + ffn_hidden
    total = per_layer * L
    print(f"    ─────────────────────────────")
    print(f"    Per layer total: {per_layer / 1e6:>8.1f} MB")
    print(f"    All {L} layers:   {total / 1e9:>8.2f} GB")

    # Flash Attention으로 절약
    flash_per_layer = qkv_input + attn_output + ffn_input + ffn_hidden  # scores 제거
    flash_total = flash_per_layer * L
    print(f"\n  Flash Attention 적용 시:")
    print(f"    Per layer: {flash_per_layer / 1e6:.1f} MB (scores O(S^2) 제거)")
    print(f"    All {L} layers: {flash_total / 1e9:.2f} GB")
    print(f"    절약: {(1 - flash_total / total) * 100:.0f}%")


# ============================================================
# Part 2: Gradient Checkpointing (Activation Recomputation)
# ============================================================

class CheckpointedBlock(nn.Module):
    """
    Gradient Checkpointing 원리:

    일반:    forward 시 모든 중간값 저장 → backward에서 사용
    체크포인트: forward 시 중간값 버림 → backward에서 다시 forward 계산

    트레이드오프:
      메모리: O(L) → O(sqrt(L))  (L개 중 sqrt(L)개만 저장)
      연산:   1x → ~1.33x forward  (backward에서 재계산)

    PyTorch API:
      torch.utils.checkpoint.checkpoint(fn, *args)
    """

    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim * 4)
        self.layer2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()

    def _inner(self, x):
        """체크포인트로 감쌀 함수. forward 중간값이 해제됨."""
        return self.layer2(self.act(self.layer1(x)))

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            # 중간값(layer1 output, GELU output) 저장 안 함
            # backward 시 _inner를 다시 실행하여 재계산
            return x + torch.utils.checkpoint.checkpoint(
                self._inner, x, use_reentrant=False
            )
        else:
            return x + self._inner(x)


def demo_gradient_checkpointing():
    """체크포인트 유무에 따른 메모리 비교."""
    print("\n" + "=" * 60)
    print("Gradient Checkpointing Demo")
    print("=" * 60)

    dim = 512
    num_layers = 8
    x = torch.randn(4, 64, dim, requires_grad=True)

    blocks = nn.ModuleList([CheckpointedBlock(dim) for _ in range(num_layers)])

    # --- 일반 forward ---
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    y = x
    for block in blocks:
        y = block(y, use_checkpoint=False)
    loss = y.sum()
    loss.backward()

    # activation 저장 개수 추정
    normal_activations = num_layers * 3  # 각 layer: input, GELU output, layer1 output
    checkpoint_activations = num_layers  # 각 layer: input만 저장 (나머지 재계산)

    print(f"\n  {num_layers} layers × (Linear→GELU→Linear)")
    print(f"  Normal:     ~{normal_activations} tensors saved for backward")
    print(f"  Checkpoint: ~{checkpoint_activations} tensors saved (나머지 재계산)")
    print(f"  메모리 절약: ~{(1 - checkpoint_activations/normal_activations)*100:.0f}%")
    print(f"  연산 증가:   ~33% (각 layer forward 1회 추가)")


# ============================================================
# Part 3: Selective Activation Checkpointing
# ============================================================
#
# 전체 layer를 체크포인트하는 대신, 비싼 activation만 선택적으로 체크포인트.
#
# 예: attention scores (O(S^2))만 재계산, FFN activation은 저장
#
# PyTorch 2.x:
#   from torch.utils.checkpoint import checkpoint
#
#   # 전체 block checkpoint
#   output = checkpoint(block, input, use_reentrant=False)
#
#   # Selective: SAC (Selective Activation Checkpointing)
#   # context manager로 특정 연산만 체크포인트
#   from torch.utils.checkpoint import (
#       create_selective_checkpoint_contexts,
#       CheckpointPolicy,
#   )
#
#   # 큰 activation(attention scores)만 재계산하도록 정책 설정
#   def policy_fn(ctx, op, *args, **kwargs):
#       if op == torch.ops.aten.mm.default:  # matmul은 재계산
#           return CheckpointPolicy.MUST_RECOMPUTE
#       return CheckpointPolicy.MUST_SAVE    # 나머지는 저장


# ============================================================
# Part 4: 기타 메모리 최적화
# ============================================================

def other_optimizations():
    print("\n" + "=" * 60)
    print("Other Memory Optimizations")
    print("=" * 60)

    tips = [
        ("CPU Offloading",
         "사용하지 않는 params/optimizer states를 CPU RAM으로 이동",
         "deepspeed.zero.Init(config={'offload_param': {'device': 'cpu'}})"),

        ("Gradient Accumulation",
         "작은 micro-batch를 여러 번 forward/backward 후 한 번에 step",
         "loss = loss / accum_steps; loss.backward(); if step % accum_steps == 0: optimizer.step()"),

        ("In-place operations",
         "가능한 곳에서 in-place 연산으로 임시 텐서 절약",
         "x.add_(residual) instead of x = x + residual"),

        ("torch.compile",
         "fusion으로 중간 텐서 제거 + 메모리 최적화",
         "model = torch.compile(model)"),

        ("8-bit Optimizer",
         "Adam states를 INT8로 저장 (bitsandbytes)",
         "optim = bnb.optim.Adam8bit(model.parameters())"),
    ]

    for name, desc, code in tips:
        print(f"\n  {name}:")
        print(f"    {desc}")
        print(f"    >>> {code}")


if __name__ == "__main__":
    activation_memory_analysis()
    demo_gradient_checkpointing()
    other_optimizations()
