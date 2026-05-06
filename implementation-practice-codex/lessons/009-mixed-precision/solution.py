"""
Mixed Precision Training
=========================
FP32 / FP16 / BF16을 섞어서 학습 → 속도 2x 향상 + 메모리 절약.

숫자 표현 비교:
  FP32:  1 sign + 8 exp + 23 mantissa = 32 bits  (기본)
  FP16:  1 sign + 5 exp + 10 mantissa = 16 bits  (범위 좁음, overflow 위험)
  BF16:  1 sign + 8 exp +  7 mantissa = 16 bits  (FP32과 같은 범위, 정밀도 낮음)

  FP16 범위: ±65504       → gradient가 이 범위 밖이면 overflow/underflow
  BF16 범위: ±3.4 × 10^38 → FP32와 동일한 범위, overflow 걱정 없음
  → 최근 모델은 대부분 BF16 사용 (loss scaling 불필요)

Mixed Precision의 3가지 규칙:
  1. Forward/Backward는 FP16/BF16으로 (빠른 연산)
  2. Weight master copy는 FP32로 유지 (정밀도 보존)
  3. Loss scaling (FP16만): gradient underflow 방지

    ┌──────────────────────────────────────────────────┐
    │  FP32 master weights                             │
    │       │ copy to FP16                             │
    │       ▼                                          │
    │  FP16 forward → FP16 loss → scale loss           │
    │       │                                          │
    │  FP16 backward (scaled gradients)                │
    │       │ unscale + clip                           │
    │       ▼                                          │
    │  FP32 optimizer step (master weights update)     │
    └──────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn


# ============================================================
# Part 1: 숫자 표현 직접 확인
# ============================================================

def explore_dtypes():
    """각 dtype의 범위와 정밀도를 직접 확인."""
    print("=" * 60)
    print("Floating Point Types")
    print("=" * 60)

    for dtype, name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
        info = torch.finfo(dtype)
        print(f"\n  {name} ({dtype}):")
        print(f"    Range: [{info.min:.2e}, {info.max:.2e}]")
        print(f"    Smallest normal: {info.tiny:.2e}")
        print(f"    Precision (eps): {info.eps:.2e}")
        # mantissa bits: FP32=23, FP16=10, BF16=7
        mantissa = {torch.float32: 23, torch.float16: 10, torch.bfloat16: 7}
        print(f"    Mantissa bits: {mantissa[dtype]}")

    # Overflow 비교
    print(f"\n  Overflow 비교:")
    big = torch.tensor(70000.0)
    print(f"    70000 → FP16: {big.half().item()}")       # inf (overflow!)
    print(f"    70000 → BF16: {big.bfloat16().item()}")   # 69632 (정밀도 손실만)

    # 정밀도 비교
    print(f"\n  정밀도 비교 (1.0 + small delta):")
    for delta in [1e-4, 1e-7, 1e-8]:
        val = 1.0 + delta
        fp32 = torch.tensor(val, dtype=torch.float32).item()
        fp16 = torch.tensor(val, dtype=torch.float16).item()
        bf16 = torch.tensor(val, dtype=torch.bfloat16).item()
        print(f"    1 + {delta}: FP32={fp32:.10f}, FP16={fp16:.10f}, BF16={bf16:.10f}")


# ============================================================
# Part 2: 수동 Mixed Precision Training
# ============================================================

def manual_mixed_precision():
    """AMP 없이 mixed precision을 수동으로 구현."""
    print("\n" + "=" * 60)
    print("Manual Mixed Precision Training")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

    # (1) FP32 master weights 보관
    master_params = [p.clone().detach().float() for p in model.parameters()]

    optimizer = torch.optim.SGD(master_params, lr=0.01)
    loss_scale = 1024.0  # FP16 gradient underflow 방지용 스케일

    X = torch.randn(8, 64)
    target = torch.randint(0, 10, (8,))

    # model을 FP16으로 변환
    model.half()

    for step in range(3):
        # (2) FP32 master → FP16 model에 복사
        for p, mp in zip(model.parameters(), master_params):
            p.data.copy_(mp.data.half())

        # (3) FP16으로 forward
        X_fp16 = X.half()
        logits = model(X_fp16)
        # loss 계산은 FP32로 (수치 안정성)
        loss = nn.functional.cross_entropy(logits.float(), target)

        # (4) Scaled backward: gradient underflow 방지
        #     loss에 큰 수를 곱해서 gradient가 FP16 범위 안에 있게 함
        scaled_loss = loss * loss_scale
        scaled_loss.backward()

        # (5) Unscale + FP32 optimizer step
        for mp, p in zip(master_params, model.parameters()):
            if p.grad is not None:
                # gradient를 FP32로 변환하고 scale 복원
                mp.grad = p.grad.float() / loss_scale

        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        print(f"  Step {step}: loss={loss.item():.4f}")

    print(f"  → FP16 forward + FP32 master weights + loss scaling")


# ============================================================
# Part 3: PyTorch AMP (실전에서 사용하는 방법)
# ============================================================

def pytorch_amp_example():
    """torch.amp를 사용한 mixed precision (실전 코드)."""
    print("\n" + "=" * 60)
    print("PyTorch AMP (Automatic Mixed Precision)")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # GradScaler: loss scaling을 자동으로 관리
    #   - scale 값을 동적으로 조절
    #   - inf/nan gradient 감지 시 step skip + scale 감소
    #   - 정상이면 scale 점진적 증가
    scaler = torch.amp.GradScaler()

    X = torch.randn(8, 64)
    target = torch.randint(0, 10, (8,))

    for step in range(3):
        optimizer.zero_grad()

        # autocast: forward에서 자동으로 FP16/BF16 적용
        #   matmul, conv → FP16 (Tensor Core 활용)
        #   softmax, layernorm, loss → FP32 유지 (수치 안정성)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(X)
            loss = nn.functional.cross_entropy(logits, target)

        # BF16이면 scaler 불필요 (overflow 위험 없음)
        # FP16이면 scaler 사용:
        #   scaler.scale(loss).backward()  # scaled backward
        #   scaler.step(optimizer)          # unscale + step (inf면 skip)
        #   scaler.update()                 # scale 조정

        # BF16 간단 버전:
        loss.backward()
        optimizer.step()

        print(f"  Step {step}: loss={loss.item():.4f}, dtype={logits.dtype}")

    print(f"\n  BF16이면 GradScaler 불필요 → 가장 단순한 코드")
    print(f"  FP16이면 GradScaler로 loss scaling 필수")


# ============================================================
# Part 4: 메모리 절약 분석
# ============================================================

def memory_analysis():
    """Mixed precision의 메모리 절약 효과 계산."""
    print("\n" + "=" * 60)
    print("Memory Analysis")
    print("=" * 60)

    n_params = 1_000_000_000  # 1B params 가정
    bytes_per = {"FP32": 4, "FP16/BF16": 2}

    print(f"\n  1B parameter model:")
    print(f"  {'Component':<30} {'FP32':<12} {'Mixed Precision':<15}")
    print(f"  {'-'*30} {'-'*12} {'-'*15}")

    # Model weights
    print(f"  {'Weights':<30} {'4.0 GB':<12} {'2.0 GB (FP16)':<15}")

    # Optimizer (Adam: m, v per param)
    print(f"  {'Optimizer states (Adam m,v)':<30} {'8.0 GB':<12} {'8.0 GB (FP32)':<15}")

    # Gradients
    print(f"  {'Gradients':<30} {'4.0 GB':<12} {'2.0 GB (FP16)':<15}")

    # Master weights (mixed precision only)
    print(f"  {'Master weights (FP32 copy)':<30} {'N/A':<12} {'4.0 GB':<15}")

    print(f"  {'-'*30} {'-'*12} {'-'*15}")
    print(f"  {'Total':<30} {'16.0 GB':<12} {'16.0 GB':<15}")
    print(f"\n  → Mixed precision은 메모리 총량은 비슷하지만,")
    print(f"    forward/backward의 activation 메모리가 절반 + 연산 속도 2x")


# ============================================================
# Part 5: FP8 Training (Hopper GPU, H100+)
# ============================================================
#
# FP8: 8-bit 부동소수점. H100의 Transformer Engine에서 지원.
#
# 두 가지 FP8 format:
#   E4M3: 1 sign + 4 exp + 3 mantissa → 범위 ±448, 정밀도 높음
#         → forward pass (weights, activations)에 사용
#   E5M2: 1 sign + 5 exp + 2 mantissa → 범위 ±57344, 범위 넓음
#         → backward pass (gradients)에 사용
#
#   비교:
#     FP32:  1+8+23 = 32 bits, 범위 ±3.4e38
#     BF16:  1+8+7  = 16 bits, 범위 ±3.4e38
#     FP16:  1+5+10 = 16 bits, 범위 ±65504
#     E4M3:  1+4+3  =  8 bits, 범위 ±448        ← forward
#     E5M2:  1+5+2  =  8 bits, 범위 ±57344      ← backward
#
# FP8 학습의 핵심: Per-tensor scaling
#   FP8의 범위가 좁으므로, 각 텐서의 값 범위에 맞게 동적으로 scale 조정.
#   scale = max_fp8_value / max(abs(tensor))
#   fp8_tensor = (tensor * scale).to(fp8)
#   → "delayed scaling": 이전 iteration의 max를 사용 (현재 값 미리 알 수 없으므로)
#
# Transformer Engine 사용법:
#   import transformer_engine.pytorch as te
#
#   # nn.Linear 대신 te.Linear 사용
#   layer = te.Linear(1024, 4096, bias=False)
#
#   # FP8 recipe: scaling 전략 설정
#   fp8_recipe = te.recipe.DelayedScaling(
#       fp8_format=te.recipe.Format.HYBRID,  # fwd=E4M3, bwd=E5M2
#       amax_history_len=16,                  # scale 결정에 사용할 history 길이
#       amax_compute_algo="max",              # max or most_recent
#   )
#
#   # FP8 context manager로 감싸기
#   with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#       output = layer(input)
#
# Megatron-Core에서 FP8:
#   # gpt_layer_specs에서 TE spec 사용
#   from megatron.core.models.gpt.gpt_layer_specs import (
#       get_gpt_layer_with_transformer_engine_spec
#   )
#   layer_spec = get_gpt_layer_with_transformer_engine_spec()
#   # → 내부적으로 te.Linear 사용 → FP8 자동 적용

def fp8_info():
    """FP8 format 비교."""
    print("\n" + "=" * 60)
    print("FP8 Formats (H100+)")
    print("=" * 60)

    formats = [
        ("FP32",  32, "1+8+23",  "±3.4e38",    "baseline"),
        ("BF16",  16, "1+8+7",   "±3.4e38",    "학습 표준"),
        ("FP16",  16, "1+5+10",  "±65504",     "loss scaling 필요"),
        ("E4M3",   8, "1+4+3",   "±448",       "FP8 forward"),
        ("E5M2",   8, "1+5+2",   "±57344",     "FP8 backward"),
    ]

    print(f"\n  {'Format':<8} {'Bits':>5} {'Layout':>10} {'Range':>12} {'용도'}")
    print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*12} {'-'*20}")
    for name, bits, layout, range_, use in formats:
        print(f"  {name:<8} {bits:>5} {layout:>10} {range_:>12} {use}")

    print(f"\n  FP8 성능 향상:")
    print(f"    H100 FP8:  ~2x throughput vs BF16 (같은 GPU)")
    print(f"    메모리:    activation 메모리 ~2x 절약 (vs BF16)")
    print(f"    통신:      gradient 크기 ~2x 감소")

    print(f"\n  FP8 학습 요구사항:")
    print(f"    - H100 / H200 / B200 GPU (FP8 Tensor Core)")
    print(f"    - Transformer Engine 라이브러리")
    print(f"    - Per-tensor dynamic scaling (delayed scaling)")


# ============================================================
# Part 6: FP4 / NVFP4 (Blackwell GPU, B200+)
# ============================================================
#
# NVFP4 (= MXFP4, Microscaling FP4):
#   4-bit 부동소수점. B200(Blackwell) GPU에서 하드웨어 지원.
#
# Format:
#   E2M1: 1 sign + 2 exp + 1 mantissa = 4 bits
#   표현 가능한 값: {0, 0.5, 1, 1.5, 2, 3, 4, 6} × {+, -}
#   → 매우 제한적! → block scaling으로 보완
#
# Microscaling (MX) 방식:
#   텐서를 32개 원소의 블록으로 나누고, 블록마다 공유 scale factor (E8M0, 8-bit)
#   effective precision ≈ 각 블록 내에서 FP4 + 블록 단위 FP8 scale
#
#   ┌────────────────────────────────────────┐
#   │ Block (32 elements)                    │
#   │ scale (E8M0, 8-bit): 2^exponent       │
#   │ values: [FP4, FP4, ..., FP4] × scale  │
#   └────────────────────────────────────────┘
#
# NVFP4 학습 패턴:
#   Forward:  weights를 NVFP4로 quantize → FP4 matmul (4x throughput)
#   Backward: FP8 또는 BF16 (gradient는 정밀도 필요)
#   Master weights: FP32 유지
#
# Transformer Engine에서 NVFP4:
#   fp8_recipe = te.recipe.MXFP4Scaling(
#       fp8_format=te.recipe.Format.HYBRID,
#   )
#
#   with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#       output = model(input)
#
# 성능:
#   B200 NVFP4: ~2x throughput vs FP8, ~4x vs BF16
#   주로 inference에 사용, 학습은 아직 실험적

def fp4_info():
    """FP4/NVFP4 정보."""
    print("\n" + "=" * 60)
    print("NVFP4 / Microscaling FP4 (B200+)")
    print("=" * 60)

    print(f"\n  E2M1 (4-bit) 표현 가능한 양수 값:")
    # E2M1: exponent bias = 1
    # values: (-1)^s * 2^(e-1) * (1 + m/2)
    values = []
    for e in range(4):  # 2-bit exponent: 0-3
        for m in range(2):  # 1-bit mantissa: 0-1
            if e == 0:  # subnormal
                val = (m / 2) * 2**0  # = 0 or 0.5
            else:
                val = (1 + m / 2) * 2**(e - 1)
    print(f"    {{0, 0.5, 1, 1.5, 2, 3, 4, 6}} (+ zero, + negatives)")
    print(f"    총 16개 값만 표현 가능 → block scaling 필수!")

    print(f"\n  Microscaling (MX) block 구조:")
    print(f"    32개 FP4 원소 + 1개 E8M0 scale factor")
    print(f"    실제 값 = FP4_value × 2^(scale_exponent)")
    print(f"    overhead: 8 bits / 32 elements = 0.25 bit/element")
    print(f"    → effective ~4.25 bits per element")

    print(f"\n  세대별 throughput 비교 (matmul, 상대적):")
    generations = [
        ("A100", "FP32:1x  BF16:2x   FP8:N/A   FP4:N/A"),
        ("H100", "FP32:1x  BF16:2x   FP8:4x    FP4:N/A"),
        ("B200", "FP32:1x  BF16:2x   FP8:4x    FP4:8x"),
    ]
    for gpu, perf in generations:
        print(f"    {gpu}: {perf}")

    print(f"\n  사용 시나리오:")
    print(f"    FP4 weights + FP8 activations → inference 최적 (메모리 4x 절약)")
    print(f"    FP4 weights + BF16 backward  → 학습 (실험적, 정확도 검증 중)")


if __name__ == "__main__":
    explore_dtypes()
    manual_mixed_precision()
    pytorch_amp_example()
    memory_analysis()
    fp8_info()
    fp4_info()
