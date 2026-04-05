"""
PyTorch Deep Learning Fundamentals
=====================================
인터뷰에서 자주 나오는 PyTorch 핵심 개념들.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: Autograd 동작 원리
# ============================================================

def autograd_internals():
    """PyTorch autograd의 핵심 동작."""
    print("=" * 60)
    print("Autograd Internals")
    print("=" * 60)

    # (1) Computation graph 구축
    # requires_grad=True인 텐서에 연산하면 자동으로 graph 생성
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3 * x     # y = x^2 + 3x
    z = y.sum()             # scalar로 만들어야 backward 가능

    print(f"\n  x = {x.data.tolist()}")
    print(f"  y = x^2 + 3x = {y.data.tolist()}")
    print(f"  z = sum(y) = {z.item()}")

    # (2) backward: chain rule로 gradient 계산
    z.backward()
    # dz/dx = dy/dx = 2x + 3
    print(f"  dz/dx = 2x + 3 = {x.grad.tolist()} (expected: [7.0, 9.0])")

    # (3) grad_fn: 어떤 연산으로 만들어졌는지 추적
    print(f"\n  z.grad_fn = {z.grad_fn}")          # SumBackward
    print(f"  y.grad_fn = {y.grad_fn}")            # AddBackward

    # (4) detach: graph에서 분리 (gradient 흐름 차단)
    x_detached = x.detach()  # gradient 추적 중단
    print(f"\n  x.requires_grad = {x.requires_grad}")
    print(f"  x.detach().requires_grad = {x_detached.requires_grad}")

    # (5) no_grad: 일시적으로 gradient 추적 끄기 (inference 시)
    with torch.no_grad():
        y_no_grad = x ** 2
        print(f"  torch.no_grad() 내부: y.requires_grad = {y_no_grad.requires_grad}")

    # (6) gradient accumulation: backward() 호출마다 grad가 누적됨
    x = torch.tensor([1.0], requires_grad=True)
    for _ in range(3):
        (x * 2).sum().backward()
    print(f"\n  3번 backward 후 grad = {x.grad.item()} (= 2 × 3 = 6)")
    # → optimizer.zero_grad() 또는 x.grad.zero_() 필요!


# ============================================================
# Part 2: Custom autograd Function
# ============================================================

class StraightThroughEstimator(torch.autograd.Function):
    """
    STE: forward에서 threshold, backward에서 identity.
    Quantization 학습에서 사용.

    forward: y = 1 if x > 0 else 0  (미분 불가능)
    backward: dy/dx = 1              (미분을 identity로 근사)
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient를 그대로 통과
        return grad_output


def custom_function_demo():
    print("\n" + "=" * 60)
    print("Custom autograd.Function (STE)")
    print("=" * 60)

    x = torch.randn(5, requires_grad=True)
    y = StraightThroughEstimator.apply(x)
    y.sum().backward()

    print(f"  x:    {x.data.tolist()}")
    print(f"  y:    {y.data.tolist()} (threshold at 0)")
    print(f"  grad: {x.grad.tolist()} (STE: all 1.0)")


# ============================================================
# Part 3: Hooks (gradient/activation 모니터링)
# ============================================================

def hooks_demo():
    """Forward/backward hook으로 중간값 모니터링."""
    print("\n" + "=" * 60)
    print("Hooks Demo")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    # Forward hook: layer 출력 모니터링
    activations = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    model[0].register_forward_hook(save_activation("linear1"))
    model[1].register_forward_hook(save_activation("relu"))

    # Backward hook: gradient 모니터링
    gradients = {}

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    model[2].register_full_backward_hook(save_gradient("linear2"))

    # Forward + backward
    x = torch.randn(2, 4)
    out = model(x)
    out.sum().backward()

    print(f"\n  Activations captured:")
    for name, act in activations.items():
        print(f"    {name}: shape={act.shape}, mean={act.mean():.4f}")

    print(f"\n  Gradients captured:")
    for name, grad in gradients.items():
        print(f"    {name}: shape={grad.shape}, mean={grad.mean():.4f}")


# ============================================================
# Part 4: torch.compile
# ============================================================
#
# PyTorch 2.x의 핵심 최적화 도구.
#
# model = torch.compile(model)
#
# 동작:
#   1. TorchDynamo: Python bytecode를 분석하여 FX graph 추출
#   2. TorchInductor: FX graph를 최적화하여 Triton kernel 생성
#
# 최적화 종류:
#   - Operator fusion: 여러 연산을 하나의 kernel로 합침
#     예: LayerNorm = mean → sub → var → div → mul → add  →  하나의 fused kernel
#   - Memory planning: 중간 텐서 할당/해제 최적화
#
# 주의:
#   - 첫 호출이 느림 (컴파일 overhead)
#   - Dynamic shapes 지원하지만 recompile 발생 가능
#   - 일부 Python 문법 (data-dependent control flow 등) 지원 안 될 수 있음
#
# 사용법:
#   compiled_model = torch.compile(model, mode="reduce-overhead")
#   # mode options: "default", "reduce-overhead", "max-autotune"


# ============================================================
# Part 5: Weight Initialization
# ============================================================

def initialization_demo():
    """다양한 초기화 방법과 그 이유."""
    print("\n" + "=" * 60)
    print("Weight Initialization")
    print("=" * 60)

    dim = 256

    methods = {
        # Xavier: sigmoid/tanh용. 입출력 분산을 동일하게.
        "Xavier Uniform": lambda w: nn.init.xavier_uniform_(w),
        # Kaiming: ReLU용. ReLU가 절반을 죽이므로 2x 보정.
        "Kaiming Normal": lambda w: nn.init.kaiming_normal_(w, mode='fan_in'),
        # GPT-2 style: 작은 std로 초기화 + residual layer에 1/sqrt(N) 적용
        "GPT-2 (0.02)": lambda w: nn.init.normal_(w, std=0.02),
    }

    print(f"\n  {'Method':<20} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for name, init_fn in methods.items():
        w = torch.empty(dim, dim)
        init_fn(w)
        print(f"  {name:<20} {w.mean():>10.6f} {w.std():>10.6f} {w.abs().max():>10.4f}")

    print(f"\n  GPT-2 residual scaling:")
    print(f"    output_proj의 std = 0.02 / sqrt(2 * num_layers)")
    print(f"    → layer를 쌓을수록 residual 기여를 줄여서 학습 안정화")


# ============================================================
# Part 6: Learning Rate Schedule
# ============================================================

def lr_schedule_demo():
    """Cosine annealing with warmup (가장 흔한 LR schedule)."""
    print("\n" + "=" * 60)
    print("Learning Rate Schedule")
    print("=" * 60)

    total_steps = 1000
    warmup_steps = 100
    max_lr = 3e-4
    min_lr = 3e-5

    def cosine_with_warmup(step):
        if step < warmup_steps:
            # Linear warmup: 0 → max_lr
            return max_lr * step / warmup_steps
        else:
            # Cosine decay: max_lr → min_lr
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    # 샘플 출력
    print(f"\n  Schedule: {warmup_steps} warmup → cosine decay")
    print(f"  Max LR: {max_lr}, Min LR: {min_lr}")
    checkpoints = [0, 50, 100, 250, 500, 750, 1000]
    print(f"\n  {'Step':>6} {'LR':>12}")
    for s in checkpoints:
        s = min(s, total_steps - 1)
        print(f"  {s:>6} {cosine_with_warmup(s):>12.6f}")


if __name__ == "__main__":
    autograd_internals()
    custom_function_demo()
    hooks_demo()
    initialization_demo()
    lr_schedule_demo()
