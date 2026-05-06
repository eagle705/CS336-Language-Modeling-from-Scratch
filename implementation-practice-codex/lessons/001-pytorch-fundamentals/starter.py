"""PyTorch Deep Learning Fundamentals
=====================================
인터뷰에서 자주 나오는 PyTorch 핵심 개념들.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def autograd_internals():
    """PyTorch autograd의 핵심 동작."""
    raise NotImplementedError('TODO: implement autograd_internals; compare with solution.py only after trying.')

class StraightThroughEstimator(torch.autograd.Function):
    """STE: forward에서 threshold, backward에서 identity.
Quantization 학습에서 사용.

forward: y = 1 if x > 0 else 0  (미분 불가능)
backward: dy/dx = 1              (미분을 identity로 근사)"""

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError('TODO: implement StraightThroughEstimator.forward; compare with solution.py only after trying.')

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError('TODO: implement StraightThroughEstimator.backward; compare with solution.py only after trying.')

def custom_function_demo():
    raise NotImplementedError('TODO: implement custom_function_demo; compare with solution.py only after trying.')

def hooks_demo():
    """Forward/backward hook으로 중간값 모니터링."""
    raise NotImplementedError('TODO: implement hooks_demo; compare with solution.py only after trying.')

def initialization_demo():
    """다양한 초기화 방법과 그 이유."""
    raise NotImplementedError('TODO: implement initialization_demo; compare with solution.py only after trying.')

def lr_schedule_demo():
    """Cosine annealing with warmup (가장 흔한 LR schedule)."""
    raise NotImplementedError('TODO: implement lr_schedule_demo; compare with solution.py only after trying.')
if __name__ == '__main__':
    autograd_internals()
    custom_function_demo()
    hooks_demo()
    initialization_demo()
    lr_schedule_demo()
