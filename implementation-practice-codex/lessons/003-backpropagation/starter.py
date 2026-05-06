"""Backpropagation from Scratch (NumPy only)
==========================================
2-layer MLP의 forward/backward를 수동 구현하고,
예시 입력으로 각 단계별 값을 확인.

Network Architecture:

    X (batch, input_dim)
    |
    v
  +--------------------------+
  | Linear Layer 1           |
  | z1 = X @ W1 + b1        |    W1: (input_dim, hidden_dim)
  +--------------------------+    b1: (hidden_dim,)
    |
    v
  +--------------------------+
  | ReLU                     |
  | a1 = max(0, z1)         |
  +--------------------------+
    |
    v
  +--------------------------+
  | Linear Layer 2           |
  | z2 = a1 @ W2 + b2       |    W2: (hidden_dim, output_dim)
  +--------------------------+    b2: (output_dim,)
    |
    v
  y_pred (batch, output_dim)
    |
    v
  +--------------------------+
  | MSE Loss                 |
  | L = mean((y_pred - y)^2) |  <-- scalar
  +--------------------------+

Backward (chain rule, 출력→입력 역순).
코드에서는 스칼라 손실 L에 대한 편미분을 `dL_d<변수명>` 으로 씀 (∂L/∂· 와 동일).

  dL_dz2    = (2/N)(y_pred - y)                 ... Loss에서 시작
      |
      +---> dL_dW2 = a1.T @ dL_dz2              ... W2에 대한 ∂L/∂W2 (학습 대상)
      +---> dL_db2 = sum(dL_dz2, axis=0)        ... b2에 대한 ∂L/∂b2 (학습 대상)
      |
      v
  dL_da1    = dL_dz2 @ W2.T                     ... upstream을 a1 쪽으로 전파
      |
      v
  dL_dz1    = dL_da1 * (z1 > 0)                 ... ReLU 미분 (z1>0이면 1, 아니면 0)
      |
      +---> dL_dW1 = X.T @ dL_dz1               ... W1에 대한 ∂L/∂W1 (학습 대상)
      +---> dL_db1 = sum(dL_dz1, axis=0)        ... b1에 대한 ∂L/∂b1 (학습 대상)

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import numpy as np

def step_by_step_example():
    """작은 예시로 forward → loss → backward → update 전 과정을 추적."""
    raise NotImplementedError('TODO: implement step_by_step_example; compare with solution.py only after trying.')

class ManualMLP:
    """2-layer MLP: Input → Linear → ReLU → Linear → MSE Loss

수식 정리:
  Forward:  z1 = X@W1+b1 → a1 = ReLU(z1) → z2 = a1@W2+b2
  Loss:     L = mean((z2 - y)^2)
  Backward: dL_dz2 = (2/N)(z2-y)
            dL_dW2 = a1.T @ dL_dz2,   dL_db2 = sum(dL_dz2)
            dL_da1 = dL_dz2 @ W2.T
            dL_dz1 = dL_da1 * (z1 > 0)
            dL_dW1 = X.T @ dL_dz1,     dL_db1 = sum(dL_dz1)"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        raise NotImplementedError('TODO: implement ManualMLP.__init__; compare with solution.py only after trying.')

    def forward(self, X):
        raise NotImplementedError('TODO: implement ManualMLP.forward; compare with solution.py only after trying.')

    def mse_loss(self, y_pred, y_true):
        raise NotImplementedError('TODO: implement ManualMLP.mse_loss; compare with solution.py only after trying.')

    def backward(self):
        raise NotImplementedError('TODO: implement ManualMLP.backward; compare with solution.py only after trying.')

    def update(self, lr=0.01):
        raise NotImplementedError('TODO: implement ManualMLP.update; compare with solution.py only after trying.')

def gradient_check():
    """유한 차분법으로 analytic gradient가 맞는지 검증."""
    raise NotImplementedError('TODO: implement gradient_check; compare with solution.py only after trying.')

def train_xor():
    """XOR 문제로 전체 학습 루프 확인."""
    raise NotImplementedError('TODO: implement train_xor; compare with solution.py only after trying.')
if __name__ == '__main__':
    step_by_step_example()
    gradient_check()
    train_xor()
