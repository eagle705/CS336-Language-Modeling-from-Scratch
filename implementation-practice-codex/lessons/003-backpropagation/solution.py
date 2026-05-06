"""
Backpropagation from Scratch (NumPy only)
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
"""

import numpy as np


# ============================================================
# 예시 입력으로 한 단계씩 따라가기
# ============================================================

def step_by_step_example():
    """작은 예시로 forward → loss → backward → update 전 과정을 추적."""
    print("=" * 60)
    print("Step-by-step Backprop Example")
    print("=" * 60)

    # --- 예시 입력 (1개 샘플, 2차원) ---
    X = np.array([[1.0, 2.0]])   # (1, 2)
    y = np.array([[1.0]])        # (1, 1)

    # 고정된 weight (재현 가능하도록)
    W1 = np.array([[0.1, 0.3],
                    [0.2, -0.1]])  # (2, 2)
    b1 = np.array([0.0, 0.0])     # (2,)
    W2 = np.array([[0.5],
                    [-0.4]])       # (2, 1)
    b2 = np.array([0.0])          # (1,)

    # ========== FORWARD ==========
    print("\n[Forward Pass]")

    # Layer 1
    z1 = X @ W1 + b1              # (1, 2)
    print(f"  z1 = X @ W1 + b1 = {z1}")
    # [1,2] @ [[0.1,0.3],[0.2,-0.1]] = [0.1+0.4, 0.3-0.2] = [0.5, 0.1]

    a1 = np.maximum(0, z1)        # ReLU
    print(f"  a1 = ReLU(z1)    = {a1}")

    # Layer 2
    z2 = a1 @ W2 + b2             # (1, 1)
    print(f"  z2 = a1 @ W2 + b2 = {z2}")
    # [0.5, 0.1] @ [[0.5],[-0.4]] = [0.25 - 0.04] = [0.21]

    y_pred = z2

    # ========== LOSS ==========
    loss = np.mean((y_pred - y) ** 2)
    print(f"\n[Loss]")
    print(f"  MSE = mean((y_pred - y)^2) = mean(({y_pred[0,0]:.2f} - {y[0,0]:.2f})^2) = {loss:.4f}")

    # ========== BACKWARD ==========
    print(f"\n[Backward Pass]")
    N = y.size  # 전체 원소 수

    # 변수명: dL_dx ≡ ∂L/∂x (스칼라 손실 L을 텐서 x로 편미분한 것과 같은 shape).
    # 짧은 dz 같은 이름은 “무엇에 대한 미분인지”가 드러나지 않아 여기서는 쓰지 않음.

    # MSE Loss를 y_pred로 미분하는 유도 과정:
    #
    #   L = (1/N) * sum_i (y_pred_i - y_i)^2
    #
    # 각 원소 y_pred_i에 대해 편미분:
    #   dL/d(y_pred_i) = (1/N) * 2 * (y_pred_i - y_i)
    #                     ^^^^   ^^^^^^^^^^^^^^^^^^^
    #                     mean   (x^2)' = 2x (chain rule)
    #
    # 벡터로 쓰면:
    #   dL/d(y_pred) = (2/N) * (y_pred - y)
    #
    # 여기서 z2 = y_pred 이므로 ∂L/∂z2 도 동일.
    dL_dz2 = (2.0 / N) * (y_pred - y)
    print(f"  dL_dz2 = ∂L/∂z2 = (2/N)(y_pred - y) = {dL_dz2}")

    # z2 = a1 @ W2 + b2 에서 W2, b2로 각각 미분:
    #
    # ∂L/∂W2: chain rule → ∂L/∂z2 * ∂z2/∂W2
    #   z2 = a1 @ W2 를 원소별로 쓰면: z2[i,j] = sum_k( a1[i,k] * W2[k,j] )
    #   따라서: ∂z2[i,j]/∂W2[k,j] = a1[i,k]
    #
    #   chain rule로 합치면:
    #     ∂L/∂W2[k,j] = sum_i( ∂L/∂z2[i,j] * a1[i,k] )   ... i는 batch 합산
    #                 = sum_i( a1.T[k,i] * dL_dz2[i,j] ) ... a1 transpose
    #                 = (a1.T @ dL_dz2)[k,j]             ... 행렬곱 정의 그 자체
    #
    #   shape: (hidden, batch) @ (batch, output) = (hidden, output) ... W2와 같은 shape
    #
    # ∂L/∂b2: chain rule → ∂L/∂z2 * ∂z2/∂b2
    #   z2 = ... + b2 이므로 ∂z2/∂b2 = 1
    #   batch 차원을 합산: sum(dL_dz2, axis=0)
    #   shape: (output_dim,) ... b2와 같은 shape
    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = np.sum(dL_dz2, axis=0)
    print(f"  dL_dW2 = ∂L/∂W2 = a1.T @ dL_dz2 = {dL_dW2.flatten()}")
    print(f"  dL_db2 = ∂L/∂b2 = sum(dL_dz2)    = {dL_db2}")

    # ∂L/∂a1 = dL_dz2 @ W2.T (upstream gradient)
    dL_da1 = dL_dz2 @ W2.T
    print(f"  dL_da1 = ∂L/∂a1 = dL_dz2 @ W2.T = {dL_da1}")

    # ∂L/∂z1 = ∂L/∂a1 * ReLU'(z1)
    dL_dz1 = dL_da1 * (z1 > 0).astype(float)
    print(f"  dL_dz1 = ∂L/∂z1 = dL_da1 * (z1>0) = {dL_dz1}")

    # ∂L/∂W1 = X.T @ dL_dz1
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0)
    print(f"  dL_dW1 = ∂L/∂W1 = X.T @ dL_dz1 =\n{dL_dW1}")
    print(f"  dL_db1 = ∂L/∂b1 = sum(dL_dz1)  = {dL_db1}")

    # ========== UPDATE ==========
    lr = 0.1
    print(f"\n[SGD Update (lr={lr})]")
    print(f"  W2 before: {W2.flatten()}")
    W2_new = W2 - lr * dL_dW2
    print(f"  W2 after:  {W2_new.flatten()}")

    print(f"  W1 before:\n{W1}")
    W1_new = W1 - lr * dL_dW1
    print(f"  W1 after:\n{W1_new}")


# ============================================================
# MLP 클래스 (학습 루프용)
# ============================================================

class ManualMLP:
    """
    2-layer MLP: Input → Linear → ReLU → Linear → MSE Loss

    수식 정리:
      Forward:  z1 = X@W1+b1 → a1 = ReLU(z1) → z2 = a1@W2+b2
      Loss:     L = mean((z2 - y)^2)
      Backward: dL_dz2 = (2/N)(z2-y)
                dL_dW2 = a1.T @ dL_dz2,   dL_db2 = sum(dL_dz2)
                dL_da1 = dL_dz2 @ W2.T
                dL_dz1 = dL_da1 * (z1 > 0)
                dL_dW1 = X.T @ dL_dz1,     dL_db1 = sum(dL_dz1)
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def mse_loss(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        #  Backward 흐름: Loss → Layer2 → ReLU → Layer1
        #
        #  dL_dz2 ──┬── dL_dW2 (= a1.T @ dL_dz2)   ... ∂L/∂W2
        #           ├── dL_db2 (= sum dL_dz2)     ... ∂L/∂b2
        #           v
        #  dL_da1 = dL_dz2 @ W2.T                   ... ∂L/∂a1
        #           v
        #  dL_dz1 = dL_da1 * ReLU'(z1)              ... ∂L/∂z1
        #           ├── dL_dW1 (= X.T @ dL_dz1)     ... ∂L/∂W1
        #           └── dL_db1 (= sum dL_dz1)      ... ∂L/∂b1
        #
        # 이름 규칙: dL_dx ≡ ∂L/∂x (스칼라 손실 L에 대한 편미분).

        N = self.y_true.size  # 전체 원소 수 (batch * output_dim). np.mean이 N개로 나누므로

        # (1) Loss → z2: MSE 미분. d/dz [mean((z-y)^2)] = (2/N)(z-y)
        dL_dz2 = (2.0 / N) * (self.y_pred - self.y_true)  # (batch, output_dim)

        # (2) z2 = a1 @ W2 + b2 이므로:
        #     ∂L/∂W2 = a1.T @ dL_dz2   ... z2를 W2로 미분하면 a1이 남음
        #     ∂L/∂b2 = sum(dL_dz2)     ... z2를 b2로 미분하면 1이 남고, batch 합산
        self.dL_dW2 = self.a1.T @ dL_dz2  # (hidden, batch) @ (batch, output) = (hidden, output)
        self.dL_db2 = np.sum(dL_dz2, axis=0)  # (output_dim,)

        # (3) z2 = a1 @ W2 + b2 에서 a1 방향으로 gradient 전파:
        #     ∂L/∂a1 = dL_dz2 @ W2.T   ... z2를 a1로 미분하면 W2가 남음
        dL_da1 = dL_dz2 @ self.W2.T  # (batch, hidden_dim)

        # (4) a1 = ReLU(z1) 이므로:
        #     ∂L/∂z1 = ∂L/∂a1 * ReLU'(z1)
        #     ReLU'(z1) = 1 if z1 > 0, else 0  ... 양수면 그대로 통과, 음수면 차단
        dL_dz1 = dL_da1 * (self.z1 > 0).astype(float)  # (batch, hidden_dim)

        # (5) z1 = X @ W1 + b1 이므로: (2번과 동일한 패턴)
        #     ∂L/∂W1 = X.T @ dL_dz1    ... z1을 W1로 미분하면 X가 남음
        #     ∂L/∂b1 = sum(dL_dz1)     ... batch 합산
        self.dL_dW1 = self.X.T @ dL_dz1  # (input, batch) @ (batch, hidden) = (input, hidden)
        self.dL_db1 = np.sum(dL_dz1, axis=0)  # (hidden_dim,)

    def update(self, lr=0.01):
        self.W1 -= lr * self.dL_dW1
        self.b1 -= lr * self.dL_db1
        self.W2 -= lr * self.dL_dW2
        self.b2 -= lr * self.dL_db2


# ============================================================
# Gradient Check (수치 미분으로 검증)
# ============================================================

def gradient_check():
    """유한 차분법으로 analytic gradient가 맞는지 검증."""
    print("\n" + "=" * 60)
    print("Gradient Check (finite difference)")
    print("=" * 60)

    np.random.seed(42)
    model = ManualMLP(3, 4, 2)
    X = np.random.randn(5, 3)
    y = np.random.randn(5, 2)

    # Analytic gradient
    W1_orig = model.W1.copy()
    model.mse_loss(model.forward(X), y)
    model.backward()
    analytic = model.dL_dW1.copy()

    # Numerical gradient
    eps = 1e-5
    numerical = np.zeros_like(analytic)
    for i in range(W1_orig.shape[0]):
        for j in range(W1_orig.shape[1]):
            model.W1 = W1_orig.copy()
            model.W1[i, j] += eps
            loss_p = model.mse_loss(model.forward(X), y)

            model.W1 = W1_orig.copy()
            model.W1[i, j] -= eps
            loss_m = model.mse_loss(model.forward(X), y)

            numerical[i, j] = (loss_p - loss_m) / (2 * eps)

    model.W1 = W1_orig
    diff = np.max(np.abs(analytic - numerical))
    print(f"  Max diff (W1): {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")


# ============================================================
# XOR 학습 데모
# ============================================================

def train_xor():
    """XOR 문제로 전체 학습 루프 확인."""
    print("\n" + "=" * 60)
    print("XOR Training Demo")
    print("=" * 60)

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = ManualMLP(input_dim=2, hidden_dim=8, output_dim=1)

    for epoch in range(1001):
        y_pred = model.forward(X)
        loss = model.mse_loss(y_pred, y)
        model.backward()
        model.update(lr=0.1)
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")

    print(f"\nPredictions:")
    y_pred = model.forward(X)
    for i in range(4):
        print(f"  {X[i]} → {y_pred[i,0]:.4f} (target: {y[i,0]:.0f})")


if __name__ == "__main__":
    step_by_step_example()
    gradient_check()
    train_xor()
