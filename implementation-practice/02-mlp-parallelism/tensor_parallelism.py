"""
Tensor Parallelism (TP) for MLP
=================================
MLP의 weight를 여러 GPU에 column/row 방향으로 분할.

핵심 아이디어 (Megatron-LM style):
- FC1: Column Parallel (각 GPU가 hidden_dim의 일부를 담당)
- FC2: Row Parallel (각 GPU가 input의 일부를 받아 output을 all-reduce)

    [Input X]  ← 모든 GPU에 동일 (Replicate)
        |
    FC1 (Column Parallel): W1을 column 방향으로 split
        |
    [X @ W1_0]  [X @ W1_1]   ← 각 GPU에서 독립 계산 (통신 없음)
        |           |
      GELU        GELU
        |           |
    FC2 (Row Parallel): W2를 row 방향으로 split
        |           |
    [a1_0 @ W2_0] [a1_1 @ W2_1]  ← 각 GPU에서 partial sum 계산
        \\          /
       All-Reduce (sum)           ← 여기서만 통신 1회!
            |
        [Output Y]

인터뷰 포인트:
1. Forward: all-reduce 1회 (FC2 output 합산)
2. Backward: all-reduce 1회 (FC1 input gradient 합산)
3. 통신량: O(batch * seq_len * embed_dim) - hidden_dim과 무관!

멀티 GPU 실행 (dist API, DeviceMesh 없음):
  torchrun --nproc_per_node=2 tensor_parallelism.py dist

CPU 단계별 튜토리얼 (주석·출력 위주, GPU 불필요):
  python tensor_parallelism.py step
"""

import torch
import torch.distributed as dist
import torch.nn as nn


# ============================================================
# Part 1: torch.distributed 핵심 API 정리
# ============================================================
#
# --- 집합 통신 (Collective Communication) ---
#
# dist.all_reduce(tensor, op=ReduceOp.SUM)
#   모든 GPU의 tensor를 합산(SUM)하여 결과를 모든 GPU에 저장.
#   통신 후 모든 GPU가 동일한 값을 가짐.
#
#   GPU 0: [1, 2]                  GPU 0: [4, 6]
#   GPU 1: [3, 4]  → all_reduce → GPU 1: [4, 6]    (모두 같은 값)
#
# dist.all_gather(output_list, tensor)
#   각 GPU의 tensor를 모아서 모든 GPU에 리스트로 전달.
#
#   GPU 0: [1, 2]                    GPU 0: [[1,2], [3,4]]
#   GPU 1: [3, 4]  → all_gather →   GPU 1: [[1,2], [3,4]]
#
# dist.reduce_scatter(output, input_list, op=ReduceOp.SUM)
#   all_reduce + scatter. 합산 후 결과를 쪼개서 각 GPU에 분배.
#
#   GPU 0: [1, 2, 3, 4]                       GPU 0: [4, 6]   (앞 절반의 합)
#   GPU 1: [3, 4, 5, 6]  → reduce_scatter →   GPU 1: [8, 10]  (뒷 절반의 합)
#
# dist.send(tensor, dst) / dist.recv(tensor, src)
#   Point-to-point 통신. Pipeline Parallelism에서 stage 간 데이터 전송에 사용.
#
#
# --- autograd.Function으로 통신을 backward에 연결하는 패턴 ---
#
# 문제: dist.all_reduce 같은 통신은 autograd 그래프에 자동 포함 안 됨.
# 해결: torch.autograd.Function을 상속해서 forward/backward에 통신을 명시.
#
# class MyAllReduce(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         dist.all_reduce(x, op=dist.ReduceOp.SUM)  # forward에서 합산
#         return x
#     @staticmethod
#     def backward(ctx, grad):
#         dist.all_reduce(grad, op=dist.ReduceOp.SUM)  # backward에서도 합산
#         return grad
#
# y = MyAllReduce.apply(x)  # .apply()로 호출 → autograd 그래프에 등록됨


# ============================================================
# Part 2: TP Communication Primitives
# ============================================================
#
# Megatron-LM의 핵심 트릭: f와 g 두 개의 연산자 쌍
#
#   f: forward = identity,    backward = all-reduce
#   g: forward = all-reduce,  backward = identity
#
# MLP에 적용하면:
#   forward:  f(X) → ColParallel → GELU → RowParallel → g(output)
#   backward: f(all-reduce grad) ← ... ← g(identity grad)
#
# → forward/backward 각각 all-reduce 1회만 필요!

# f: Column Parallel 앞에 배치
class _CopyToParallelRegion(torch.autograd.Function):
    """forward: identity (각 GPU가 동일 input 받음) / backward: all-reduce (grad 합산)"""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


# g: Row Parallel 뒤에 배치
class _ReduceFromParallelRegion(torch.autograd.Function):
    """forward: all-reduce (partial sum 합산) / backward: identity"""

    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


# ============================================================
# Part 3: TP MLP (수동 구현 - Megatron-LM style)
# ============================================================

class ColumnParallelLinear(nn.Module):
    """
    FC1: Weight를 column 방향으로 split.

    전체 W1: (embed_dim, hidden_dim)
    이 GPU:  (embed_dim, hidden_dim // tp_size)  ← column slice

    통신 없이 독립 계산 가능.
    """

    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert out_features % tp_size == 0
        self.out_per_rank = out_features // tp_size

        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_per_rank) * (2.0 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank))

    def forward(self, x):
        return x @ self.weight + self.bias


class RowParallelLinear(nn.Module):
    """
    FC2: Weight를 row 방향으로 split.

    전체 W2: (hidden_dim, embed_dim)
    이 GPU:  (hidden_dim // tp_size, embed_dim)  ← row slice

    각 GPU가 partial output 계산 → all-reduce 필요.
    """

    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_per_rank = in_features // tp_size
        self.tp_size = tp_size

        self.weight = nn.Parameter(
            torch.randn(self.in_per_rank, out_features) * (2.0 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # bias는 all-reduce 후 1번만 더해야 하므로 tp_size로 나눔
        return x @ self.weight + self.bias / self.tp_size


class TensorParallelMLP(nn.Module):
    """수동 TP MLP: f → ColParallel → GELU → RowParallel → g"""

    def __init__(self, embed_dim, hidden_dim, tp_size, tp_rank):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, hidden_dim, tp_size, tp_rank)
        self.fc2 = RowParallelLinear(hidden_dim, embed_dim, tp_size, tp_rank)
        self.act = nn.GELU()

    def forward(self, x):
        x = _CopyToParallelRegion.apply(x)       # f: identity fwd, all-reduce bwd
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = _ReduceFromParallelRegion.apply(x)   # g: all-reduce fwd, identity bwd
        return x


# ============================================================
# Part 4: torch.distributed 직접 사용 (NCCL, DeviceMesh/DTensor 없음)
# ============================================================
#
# TensorParallelMLP + Part 2의 autograd.Function이 내부에서
#   torch.distributed.all_reduce(..., op=ReduceOp.SUM)
# 를 호출한다 (forward: Row parallel 끝, backward: replicated input 쪽 grad).
#
# (참고) PyTorch 2.x에서는 init_device_mesh + parallelize_module(ColwiseParallel,
# RowwiseParallel)로 같은 TP를 선언적으로 줄 수 있음 — 내부적으로도 collective에 매핑됨.


def distributed_tp_example():
    """
    dist.init_process_group + TensorParallelMLP (명시적 all_reduce 경로).

    실행:
      torchrun --nproc_per_node=<TP> tensor_parallelism.py dist

    환경 변수 LOCAL_RANK가 있으면 해당 CUDA 디바이스 사용 (torchrun 표준).
    """
    import os

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)

    tp_size = world_size
    tp_rank = rank
    # 각 rank가 자신의 column/row 샤드를 독립 초기화 (TP 관례)
    torch.manual_seed(42 + rank)

    embed_dim, hidden_dim = 256, 1024
    assert hidden_dim % tp_size == 0, "hidden_dim must divide world_size"

    model = TensorParallelMLP(embed_dim, hidden_dim, tp_size, tp_rank).cuda()
    x = torch.randn(2, 16, embed_dim, device="cuda", requires_grad=True)

    out = model(x)
    loss = out.sum()
    loss.backward()

    dist.barrier()
    if rank == 0:
        print(f"Input:  {x.shape}")
        print(f"Output: {out.shape}")
        print(f"fc1.weight shard: {model.fc1.weight.shape}")
        print(f"fc2.weight shard: {model.fc2.weight.shape}")
        print("Collectives: torch.distributed.all_reduce in _ReduceFromParallelRegion (fwd)")
        print("               torch.distributed.all_reduce in _CopyToParallelRegion (bwd)")

    dist.destroy_process_group()


# ============================================================
# Part 5: 시뮬레이션 (GPU 없이 TP 수학적 동작 검증)
# ============================================================

def simulate_tensor_parallelism():
    """W를 split해서 각각 계산 후 합치면 원래 결과와 동일함을 검증."""
    print("=" * 60)
    print("Tensor Parallelism Simulation (no GPUs needed)")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seq_len, embed_dim, hidden_dim = 2, 4, 8, 16
    tp_size = 2

    # 원본 weight
    W1 = torch.randn(embed_dim, hidden_dim)
    b1 = torch.zeros(hidden_dim)
    W2 = torch.randn(hidden_dim, embed_dim)
    b2 = torch.zeros(embed_dim)
    X = torch.randn(batch, seq_len, embed_dim)

    # --- Single GPU ---
    out_single = torch.nn.functional.gelu(X @ W1 + b1) @ W2 + b2

    # --- 2-way TP 시뮬레이션 ---
    half = hidden_dim // 2

    # FC1 column split: 각 GPU가 hidden의 절반 담당
    #   W1[:, :half]  →  GPU 0
    #   W1[:, half:]  →  GPU 1
    a1_gpu0 = torch.nn.functional.gelu(X @ W1[:, :half] + b1[:half])
    a1_gpu1 = torch.nn.functional.gelu(X @ W1[:, half:] + b1[half:])

    # FC2 row split: 각 GPU가 partial output 계산
    #   W2[:half, :]  →  GPU 0
    #   W2[half:, :]  →  GPU 1
    partial_0 = a1_gpu0 @ W2[:half, :] + b2 / 2  # bias를 tp_size로 나눠서 중복 방지
    partial_1 = a1_gpu1 @ W2[half:, :] + b2 / 2

    # All-reduce (sum): 이 시점에서만 GPU간 통신!
    out_tp = partial_0 + partial_1

    diff = (out_single - out_tp).abs().max().item()
    print(f"  Single GPU: {out_single.shape}")
    print(f"  TP output:  {out_tp.shape}")
    print(f"  Max diff:   {diff:.2e}")
    print(f"  Result:     {'PASSED' if diff < 1e-5 else 'FAILED'}")
    print(f"\n  Communication: all-reduce {batch * seq_len * embed_dim} elements")
    print(f"  (= batch * seq * embed, hidden_dim과 무관!)")


def step_by_step_tensor_parallelism():
    """
    CPU만으로 Megatron-style MLP TP의 forward를 한 단계씩 따라간다.

    이 함수는 dist를 쓰지 않는다. 대신 “GPU 0이 가질 텐서 / GPU 1이 가질 텐서”를
    같은 프로세스 안의 변수로 나란히 두고, all-reduce는 수학적으로 partial_0 + partial_1
    한 줄로 대체해 동작을 검증한다.

    실행:
        python tensor_parallelism.py step
    """
    # -------------------------------------------------------------------------
    # Step 0 — 문제 설정: 단일 GPU MLP 한 블록이 무엇을 계산하는지
    #
    #   z1 = X @ W1 + b1          … (1) 첫 번째 선형층
    #   a1 = GELU(z1)             … (2) 비선형
    #   Y  = a1 @ W2 + b2         … (3) 두 번째 선형층 → 최종 출력
    #
    # X: (B, S, E)  배치 B, 시퀀스 S, 임베딩 E
    # W1: (E, H)    hidden H로 확장
    # b1: (H,)
    # W2: (H, E)    다시 E로 되돌림
    # b2: (E,)
    #
    # TP의 목표: W1·W2를 쪼개서 두 GPU가 나눠 갖되, 최종 Y는 “통신 후” 단일 GPU와
    # 동일해야 한다. 여기서는 tp_size=2 한 가지만 다룬다.
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step-by-step Tensor Parallel MLP (CPU, tp_size=2)")
    print("=" * 70)

    torch.manual_seed(0)
    B, S, E, H = 1, 2, 4, 6  # 작은 숫자로 shape 추적이 쉽게
    tp_size = 2
    assert H % tp_size == 0, "hidden은 tp_size로 나누어떨어져야 column/row split이 됨"
    h_local = H // tp_size  # 각 GPU가 담당하는 hidden chunk 크기 (= 3)

    # -------------------------------------------------------------------------
    # Step 1 — 가상의 “전체(레플리카)” 가중치를 한번에 만든다
    #
    # 실제 멀티 GPU 학습에서는 보통 rank0에서 초기화 후 scatter하거나,
    # 각 rank가 동일 시드로 shard만 다르게 그리는 방식이 있다. 여기서는 검증이 목적이므로
    # 먼저 전체 W1, W2를 만들고, 아래 Step에서 잘라서 GPU0/GPU1 변수에 넣는다.
    # -------------------------------------------------------------------------
    print("\n[Step 1] 전체 weight / bias / 입력 X 생성 (단일 GPU 기준 텐서)")
    W1_full = torch.randn(E, H)
    b1_full = torch.zeros(H)
    W2_full = torch.randn(H, E)
    b2_full = torch.zeros(E)
    X = torch.randn(B, S, E)
    print(f"  X.shape       = {tuple(X.shape)}   (B, S, E)")
    print(f"  W1_full.shape = {tuple(W1_full.shape)} (E, H)")
    print(f"  W2_full.shape = {tuple(W2_full.shape)} (H, E)")

    # -------------------------------------------------------------------------
    # Step 2 — Column Parallel (FC1): W1을 “열(column)” 방향으로 자른다
    #
    # W1_full을 두 덩어리로 나눈다:
    #   W1_gpu0 = W1_full[:, 0:h_local]      … hidden 인덱스 0..h_local-1
    #   W1_gpu1 = W1_full[:, h_local:H]      … hidden 인덱스 h_local..H-1
    #
    # 중요: 두 조각의 “열 개수” 합이 H이므로, 논리적으로는 W1_full = [W1_gpu0 | W1_gpu1]
    # (가로로 붙인 행렬)과 같다. 각 GPU는 자기 열만 저장하므로 FC1 통신은 필요 없다.
    #
    # z1 = X @ W1 에서 (B,S,E) @ (E,H) = (B,S,H) 인데, GPU0은 H 중 앞 h_local열만 계산:
    #   z1_gpu0 = X @ W1_gpu0 + b1_gpu0   … shape (B, S, h_local)
    # 마찬가지로 GPU1은 뒤쪽 열.
    # -------------------------------------------------------------------------
    print("\n[Step 2] Column Parallel — W1, b1을 hidden 축으로 분할 (GPU0 / GPU1)")
    W1_gpu0 = W1_full[:, :h_local].contiguous()
    W1_gpu1 = W1_full[:, h_local:].contiguous()
    b1_gpu0 = b1_full[:h_local]
    b1_gpu1 = b1_full[h_local:]
    print(f"  W1_gpu0.shape = {tuple(W1_gpu0.shape)}  ← (E, H//tp)")
    print(f"  W1_gpu1.shape = {tuple(W1_gpu1.shape)}")
    print("  해석: 두 GPU가 서로 다른 hidden 부공간만 담당 (같은 X를 각자 곱함).")

    # -------------------------------------------------------------------------
    # Step 3 — 각 GPU에서 FC1 + GELU (통신 없음)
    #
    # Forward에서 이 단계까지는 GPU끼리 데이터를 주고받을 필요가 없다.
    # 이유: z1_gpu0과 z1_gpu1은 서로 다른 출력 채널(부분 hidden)을 내고 있을 뿐,
    # 아직 “합쳐야 하는” 축이 출력 Y에 직접 닿지 않았기 때문이다.
    # -------------------------------------------------------------------------
    print("\n[Step 3] 각 GPU에서 z1 = X@W1_shard + b_shard,  a1 = GELU(z1)")
    z1_gpu0 = X @ W1_gpu0 + b1_gpu0
    z1_gpu1 = X @ W1_gpu1 + b1_gpu1
    a1_gpu0 = torch.nn.functional.gelu(z1_gpu0)
    a1_gpu1 = torch.nn.functional.gelu(z1_gpu1)
    print(f"  z1_gpu0.shape = {tuple(z1_gpu0.shape)}  (B, S, h_local)")
    print(f"  a1_gpu0.shape = {tuple(a1_gpu0.shape)}  (GELU는 원소별이라 shape 동일)")

    # -------------------------------------------------------------------------
    # Step 4 — Row Parallel (FC2): W2를 “행(row)” 방향으로 자른다
    #
    # 전체 W2 (H, E)를 행 기준으로 반으로 쪼갠다:
    #   W2_gpu0 = W2_full[0:h_local, :]       … 위쪽 h_local 행
    #   W2_gpu1 = W2_full[h_local:H, :]       … 아래쪽 h_local 행
    #
    # 단일 GPU에서 Y = a1 @ W2 + b2 를 쓰면 (B,S,H) @ (H,E) = (B,S,E) 이다.
    # 행렬곱의 “약속된” 축은 H: a1의 마지막 축과 W2의 첫 축이 내적된다.
    #
    # GPU0이 가진 a1_gpu0은 원래 a1의 앞 h_local 성분에 해당하고,
    # W2_gpu0은 W2의 앞 h_local 행에 해당하므로,
    #   partial0 = a1_gpu0 @ W2_gpu0   … shape (B, S, E)
    # 는 전체 행렬곱에서 “H축의 앞 절반”만 내적에 참여한 부분합(partial sum)이다.
    # GPU1도 마찬가지로 partial1을 낸다.
    #
    # 따라서 (단일 GPU와 동일한 수학을 유지하려면)
    #   Y = partial0 + partial1   … H축으로 쪼개진 두 내적의 합
    # 이 된다 (행렬곱의 분배법칙).
    # -------------------------------------------------------------------------
    print("\n[Step 4] Row Parallel — W2를 행 방향으로 분할")
    W2_gpu0 = W2_full[:h_local, :].contiguous()
    W2_gpu1 = W2_full[h_local:, :].contiguous()
    print(f"  W2_gpu0.shape = {tuple(W2_gpu0.shape)}  ← (H//tp, E)")
    print(f"  W2_gpu1.shape = {tuple(W2_gpu1.shape)}")

    # -------------------------------------------------------------------------
    # Step 5 — bias b2를 tp_size로 나눠 각 partial에 더하는 이유
    #
    # 단일 GPU: Y = ... + b2  (한 번만 더함)
    # TP에서 partial0, partial1 둘 다 (B,S,E) 전체 shape을 가지므로,
    # 만약 각 partial에 b2를 그대로 더하면 all-reduce(SUM) 후 b2가 tp_size번 더해진다.
    # 그래서 RowParallelLinear에서는 forward에 (b2 / tp_size)를 각 rank에 더하고,
    # 이후 SUM all-reduce로 합치면 b2가 정확히 한 번만 반영된다:
    #   (b2/tp) + (b2/tp) + ...  (tp개) = b2
    # -------------------------------------------------------------------------
    print("\n[Step 5] partial_r = a1_gpu_r @ W2_gpu_r + (b2 / tp_size)")
    partial0 = a1_gpu0 @ W2_gpu0 + b2_full / tp_size
    partial1 = a1_gpu1 @ W2_gpu1 + b2_full / tp_size
    print(f"  partial0.shape = {tuple(partial0.shape)}  (B, S, E)")
    print(f"  partial1.shape = {tuple(partial1.shape)}")

    # -------------------------------------------------------------------------
    # Step 6 — All-Reduce(SUM) = “같은 shape 텐서를 모든 rank에서 더해 동기화”
    #
    # NCCL 등으로는 dist.all_reduce(partial, op=SUM) 한 번이지만,
    # CPU 시뮬레이션에서는 두 partial을 더한 것이 전 rank가 받는 최종 텐서와 같다.
    # 통신량: 원소 개수는 B*S*E (= 출력 크기). H(히든) 크기와 무관한 이유가 여기에 있다.
    # -------------------------------------------------------------------------
    print("\n[Step 6] All-Reduce(SUM) — 여기서만 GPU 간 합산 (시뮬: partial0 + partial1)")
    Y_tp = partial0 + partial1
    print(f"  Y_tp.shape = {tuple(Y_tp.shape)}")

    # -------------------------------------------------------------------------
    # Step 7 — 단일 GPU forward와 비교 (수식이 같으면 오차는 부동소수 한계뿐)
    # -------------------------------------------------------------------------
    print("\n[Step 7] 단일 GPU 레퍼런스와 비교")
    z1_ref = X @ W1_full + b1_full
    a1_ref = torch.nn.functional.gelu(z1_ref)
    Y_ref = a1_ref @ W2_full + b2_full
    max_abs = (Y_tp - Y_ref).abs().max().item()
    print(f"  max |Y_tp - Y_ref| = {max_abs:.2e}")
    print(f"  결과: {'PASSED' if max_abs < 1e-5 else 'FAILED'}")

    # -------------------------------------------------------------------------
    # Step 8 (설명만) — Part 2의 f / g 와 이 시뮬레이션의 대응
    #
    # TensorParallelMLP.forward:
    #   x = f(x)                 … f = _CopyToParallelRegion: forward는 그대로 통과
    #   x = ColParallelLinear(x)
    #   x = GELU(x)
    #   x = RowParallelLinear(x) … 각 rank에서 partial 생성
    #   x = g(x)                 … g = _ReduceFromParallelRegion: forward에서 all_reduce
    #
    # f의 backward에서 all_reduce가 한 번 더 필요한 이유(요약):
    #   입력 X는 replicate 상태인데, 두 GPU의 FC1이 각각 다른 W1 shard로 미분되면
    #   X에 대한 gradient가 rank마다 “자기 shard에 해당하는 부분”만 기여한다.
    #   전체 입력 X는 동일해야 하므로 grad를 SUM all-reduce로 맞춘다.
    #
    # g의 backward는 identity인 이유(요약):
    #   forward에서 이미 합쳐진(replicated) 텐서를 넘겼기 때문에, backward에서는
    #   upstream gradient가 모든 rank에 동일하게 전달되면 된다.
    # -------------------------------------------------------------------------
    print("\n[Step 8] Part 2의 f(_CopyToParallelRegion) / g(_ReduceFromParallelRegion)와 대응")
    print("  forward:  f(입력 복제 표시) → Col FC1 → GELU → Row FC2 → g(all_reduce 합산)")
    print("  backward: f에서 grad 합산(all_reduce) ← … ← g는 grad 그대로 통과")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode == "dist":
        distributed_tp_example()
    elif mode == "step":
        step_by_step_tensor_parallelism()
    else:
        simulate_tensor_parallelism()
