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
"""

import torch
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
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
        return grad


# g: Row Parallel 뒤에 배치
class _ReduceFromParallelRegion(torch.autograd.Function):
    """forward: all-reduce (partial sum 합산) / backward: identity"""

    @staticmethod
    def forward(ctx, x):
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
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
# Part 4: DTensor + DeviceMesh 버전 (PyTorch 2.x native)
# ============================================================
#
# --- DTensor 핵심 API ---
#
# DeviceMesh: GPU들의 논리적 배치를 정의
#   mesh = init_device_mesh("cuda", (tp_size,))
#   mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))
#   tp_mesh = mesh["tp"]   # 특정 차원만 슬라이싱
#
# DTensor: 여러 GPU에 분산된 텐서. placement로 분산 방식 지정.
#   Shard(dim)   : dim 차원으로 쪼개서 각 GPU에 분배
#   Replicate()  : 모든 GPU에 동일 복사본
#   Partial()    : 각 GPU가 partial sum 보유 (all-reduce 필요)
#
# distribute_tensor(tensor, mesh, [Shard(0)])  # 전체 텐서 → 분산
# dtensor.redistribute(mesh, [Replicate()])    # 분산 방식 변경 (통신 발생)
#   Shard → Replicate  = all-gather
#   Partial → Replicate = all-reduce
#   Replicate → Shard  = local slice (통신 없음)
#
# parallelize_module: nn.Module에 TP를 선언적으로 적용
#   ColwiseParallel()  : weight를 output dim(dim=0)으로 shard → Column Parallel
#   RowwiseParallel()  : weight를 input dim(dim=1)으로 shard → Row Parallel
#
# 예시 (이게 전부!):
#   from torch.distributed.tensor.parallel import parallelize_module
#   from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
#
#   mesh = init_device_mesh("cuda", (world_size,))
#   model = parallelize_module(model, mesh, {
#       "fc1": ColwiseParallel(),    # W1을 column split
#       "fc2": RowwiseParallel(),    # W2를 row split + all-reduce
#   })
#   # 끝! forward/backward가 자동으로 TP 통신 포함.

def dtensor_tp_example():
    """
    DTensor로 TP MLP 적용하는 전체 코드.
    실행하려면 torchrun --nproc_per_node=2 tensor_parallelism.py dtensor
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 1) DeviceMesh 생성: GPU들의 논리적 배치
    tp_mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # 2) 일반 MLP 모델 생성 (single-GPU 코드 그대로)
    class MLP(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
            self.act = nn.GELU()

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = MLP(256, 1024).cuda()

    # 3) parallelize_module: 한 줄로 TP 적용!
    #    내부적으로 weight를 DTensor로 변환하고 통신 op 삽입.
    model = parallelize_module(model, tp_mesh, {
        "fc1": ColwiseParallel(),    # W1: (256, 1024) → 각 GPU (256, 512)
        "fc2": RowwiseParallel(),    # W2: (1024, 256) → 각 GPU (512, 256)
    })

    # 4) 사용: 일반 forward/backward와 동일
    x = torch.randn(2, 16, 256, device="cuda")
    out = model(x)
    out.sum().backward()

    if rank == 0:
        print(f"Input:  {x.shape}")
        print(f"Output: {out.shape}")
        print(f"fc1.weight: {model.fc1.weight.shape}")  # DTensor (256, 512) per GPU

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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dtensor":
        dtensor_tp_example()
    else:
        simulate_tensor_parallelism()
