"""Tensor Parallelism (TP) for MLP
=================================
MLP의 두 Linear weight를 여러 GPU가 나눠 갖는 예제.

핵심은 딱 두 가지:
1. FC1에서는 hidden 차원을 "나눠서 만들고" (Column Parallel)
2. FC2에서는 나눠 만든 값을 "더해서 최종 출력으로 합친다" (Row Parallel)

핵심 아이디어 (Megatron-LM style):
- FC1: W1의 column(hidden 출력)을 쪼갬.
       각 GPU가 hidden의 일부만 만들기 때문에 forward 통신 없음.
- FC2: W2의 row(hidden 입력)를 쪼갬.
       각 GPU가 최종 출력의 partial 값을 만들고, 마지막에 SUM으로 합침.

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
3. 통신량: O(batch * seq_len * embed_dim) - hidden_dim을 크게 잡아도 이 통신량은 그대로

멀티 GPU 실행 (dist API, DeviceMesh 없음) — 아래 예제는 GPU 8장 단일 노드 가정:
  torchrun --nproc_per_node=8 tensor_parallelism.py dist

CPU 단계별 튜토리얼 (주석·출력 위주, GPU 불필요):
  python tensor_parallelism.py step

노트북에서 dist.init_process_group 한 줄만 실행하면 RANK 미설정으로 실패한다.
  → distributed_tp_example() 전체 실행, 또는 init_dist_env_or_notebook_single_process() 선호출.

TP 예제를 꼭 torchrun으로만 해야 하냐?
- 실제 여러 GPU에서 dist.all_reduce까지 돌리려면 torchrun 같은 런처가 정석.
- 원리만 보려면 simulate_tensor_parallelism() / step_by_step_tensor_parallelism() 로 충분.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.distributed as dist
import torch.nn as nn

class _IdentityFwd_AllreduceGradBwd(torch.autograd.Function):
    """입력 X 쪽에 놓는 op.

forward: x를 그대로 통과시킨다.
backward: 각 rank가 가진 dL/dX 조각을 SUM all_reduce로 합친다.

Megatron 논문의 f 연산자."""

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError('TODO: implement _IdentityFwd_AllreduceGradBwd.forward; compare with solution.py only after trying.')

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError('TODO: implement _IdentityFwd_AllreduceGradBwd.backward; compare with solution.py only after trying.')

class _AllreduceSumFwd_IdentityBwd(torch.autograd.Function):
    """출력 Y 쪽에 놓는 op.

forward: RowParallel이 만든 partial output들을 SUM all_reduce로 합친다.
backward: 들어온 gradient를 그대로 통과시킨다.

Megatron 논문의 g 연산자."""

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError('TODO: implement _AllreduceSumFwd_IdentityBwd.forward; compare with solution.py only after trying.')

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError('TODO: implement _AllreduceSumFwd_IdentityBwd.backward; compare with solution.py only after trying.')

class ColumnParallelLinear(nn.Module):
    """FC1: W1의 column(hidden 출력)을 rank별로 나눠 갖는 Linear.

전체 W1: (embed_dim, hidden_dim)
이 rank: (embed_dim, hidden_dim // tp_size)

각 rank가 서로 다른 hidden 칸을 만들기 때문에 forward 통신이 필요 없다."""

    def __init__(self, in_features, out_features, tp_size, tp_rank):
        raise NotImplementedError('TODO: implement ColumnParallelLinear.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement ColumnParallelLinear.forward; compare with solution.py only after trying.')

class RowParallelLinear(nn.Module):
    """FC2: W2의 row(hidden 입력)를 rank별로 나눠 갖는 Linear.

전체 W2: (hidden_dim, embed_dim)
이 rank: (hidden_dim // tp_size, embed_dim)

각 rank가 최종 출력 shape의 partial 값을 만든다.
이 partial들은 concat이 아니라 SUM으로 합쳐야 전체 출력이 된다."""

    def __init__(self, in_features, out_features, tp_size, tp_rank):
        raise NotImplementedError('TODO: implement RowParallelLinear.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement RowParallelLinear.forward; compare with solution.py only after trying.')

class TensorParallelMLP(nn.Module):
    """수동 TP MLP.

흐름:
  f(입력 grad 합산 예약) → Column FC1 → GELU → Row FC2 → g(partial output 합산)"""

    def __init__(self, embed_dim, hidden_dim, tp_size, tp_rank):
        raise NotImplementedError('TODO: implement TensorParallelMLP.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement TensorParallelMLP.forward; compare with solution.py only after trying.')

def init_dist_env_or_notebook_single_process(backend=None, *, master_addr='127.0.0.1', master_port=None):
    """process group이 아직 없을 때만 초기화한다.

torchrun으로 실행하면 RANK/WORLD_SIZE 등이 이미 들어 있으므로 그대로 init한다.
노트북처럼 RANK가 없으면 world_size=1짜리 작은 그룹을 만들어 코드 경로만 확인한다.
실제 여러 GPU TP는 torchrun이 정석이다.

backend가 None이면 CUDA 있으면 nccl, 없으면 gloo.
master_port가 None이면 비어 있는 포트를 골라 MASTER_PORT에 넣는다(충돌 완화)."""
    raise NotImplementedError('TODO: implement init_dist_env_or_notebook_single_process; compare with solution.py only after trying.')

def distributed_tp_example():
    """dist API로 실제 process group을 만들고 TensorParallelMLP를 실행한다.

이 예제의 shape는 GPU 8장 단일 노드에서 TP=8 (torchrun --nproc_per_node=8)을
가정해 맞춰 두었다. hidden_dim은 world_size로 나누어떨어져야 한다.

실행 (8 GPU 1노드):
  torchrun --nproc_per_node=8 tensor_parallelism.py dist

다른 GPU 개수면 --nproc_per_node와 hidden_dim을 같이 조정 (hidden_dim % tp == 0).

노트북에서 RANK 없이 돌리려면 이 함수 전체를 실행하거나, 최소한
init_dist_env_or_notebook_single_process() 를 먼저 호출한 뒤 나머지 코드를 실행한다."""
    raise NotImplementedError('TODO: implement distributed_tp_example; compare with solution.py only after trying.')

def simulate_tensor_parallelism():
    """CPU에서 TP 수식만 검증한다. 실제 dist 통신은 쓰지 않는다."""
    raise NotImplementedError('TODO: implement simulate_tensor_parallelism; compare with solution.py only after trying.')

def step_by_step_tensor_parallelism():
    """CPU만으로 MLP Tensor Parallel forward를 한 단계씩 따라간다.

실제 GPU 2개를 쓰지는 않는다. 대신 GPU0/GPU1이 가질 텐서를 변수로 나눠 놓고,
all_reduce(SUM)는 partial0 + partial1로 흉내 낸다.

실행:
    python tensor_parallelism.py step"""
    raise NotImplementedError('TODO: implement step_by_step_tensor_parallelism; compare with solution.py only after trying.')
if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode == 'dist':
        distributed_tp_example()
    elif mode == 'step':
        step_by_step_tensor_parallelism()
    else:
        simulate_tensor_parallelism()
