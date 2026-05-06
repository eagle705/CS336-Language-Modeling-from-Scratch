"""
Tensor Parallelism (TP) for MLP
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
#   모든 rank의 tensor를 더하고, 그 결과를 다시 모든 rank가 갖는다.
#   TP에서는 "각 rank가 만든 partial output/grad를 모두 더할 때" 사용.
#
#   GPU 0: [1, 2]                  GPU 0: [4, 6]
#   GPU 1: [3, 4]  → all_reduce → GPU 1: [4, 6]    (모두 같은 값)
#
# dist.all_gather(output_list, tensor)
#   각 rank의 tensor를 이어 모은다. "조각들을 concat해서 큰 tensor를 만들 때" 사용.
#
#   GPU 0: [1, 2]                    GPU 0: [[1,2], [3,4]]
#   GPU 1: [3, 4]  → all_gather →   GPU 1: [[1,2], [3,4]]
#
# dist.reduce_scatter(output, input_list, op=ReduceOp.SUM)
#   먼저 더하고(reduce), 결과를 다시 rank별로 나눠 준다(scatter).
#   큰 모델에서는 all_reduce보다 메모리를 아끼기 위해 자주 쓰인다.
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
# 문제: dist.all_reduce는 그냥 호출하면 "통신 함수"일 뿐, 원하는 backward 규칙을
# 자동으로 만들어 주지 않는다.
# 해결: torch.autograd.Function에 forward/backward를 직접 써서,
#       "forward에서는 통신, backward에서는 통과" 같은 규칙을 명시한다.
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
# Megatron-LM의 핵심 트릭: f와 g 두 개의 작은 autograd op
#
#   f = _IdentityFwd_AllreduceGradBwd
#       forward: 입력을 그대로 넘김
#       backward: 입력 X의 gradient를 rank끼리 더함
#
#   g = _AllreduceSumFwd_IdentityBwd
#       forward: RowParallel이 만든 partial output을 rank끼리 더함
#       backward: gradient를 그대로 넘김
#
# 결과: MLP forward에서 all_reduce 1번, backward에서 all_reduce 1번만 필요.
#
# --- ASCII 그림: 어디서 값을 더해야 하는지 ---
#
# Forward: RowParallel 뒤에서는 "같은 출력 위치에 대한 partial"을 더해야 한다.
#
#   +---------+     +----------------+     +------+     +-----------------+
#   | X       | --> | Column FC1     | --> | GELU | --> | Row FC2         |
#   | same on |     | hidden 조각 만듦 |     |      |     | partial Y 만듦  |
#   | ranks   |     +----------------+     +------+     +--------+--------+
#   +---------+                                                   |
#                                                                 v
#                                                        +--------+--------+
#                                                        | g: SUM all_reduce|
#                                                        | partial들을 더함 |
#                                                        +--------+--------+
#                                                                 |
#                                                                 v
#                                                        +-----------------+
#                                                        | Y               |
#                                                        | same on ranks   |
#                                                        +-----------------+
#
# 핵심: Row FC2의 partial들은 "이어붙일 조각"이 아니라 "더할 조각"이다.
# 그래서 concat이 아니라 all_reduce(SUM).
#
# Backward: Column FC1을 거쳐 나온 dL/dX 조각들을 더해야 한다.
#
#   +----------------+     +------+     +------------------+
#   | dL/dY          | --> | ...  | --> | Column FC1 bwd   |
#   +----------------+     +------+     | 이 rank의 dL/dX  |
#                                       +---------+--------+
#                                                 |
#                                                 v
#                                       +---------+--------+
#                                       | f: SUM all_reduce|
#                                       | dL/dX 조각 합침  |
#                                       +---------+--------+
#                                                 |
#                                                 v
#                                       +------------------+
#                                       | full dL/dX       |
#                                       | same on ranks    |
#                                       +------------------+
#
# 핵심: X는 모든 rank가 같은 값을 썼다. 따라서 X의 gradient도 모든 shard 경로의
# 기여를 더한 값이어야 한다.
#
# autograd.Function은 SubClass.apply(...)로 호출해야 한다.
# apply가 autograd 그래프 노드를 만들고, 나중에 우리가 정의한 backward를 호출한다.
# forward(...)를 직접 부르면 backward 규칙이 그래프에 등록되지 않는다.

# f: Column Parallel 앞에 배치 (클래스명 = 동작을 그대로 읽기)
class _IdentityFwd_AllreduceGradBwd(torch.autograd.Function):
    """입력 X 쪽에 놓는 op.

    forward: x를 그대로 통과시킨다.
    backward: 각 rank가 가진 dL/dX 조각을 SUM all_reduce로 합친다.

    Megatron 논문의 f 연산자.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


# --- _IdentityFwd_AllreduceGradBwd 없이 더 직접 쓰는 법 (참고용) ---
#
# 목표는 같다:
#   forward에서는 x를 그대로 쓰고,
#   backward에서 x.grad 조각들을 all_reduce(SUM)으로 더한다.
# 이를 하려면 autograd.Function이나 hook처럼 "backward에 끼어드는 방법"이 필요하다.
#
# 예시 1) forward 안에서 작은 Function을 정의하고 바로 apply.
#         클래스가 파일 위에 있느냐, 함수 안에 있느냐만 다르고 원리는 같다.
#
#   def forward(self, x):
#       class _IdentityFwdInline(torch.autograd.Function):
#           @staticmethod
#           def forward(ctx, t):
#               return t
#           @staticmethod
#           def backward(ctx, grad):
#               dist.all_reduce(grad, op=dist.ReduceOp.SUM)
#               return grad
#       x = _IdentityFwdInline.apply(x)
#       ...
#
# 예시 2) 입력 텐서에 hook을 건다.
#         forward는 그대로 두고, backward 때 들어오는 grad만 all_reduce한다.
#
#   def forward(self, x):
#       if dist.is_initialized() and dist.get_world_size() > 1:
#           def _sum_grad_across_ranks(g):
#               dist.all_reduce(g, op=dist.ReduceOp.SUM)
#               return g
#           x.register_hook(_sum_grad_across_ranks)
#       x = self.fc1(x)
#       ...
#
# 예시 2 풀버전) TensorParallelMLP.forward에서 f만 hook으로 바꾼 모습.
#
#   def forward(self, x):
#       # --- f 대체: replicate 입력 X 에 대한 grad 를 rank 간 SUM ---
#       if dist.is_initialized() and dist.get_world_size() > 1:
#           def _sum_grad_across_ranks(g):
#               dist.all_reduce(g, op=dist.ReduceOp.SUM)
#               return g
#           x.register_hook(_sum_grad_across_ranks)
#       # --- 이하 원래 MLP (ColumnParallel → GELU → RowParallel → g) ---
#       x = self.fc1(x)
#       x = self.act(x)
#       x = self.fc2(x)
#       x = _AllreduceSumFwd_IdentityBwd.apply(x)   # g 는 그대로 Function 권장
#       return x
#
# world_size==1이면 더할 다른 rank가 없으므로 hook을 생략해도 결과가 같다.
#
# 단, hook은 여러 번 forward하거나 텐서가 재사용될 때 헷갈릴 수 있다.
# 그래서 실전 코드에서는 지금처럼 별도 Function으로 고정하는 편이 읽기 쉽고 안전하다.
#
# g(_AllreduceSumFwd_IdentityBwd)는 forward에서 통신을 직접 하므로 Function으로 두는 편이 낫다.

# g: Row Parallel 뒤에 배치 (클래스명 = 동작을 그대로 읽기)
class _AllreduceSumFwd_IdentityBwd(torch.autograd.Function):
    """출력 Y 쪽에 놓는 op.

    forward: RowParallel이 만든 partial output들을 SUM all_reduce로 합친다.
    backward: 들어온 gradient를 그대로 통과시킨다.

    Megatron 논문의 g 연산자.
    """

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
    FC1: W1의 column(hidden 출력)을 rank별로 나눠 갖는 Linear.

    전체 W1: (embed_dim, hidden_dim)
    이 rank: (embed_dim, hidden_dim // tp_size)

    각 rank가 서로 다른 hidden 칸을 만들기 때문에 forward 통신이 필요 없다.
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
    FC2: W2의 row(hidden 입력)를 rank별로 나눠 갖는 Linear.

    전체 W2: (hidden_dim, embed_dim)
    이 rank: (hidden_dim // tp_size, embed_dim)

    각 rank가 최종 출력 shape의 partial 값을 만든다.
    이 partial들은 concat이 아니라 SUM으로 합쳐야 전체 출력이 된다.
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
        # 각 rank가 bias 전체를 더하면 all_reduce 후 bias가 tp_size번 들어간다.
        # 그래서 rank마다 bias/tp_size를 더하고, SUM 후 bias가 정확히 한 번만 남게 한다.
        return x @ self.weight + self.bias / self.tp_size


class TensorParallelMLP(nn.Module):
    """수동 TP MLP.

    흐름:
      f(입력 grad 합산 예약) → Column FC1 → GELU → Row FC2 → g(partial output 합산)
    """

    def __init__(self, embed_dim, hidden_dim, tp_size, tp_rank):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, hidden_dim, tp_size, tp_rank)
        self.fc2 = RowParallelLinear(hidden_dim, embed_dim, tp_size, tp_rank)
        self.act = nn.GELU()

    def forward(self, x):
        # .apply(x): autograd.Function의 공식 호출 방식.
        # 이걸 써야 PyTorch가 forward뿐 아니라 우리가 정의한 backward도 기억한다.
        #
        # --- Q1. Row parallel 다음에는 왜 all_reduce? concat 하면 안 되나? ---
        # RowParallel FC2 는 hidden 차원을 rank 별로 나눠서 계산한다.
        # 예를 들어 hidden 을 앞/뒤 절반으로 나누면:
        #
        #   single GPU: Y = [a1_left, a1_right] @ [W2_left; W2_right]
        #              = (a1_left @ W2_left) + (a1_right @ W2_right)
        #
        # 각 rank는 괄호 안의 한 조각(partial)만 만든다. 그런데 그 partial의 shape은
        # 둘 다 최종 출력과 같은 (B, S, E) 이다. 즉 “왼쪽 출력 조각 / 오른쪽 출력 조각”이
        # 아니라, “같은 출력 자리에 더해져야 하는 값”이다.
        #
        # 그래서 concat(이어붙이기)이 아니라 sum(더하기)이 필요하다.
        # 실제 멀티 GPU에서는 rank 들이 가진 partial 을 모두 더해야 하므로 all_reduce(SUM).
        #
        # --- Q2. Column parallel 은 왜 forward가 아니라 backward에서 all_reduce? ---
        # ColumnParallel FC1 은 hidden 출력 자체를 rank 별로 나눠 만든다.
        # rank0 은 a1_left, rank1 은 a1_right 를 만든다고 생각하면 된다.
        # 이 값들은 서로 다른 hidden 칸이므로 forward 에서 더하면 안 된다.
        # 다음 RowParallel FC2 가 각자 자기 hidden 조각을 그대로 받아 쓰면 된다.
        #
        # 하지만 backward 에서는 상황이 다르다. 입력 X 는 모든 rank 에 똑같이 복사되어 있었다.
        # 그래서 X 에 대한 gradient 는 rank0 경로에서 온 몫 + rank1 경로에서 온 몫 + ...
        # 을 더한 값이어야 한다. 각 rank 는 자기 W1 shard 를 거친 gradient 조각만 알고 있으므로,
        # 마지막에 all_reduce(SUM) 으로 dL/dX 조각들을 합친다.
        x = _IdentityFwd_AllreduceGradBwd.apply(x)   # f: forward 통과, backward에서 dL/dX SUM
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = _AllreduceSumFwd_IdentityBwd.apply(x)   # g: forward에서 partial output SUM, backward 통과
        return x


# ============================================================
# Part 4: torch.distributed 직접 사용 (NCCL, DeviceMesh/DTensor 없음)
# ============================================================
#
# TensorParallelMLP는 내부에서 all_reduce를 직접 호출한다.
# DeviceMesh/DTensor 없이 "dist API로 TP를 어떻게 걸 수 있는지" 보는 예제다.
#
# (참고) PyTorch 2.x에서는 init_device_mesh + parallelize_module(ColwiseParallel,
# RowwiseParallel)로 같은 TP를 선언적으로 줄 수 있음 — 내부적으로도 collective에 매핑됨.
#
# --- init_process_group 과 환경 변수 (torchrun vs 노트북) ---
# dist.init_process_group()의 기본 init_method는 "env://" 이다.
# 즉 환경 변수로 "내 rank가 몇 번인지, 총 몇 명인지, 어디서 만날지"를 읽는다.
#
# torchrun이 각 워커 프로세스마다 자동으로 넣어 주는 값:
#   RANK         … 전체 워커 중 이 프로세스의 전역 인덱스 (0 .. WORLD_SIZE-1)
#   LOCAL_RANK   … 이 노드 안에서의 GPU 인덱스 (보통 cuda:LOCAL_RANK 에 매핑)
#   WORLD_SIZE   … 참가 프로세스 총 개수
#   MASTER_ADDR  … 프로세스들이 처음 만날 주소
#   MASTER_PORT  … 프로세스들이 처음 만날 포트
#
# RANK/WORLD_SIZE가 있어야 all_reduce에 누가 참여하는지 알 수 있고,
# MASTER_ADDR/PORT가 있어야 여러 프로세스가 같은 그룹으로 모일 수 있다.
# 노트북에서 dist.init_process_group("nccl") 한 줄만 실행하면 RANK 등이 없어 ValueError.
# 해결: torchrun으로 실행하거나, 아래 헬퍼로 world_size=1 디버그 그룹을 만든다.


def init_dist_env_or_notebook_single_process(
    backend=None,
    *,
    master_addr="127.0.0.1",
    master_port=None,
):
    """
    process group이 아직 없을 때만 초기화한다.

    torchrun으로 실행하면 RANK/WORLD_SIZE 등이 이미 들어 있으므로 그대로 init한다.
    노트북처럼 RANK가 없으면 world_size=1짜리 작은 그룹을 만들어 코드 경로만 확인한다.
    실제 여러 GPU TP는 torchrun이 정석이다.

    backend가 None이면 CUDA 있으면 nccl, 없으면 gloo.
    master_port가 None이면 비어 있는 포트를 골라 MASTER_PORT에 넣는다(충돌 완화).
    """
    import os
    import socket

    if dist.is_initialized():
        return

    if "RANK" not in os.environ:
        # 이 블록은 torchrun 경로가 아니다.
        # torchrun --nproc_per_node=8이면 프로세스가 8개 뜨고,
        # 런처가 각 프로세스에 RANK=0..7, WORLD_SIZE=8을 넣어 준다.
        # 그 경우에는 이미 RANK가 있으므로 여기로 들어오지 않는다.
        #
        # 여기서는 노트북/단일 프로세스에서만 RANK=0, WORLD_SIZE=1을 넣는다.
        # 즉 "나 혼자 있는 process group"이다. 8-way TP 검증용이 아니라 디버그용.
        if master_port is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((master_addr, 0))
            chosen = s.getsockname()[1]
            s.close()
            master_port = str(chosen)
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend)


def distributed_tp_example():
    """
    dist API로 실제 process group을 만들고 TensorParallelMLP를 실행한다.

    이 예제의 shape는 GPU 8장 단일 노드에서 TP=8 (torchrun --nproc_per_node=8)을
    가정해 맞춰 두었다. hidden_dim은 world_size로 나누어떨어져야 한다.

    실행 (8 GPU 1노드):
      torchrun --nproc_per_node=8 tensor_parallelism.py dist

    다른 GPU 개수면 --nproc_per_node와 hidden_dim을 같이 조정 (hidden_dim % tp == 0).

    노트북에서 RANK 없이 돌리려면 이 함수 전체를 실행하거나, 최소한
    init_dist_env_or_notebook_single_process() 를 먼저 호출한 뒤 나머지 코드를 실행한다.
    """
    import os

    init_dist_env_or_notebook_single_process()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    tp_size = world_size
    tp_rank = rank
    # 예제라서 rank마다 다른 shard를 갖도록 seed를 다르게 둔다.
    torch.manual_seed(42 + rank)

    # --- 8-GPU 노드 (TP world = 8) 기준 예시 shape ---
    # hidden_dim=4096이면 8개 rank에서 rank당 hidden 512개를 맡는다.
    batch_size, seq_len, embed_dim, hidden_dim = 4, 32, 512, 4096
    assert hidden_dim % tp_size == 0, "hidden_dim must divide world_size (TP group 크기)"

    model = TensorParallelMLP(embed_dim, hidden_dim, tp_size, tp_rank).to(device)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)

    out = model(x)
    loss = out.sum()
    loss.backward()

    dist.barrier()
    if rank == 0:
        print(f"WORLD_SIZE (TP): {world_size}  (8-GPU 1노드 가정이면 torchrun --nproc_per_node=8)")
        print(f"Input:  {x.shape}")
        print(f"Output: {out.shape}")
        print(f"fc1.weight shard: {model.fc1.weight.shape}")
        print(f"fc2.weight shard: {model.fc2.weight.shape}")
        print("Collectives: all_reduce in _AllreduceSumFwd_IdentityBwd.forward (partial→full)")
        print("               all_reduce in _IdentityFwd_AllreduceGradBwd.backward (grad SUM)")

    dist.destroy_process_group()


# ============================================================
# Part 5: 시뮬레이션 (GPU 없이 TP 수학적 동작 검증)
# ============================================================

def simulate_tensor_parallelism():
    """CPU에서 TP 수식만 검증한다. 실제 dist 통신은 쓰지 않는다."""
    print("=" * 60)
    print("Tensor Parallelism Simulation (no GPUs needed)")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seq_len, embed_dim, hidden_dim = 2, 4, 8, 16
    tp_size = 2

    # 전체 weight. 이걸 아래에서 두 rank가 가진 것처럼 반으로 자른다.
    W1 = torch.randn(embed_dim, hidden_dim)
    b1 = torch.zeros(hidden_dim)
    W2 = torch.randn(hidden_dim, embed_dim)
    b2 = torch.zeros(embed_dim)
    X = torch.randn(batch, seq_len, embed_dim)

    # --- Single GPU ---
    out_single = torch.nn.functional.gelu(X @ W1 + b1) @ W2 + b2

    # --- 2-way TP 시뮬레이션 ---
    half = hidden_dim // 2

    # FC1 column split: hidden 출력의 앞/뒤 절반을 각각 만든다.
    #   W1[:, :half]  →  GPU 0
    #   W1[:, half:]  →  GPU 1
    a1_gpu0 = torch.nn.functional.gelu(X @ W1[:, :half] + b1[:half])
    a1_gpu1 = torch.nn.functional.gelu(X @ W1[:, half:] + b1[half:])

    # FC2 row split: 각 rank가 최종 출력 shape의 partial 값을 만든다.
    #   W2[:half, :]  →  GPU 0
    #   W2[half:, :]  →  GPU 1
    partial_0 = a1_gpu0 @ W2[:half, :] + b2 / 2  # bias를 tp_size로 나눠서 중복 방지
    partial_1 = a1_gpu1 @ W2[half:, :] + b2 / 2

    # 실제 멀티 GPU라면 여기서 all_reduce(SUM). CPU 예제에서는 그냥 더한다.
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
    CPU만으로 MLP Tensor Parallel forward를 한 단계씩 따라간다.

    실제 GPU 2개를 쓰지는 않는다. 대신 GPU0/GPU1이 가질 텐서를 변수로 나눠 놓고,
    all_reduce(SUM)는 partial0 + partial1로 흉내 낸다.

    실행:
        python tensor_parallelism.py step
    """
    # -------------------------------------------------------------------------
    # Step 0 — 단일 GPU MLP가 계산하는 식
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
    # TP의 목표: W1/W2를 나눠 계산해도, 마지막 결과 Y는 단일 GPU 계산과 같아야 한다.
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
    # Step 1 — 먼저 단일 GPU 기준 전체 weight를 만든다
    #
    # 실제 TP에서는 처음부터 rank별 shard만 만들 수 있다.
    # 여기서는 비교가 쉬우도록 전체 weight를 만든 뒤 반으로 자른다.
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
    # Step 2 — Column Parallel (FC1): W1의 column(hidden 출력)을 자른다
    #
    # W1_full을 두 덩어리로 나눈다:
    #   W1_gpu0 = W1_full[:, 0:h_local]      … hidden 인덱스 0..h_local-1
    #   W1_gpu1 = W1_full[:, h_local:H]      … hidden 인덱스 h_local..H-1
    #
    # 두 조각을 가로로 붙이면 원래 W1이 된다.
    # 각 rank는 자기 hidden 칸만 만들기 때문에 여기서는 통신이 없다.
    #
    # z1 = X @ W1 의 전체 shape은 (B,S,H).
    # GPU0은 그중 앞 h_local칸만 계산:
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
    # Step 3 — 각 rank에서 FC1 + GELU (통신 없음)
    #
    # a1_gpu0과 a1_gpu1은 서로 다른 hidden 칸이다.
    # 서로 더하면 안 되고, 그냥 각자 들고 있으면 된다.
    # -------------------------------------------------------------------------
    print("\n[Step 3] 각 GPU에서 z1 = X@W1_shard + b_shard,  a1 = GELU(z1)")
    z1_gpu0 = X @ W1_gpu0 + b1_gpu0
    z1_gpu1 = X @ W1_gpu1 + b1_gpu1
    a1_gpu0 = torch.nn.functional.gelu(z1_gpu0)
    a1_gpu1 = torch.nn.functional.gelu(z1_gpu1)
    print(f"  z1_gpu0.shape = {tuple(z1_gpu0.shape)}  (B, S, h_local)")
    print(f"  a1_gpu0.shape = {tuple(a1_gpu0.shape)}  (GELU는 원소별이라 shape 동일)")

    # -------------------------------------------------------------------------
    # Step 4 — Row Parallel (FC2): W2의 row(hidden 입력)를 자른다
    #
    # 전체 W2 (H, E)를 행 기준으로 반으로 쪼갠다:
    #   W2_gpu0 = W2_full[0:h_local, :]       … 위쪽 h_local 행
    #   W2_gpu1 = W2_full[h_local:H, :]       … 아래쪽 h_local 행
    #
    # 단일 GPU에서는 (B,S,H) @ (H,E) = (B,S,E).
    # 여기서 H축 내적을 GPU0/GPU1이 나눠 계산한다고 보면 된다.
    #
    # GPU0이 가진 a1_gpu0은 원래 a1의 앞 h_local 성분에 해당하고,
    # W2_gpu0은 W2의 앞 h_local 행에 해당하므로,
    #   partial0 = a1_gpu0 @ W2_gpu0   … shape (B, S, E)
    # 는 전체 행렬곱에서 H축 앞 절반만 계산한 partial 값이다.
    # GPU1도 마찬가지로 partial1을 낸다.
    #
    # 단일 GPU와 같은 결과를 얻으려면:
    #   Y = partial0 + partial1   … H축으로 쪼개진 두 내적의 합
    # 이 된다. 그래서 RowParallel 뒤에는 concat이 아니라 sum이 필요하다.
    # -------------------------------------------------------------------------
    print("\n[Step 4] Row Parallel — W2를 행 방향으로 분할")
    W2_gpu0 = W2_full[:h_local, :].contiguous()
    W2_gpu1 = W2_full[h_local:, :].contiguous()
    print(f"  W2_gpu0.shape = {tuple(W2_gpu0.shape)}  ← (H//tp, E)")
    print(f"  W2_gpu1.shape = {tuple(W2_gpu1.shape)}")

    # -------------------------------------------------------------------------
    # Step 5 — bias b2를 rank마다 b2/tp_size로 더하는 이유
    #
    # 단일 GPU: Y = ... + b2  (한 번만 더함)
    # partial0과 partial1은 둘 다 최종 출력 shape (B,S,E)이다.
    # 각 partial에 b2를 그대로 더하면 sum 후 b2가 두 번 들어간다.
    # 그래서 각 rank에 b2/tp_size만 더한다:
    #   (b2/tp) + (b2/tp) + ...  (tp개) = b2
    # -------------------------------------------------------------------------
    print("\n[Step 5] partial_r = a1_gpu_r @ W2_gpu_r + (b2 / tp_size)")
    partial0 = a1_gpu0 @ W2_gpu0 + b2_full / tp_size
    partial1 = a1_gpu1 @ W2_gpu1 + b2_full / tp_size
    print(f"  partial0.shape = {tuple(partial0.shape)}  (B, S, E)")
    print(f"  partial1.shape = {tuple(partial1.shape)}")

    # -------------------------------------------------------------------------
    # Step 6 — All-Reduce(SUM): 같은 shape의 partial들을 더한다
    #
    # 실제 GPU에서는 dist.all_reduce(partial, SUM) 한 번.
    # 여기서는 partial0 + partial1로 같은 일을 한다.
    # 통신량은 출력 크기 B*S*E이고, hidden 크기 H와는 직접 관련이 없다.
    # -------------------------------------------------------------------------
    print("\n[Step 6] All-Reduce(SUM) — 여기서만 GPU 간 합산 (시뮬: partial0 + partial1)")
    Y_tp = partial0 + partial1
    print(f"  Y_tp.shape = {tuple(Y_tp.shape)}")

    # -------------------------------------------------------------------------
    # Step 7 — 단일 GPU forward와 비교
    # -------------------------------------------------------------------------
    print("\n[Step 7] 단일 GPU 레퍼런스와 비교")
    z1_ref = X @ W1_full + b1_full
    a1_ref = torch.nn.functional.gelu(z1_ref)
    Y_ref = a1_ref @ W2_full + b2_full
    max_abs = (Y_tp - Y_ref).abs().max().item()
    print(f"  max |Y_tp - Y_ref| = {max_abs:.2e}")
    print(f"  결과: {'PASSED' if max_abs < 1e-5 else 'FAILED'}")

    # -------------------------------------------------------------------------
    # Step 8 (설명만) — Part 2의 f / g 와 연결
    #
    # TensorParallelMLP.forward:
    #   x = f(x)                 … f = _IdentityFwd_AllreduceGradBwd: forward는 그대로 통과
    #   x = ColParallelLinear(x)
    #   x = GELU(x)
    #   x = RowParallelLinear(x) … 각 rank에서 partial 생성
    #   x = g(x)                 … g = _AllreduceSumFwd_IdentityBwd: forward에서 all_reduce
    #
    # f.backward에서 all_reduce가 필요한 이유:
    #   X는 모든 rank가 같은 값을 썼다. 따라서 dL/dX도 모든 rank 경로의 기여를 더해야 한다.
    #
    # g.backward가 identity인 이유:
    #   g.forward에서 이미 Y를 같은 값으로 맞췄으므로, 뒤로 오는 grad는 그대로 보내면 된다.
    # -------------------------------------------------------------------------
    print("\n[Step 8] Part 2: f(_IdentityFwd_AllreduceGradBwd) / g(_AllreduceSumFwd_IdentityBwd)")
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
