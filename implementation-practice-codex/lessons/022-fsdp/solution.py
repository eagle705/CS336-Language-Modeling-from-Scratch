"""
FSDP (Fully Sharded Data Parallel)
=====================================
PyTorch 네이티브 ZeRO-3 구현. 모든 모델 상태를 GPU에 분산.

FSDP vs DDP:
  DDP:  각 GPU가 전체 모델 복사본 보유 + gradient all-reduce
  FSDP: 모델을 shard로 쪼개서 분산 + 필요할 때만 all-gather

FSDP 동작 (각 FSDP unit = 보통 1개 Transformer block):
  ┌──────────────────────────────────────────────────────┐
  │ Forward:                                             │
  │   all-gather params → forward 계산 → params 해제     │
  │                                                      │
  │ Backward:                                            │
  │   all-gather params → backward 계산 →                │
  │   reduce-scatter grads → params 해제                 │
  │                                                      │
  │ Optimizer step:                                      │
  │   각 GPU가 자기 shard만 update (local operation)      │
  └──────────────────────────────────────────────────────┘

FSDP1 vs FSDP2:
  FSDP1 (torch.distributed.fsdp.FullyShardedDataParallel):
    - FlatParameter: 여러 params를 하나로 flatten → 통신 효율적
    - 단점: flatten 때문에 디버깅 어렵고 유연성 부족

  FSDP2 (torch.distributed.fsdp.fully_shard, PyTorch 2.x):
    - DTensor 기반: 각 param이 독립적인 DTensor
    - per-parameter sharding → 더 유연하고 디버깅 쉬움
    - DeviceMesh와 자연스럽게 통합

Megatron-FSDP는 개념상 무엇이 다른가?
  - PyTorch FSDP/FSDP2는 범용 PyTorch module을 shard하는 일반-purpose API.
  - Megatron-FSDP는 Megatron-Core의 TP/PP/CP/EP와 함께 쓰도록 만든 training-stack 통합 FSDP.
  - 핵심 차이는 "무엇을 shard하느냐"보다 "기존 Megatron parallel dimensions와 어떻게 compose하느냐"다.
  - 예: TP로 이미 쪼개진 Linear weight를 DP-shard 차원에서 다시 FSDP shard하고,
        TransformerLayer 같은 Megatron module boundary를 FSDP unit으로 삼는다.
  - overlap_grad_reduce, overlap_param_gather 같은 통신 overlap과 distributed optimizer 경로가
    Megatron training loop와 맞물리도록 설계되어 있다.
  - 따라서 단순 사용성은 PyTorch FSDP가 쉽고, Megatron 대규모 학습에서는 Megatron-FSDP가
    TP/PP/CP/EP와의 조합, overlap, checkpoint/optimizer 통합 면에서 더 목적 특화되어 있다.
"""

import os
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: FSDP1 사용법 (기존 API)
# ============================================================
#
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
#
# --- ShardingStrategy ---
# FULL_SHARD:     params + grads + opt states 모두 분산 (= ZeRO-3)
# SHARD_GRAD_OP:  grads + opt states만 분산 (= ZeRO-2)
# NO_SHARD:       분산 안 함 (= DDP)
# HYBRID_SHARD:   노드 내 FULL_SHARD + 노드 간 replicate
#
# --- 기본 사용법 ---
#
# model = MyModel()
#
# # 각 Transformer block을 별도 FSDP unit으로 wrap
# for i, block in enumerate(model.blocks):
#     model.blocks[i] = FSDP(block)
#
# # 전체 모델도 FSDP wrap
# model = FSDP(
#     model,
#     sharding_strategy=ShardingStrategy.FULL_SHARD,
#     mixed_precision=MixedPrecision(
#         param_dtype=torch.bfloat16,    # forward에서 params를 BF16으로
#         reduce_dtype=torch.float32,     # gradient reduce는 FP32로
#     ),
# )
#
# # 학습 루프는 일반 PyTorch와 동일
# output = model(input_ids)
# loss = loss_fn(output, targets)
# loss.backward()
# optimizer.step()


# ============================================================
# Part 2: FSDP2 사용법 (fully_shard, PyTorch 2.x)
# ============================================================
#
# from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
# from torch.distributed.device_mesh import init_device_mesh
#
# mesh = init_device_mesh("cuda", (world_size,))
#
# # FSDP2: fully_shard()로 선언적 적용
# # 내부적으로 각 param을 DTensor(Shard(0))로 변환
# mp_policy = MixedPrecisionPolicy(
#     param_dtype=torch.bfloat16,
#     reduce_dtype=torch.float32,
# )
#
# for block in model.blocks:
#     fully_shard(block, mesh=mesh, mp_policy=mp_policy)
# fully_shard(model, mesh=mesh, mp_policy=mp_policy)
#
# # 학습 루프 동일
# output = model(input_ids)
# loss.backward()
# optimizer.step()
#
# --- FSDP1 vs FSDP2 차이 ---
# FSDP1: model을 FSDP()로 wrap → FlatParameter로 변환
# FSDP2: fully_shard()로 적용 → 각 param이 DTensor, 원래 구조 유지


# ============================================================
# Part 3: FSDP 동작 시뮬레이션
# ============================================================

def simulate_fsdp():
    """GPU 없이 FSDP의 shard/gather 동작을 시뮬레이션."""
    print("=" * 60)
    print("FSDP Simulation (4 GPUs)")
    print("=" * 60)

    num_gpus = 4
    num_layers = 2

    # 각 layer의 params (간소화)
    layer_params = [torch.randn(8) for _ in range(num_layers)]  # 8 params per layer

    print(f"\n  전체 모델: {num_layers} layers × 8 params = {num_layers * 8} params")
    print(f"  GPU 수: {num_gpus}")

    # --- Shard: 각 GPU가 전체 params의 1/N만 보관 ---
    shards = {gpu: {} for gpu in range(num_gpus)}
    params_per_gpu = 8 // num_gpus  # 2 params per GPU per layer

    for layer_idx, params in enumerate(layer_params):
        for gpu_id in range(num_gpus):
            start = gpu_id * params_per_gpu
            end = start + params_per_gpu
            shards[gpu_id][layer_idx] = params[start:end].clone()

    print(f"\n  초기 상태 (각 GPU가 보관하는 shard):")
    for gpu_id in range(num_gpus):
        total = sum(s.numel() for s in shards[gpu_id].values())
        print(f"    GPU {gpu_id}: {total} params (전체의 1/{num_gpus})")

    # --- Forward 시뮬레이션 ---
    print(f"\n  Forward 동작:")
    for layer_idx in range(num_layers):
        # all-gather: 모든 GPU의 shard를 모아서 전체 params 복원
        gathered = torch.cat([shards[gpu][layer_idx] for gpu in range(num_gpus)])
        print(f"    Layer {layer_idx}: all-gather ({params_per_gpu}×{num_gpus}={gathered.numel()} params)"
              f" → forward 계산 → 전체 params 해제")

    # --- Backward 시뮬레이션 ---
    print(f"\n  Backward 동작:")
    for layer_idx in reversed(range(num_layers)):
        print(f"    Layer {layer_idx}: all-gather params → backward 계산"
              f" → reduce-scatter grads → params 해제")

    # --- 통신량 ---
    # total_params는 이미 전체 모델 크기다. 따라서 layer별 collective 크기를
    # 다시 num_layers와 곱할 때는 params_per_layer를 써야 한다.
    params_per_layer = layer_params[0].numel()
    total_params = sum(p.numel() for p in layer_params)
    per_rank_factor = (num_gpus - 1) / num_gpus

    print(f"\n  통신량:")
    print("    Collective payload (world-size factor 제외):")
    print(f"    model_size: {total_params} elements ({num_layers} × {params_per_layer})")
    print(f"    Forward:  {num_layers} × all-gather({params_per_layer}) = {total_params} elements")
    print(f"    Backward: {num_layers} × (all-gather({params_per_layer}) + reduce-scatter({params_per_layer}))"
          f" = {2 * total_params} elements")
    print(f"    Total:    {3 * total_params} elements = 3 × model_size (vs DDP 2 × model_size)")
    print(f"    Per-rank network traffic: ~{3 * total_params * per_rank_factor:.1f} elements"
          f" (ring collective 기준, ×{per_rank_factor:.2f})")


# ============================================================
# Part 4: FSDP API 없이 raw dist API로 구현하는 미니 예제
# ============================================================

def _flat_shard_range(numel, world_size, rank):
    """flat parameter를 rank별 균등 shard로 나눌 때 현재 rank의 [start:end) 범위."""
    assert numel % world_size == 0, "이 교육용 예시는 균등 shard만 다룹니다."
    shard_size = numel // world_size
    start = rank * shard_size
    end = start + shard_size
    return start, end, shard_size


def _all_gather_flat_param_dist(param_shard, group=None):
    """
    FSDP forward 직전: 각 rank의 parameter shard를 모아 full flat parameter를 임시 복원.

    dist.all_gather_into_tensor(output, input):
      input  = 현재 rank가 들고 있는 param_shard
      output = 모든 rank shard가 rank 순서대로 concat될 full buffer

    all_gather와의 차이:
      dist.all_gather(output_list, input)은 결과를 tensor list로 받는다.
        output_list = [rank0_shard, rank1_shard, ...]
        full = torch.cat(output_list, dim=0)

      dist.all_gather_into_tensor(output, input)은 미리 만든 하나의 큰 tensor에 바로 받는다.
        output = concat([rank0_shard, rank1_shard, ...])

    FSDP/ZeRO처럼 flat parameter buffer를 다룰 때는 list를 만들고 다시 cat할 필요가 없어서
    all_gather_into_tensor가 더 자연스럽다.

    이 full buffer는 계산용 temporary다. 계산이 끝나면 버릴 수 있고,
    optimizer step은 다시 param_shard에 대해서만 수행한다.
    """
    world_size = dist.get_world_size(group)
    full_param = torch.empty(
        param_shard.numel() * world_size,
        dtype=param_shard.dtype,
        device=param_shard.device,
    )
    dist.all_gather_into_tensor(full_param, param_shard.contiguous(), group=group)
    return full_param


def _reduce_scatter_flat_grad_dist(grad_full, group=None):
    """
    FSDP backward 후: full parameter gradient를 reduce-scatter해서 grad shard만 남긴다.

    all-reduce와 비교:
      all_reduce(grad_full): 모든 rank가 reduced full grad를 보유
      reduce_scatter:       reduced full grad 중 현재 rank shard만 보유

    그래서 FSDP/ZeRO-3는 optimizer step에 필요한 grad shard만 남겨 메모리를 줄인다.
    """
    world_size = dist.get_world_size(group)
    shard_size = grad_full.numel() // world_size
    grad_shard = torch.empty(
        shard_size,
        dtype=grad_full.dtype,
        device=grad_full.device,
    )
    dist.reduce_scatter_tensor(
        grad_shard,
        grad_full.contiguous(),
        op=dist.ReduceOp.SUM,
        group=group,
    )
    grad_shard.div_(world_size)
    return grad_shard


def raw_dist_fsdp_linear_step(param_shard, x, target, full_shape, lr=1e-2, group=None):
    """
    FSDP API 없이 raw torch.distributed collective만으로 Linear layer 1 step 수행.

    각 rank가 평소 들고 있는 것:
      param_shard: full weight를 flat하게 자른 shard

    Step:
      1. all_gather_into_tensor로 full weight temporary 복원
      2. full weight로 forward/backward 계산
      3. full weight gradient를 reduce_scatter_tensor로 grad_shard로 축소
      4. 각 rank가 자기 param_shard만 SGD update

    실제 FSDP는 이 과정을 module hook, autograd hook, prefetch/overlap, mixed precision,
    state_dict 처리와 함께 자동화한다. 여기서는 핵심 collective만 직접 보여준다.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("dist.init_process_group() 이후에 호출해야 합니다.")

    # 1. Forward에 필요한 full param temporary를 복원한다.
    full_param_flat = _all_gather_flat_param_dist(param_shard, group)
    # all-gather 결과는 param_shard와 autograd graph로 연결되어 있지 않으므로,
    # 교육용 예제에서는 full temporary에 requires_grad를 걸고 backward 후 grad를 직접 shard로 나눈다.
    full_param = full_param_flat.view(full_shape).detach().requires_grad_(True)

    # 2. 일반 Linear forward/backward.
    pred = x @ full_param.t()
    loss = F.mse_loss(pred, target)
    loss.backward()

    # 3. full grad를 reduce-scatter해서 현재 rank가 update할 grad shard만 받는다.
    grad_full_flat = full_param.grad.reshape(-1)
    grad_shard = _reduce_scatter_flat_grad_dist(grad_full_flat, group)

    # 4. Optimizer step은 local shard에 대해서만 수행한다.
    with torch.no_grad():
        param_shard.add_(grad_shard, alpha=-lr)

    return loss.detach(), grad_shard


def run_raw_dist_fsdp_smoke_test():
    """
    FSDP wrapper 없이 raw dist API로 FSDP 핵심 collective를 확인.

    단일 GPU:
      python implementation-practice-codex/lessons/022-fsdp/solution.py rawdist

    멀티 GPU:
      torchrun --nproc_per_node=2 implementation-practice-codex/lessons/022-fsdp/solution.py rawdist
    """
    rank, local_rank, world_size, device, initialized_here = _init_cuda_dist()

    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        out_dim, in_dim = 8, 4
        numel = out_dim * in_dim
        start, end, _ = _flat_shard_range(numel, world_size, rank)

        # 모든 rank가 같은 초기 full weight를 만들고, 자기 shard만 보관한다.
        # 실제 FSDP에서는 checkpoint load나 init 과정에서 shard만 유지하게 된다.
        init_full = torch.randn(out_dim, in_dim, device=device)
        param_shard = init_full.reshape(-1)[start:end].contiguous()

        x = torch.randn(3, in_dim, device=device)
        target = torch.randn(3, out_dim, device=device)

        if rank == 0:
            print("=" * 60)
            print("Raw dist FSDP-style Smoke Test")
            print("=" * 60)
            print(f"  world_size={world_size}, full_param={tuple(init_full.shape)}")

        for step in range(2):
            loss, grad_shard = raw_dist_fsdp_linear_step(
                param_shard,
                x,
                target,
                full_shape=(out_dim, in_dim),
                lr=1e-2,
            )
            avg_loss = _average_loss_for_logging(loss, world_size)
            print(
                f"  rank {rank} step {step}: "
                f"shard={param_shard.numel()} grad_shard={grad_shard.numel()} "
                f"avg_loss={avg_loss.item():.4f}"
            )

    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# Part 5: 실제 GPU FSDP smoke test
# ============================================================

class TinyFSDPBlock(nn.Module):
    """FSDP로 감쌀 작은 Transformer FFN block."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class TinyFSDPModel(nn.Module):
    """GPU smoke test용 작은 decoder-like 모델."""

    def __init__(self, vocab_size=256, embed_dim=64, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TinyFSDPBlock(embed_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_cuda_dist():
    """torchrun 또는 단일 GPU 직접 실행 모두 지원하는 process group 초기화."""
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP GPU smoke test는 CUDA GPU가 필요합니다.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed를 사용할 수 없는 PyTorch 빌드입니다.")

    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _find_free_port())

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])

    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}인데 CUDA device는 {torch.cuda.device_count()}개만 보입니다."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    initialized_here = False
    if not dist.is_initialized():
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        initialized_here = True

    return rank, local_rank, world_size, device, initialized_here


def _average_loss_for_logging(loss, world_size):
    """rank별 loss scalar를 평균내서 로그로 찍기 위한 helper."""
    # 여기서 원하는 것은 학습에 쓰는 loss가 아니라 "화면에 찍을 숫자"다.
    #
    # loss 자체는 grad_fn을 가진 autograd Tensor다. 이 텐서에 바로 all_reduce를 걸면
    # logging용 통신/나눗셈까지 autograd가 추적해야 하는 연산처럼 보인다. 우리는 이미
    # loss.backward()를 호출했고, 평균 loss는 gradient 계산에 전혀 필요 없다.
    #
    # detach()는 같은 값을 보되 autograd history를 끊은 Tensor를 만든다.
    # 단, detach()만 하면 원래 loss와 같은 storage를 공유할 수 있다.
    #
    # dist.all_reduce(avg_loss)와 avg_loss /= world_size는 avg_loss 값을 직접 바꾸는
    # in-place 연산이다. 그래서 clone()으로 별도 scalar buffer를 만든 뒤 그 buffer만
    # all_reduce한다. 이렇게 하면 logging 때문에 원래 loss Tensor 값이 바뀌지 않는다.
    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss /= world_size
    return avg_loss


def run_fsdp_gpu_smoke_test():
    """
    실제 CUDA GPU에서 PyTorch FSDP1 forward/backward/optimizer step을 확인.

    단일 GPU:
      python implementation-practice/07-fsdp/fsdp.py gpu

    멀티 GPU:
      torchrun --nproc_per_node=2 implementation-practice/07-fsdp/fsdp.py gpu
    """
    rank, local_rank, world_size, device, initialized_here = _init_cuda_dist()

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.cuda.reset_peak_memory_stats(device)

        model = TinyFSDPModel().to(device)

        for idx, block in enumerate(model.blocks):
            model.blocks[idx] = FSDP(
                block,
                device_id=device,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True,
            )

        # 왜 block을 감싼 뒤 model 전체를 또 감싸나?
        #
        # 1. 안쪽 block wrap:
        #    각 Transformer block을 독립 FSDP unit으로 만든다. FSDP는 unit 단위로
        #    "params all-gather -> compute -> params free"를 수행하므로, root만 감쌀 때보다
        #    동시에 들고 있어야 하는 full parameter 범위가 작아진다.
        #
        # 2. 바깥 root wrap:
        #    embedding, final ln, lm head처럼 block 밖에 남은 parameter도 shard 대상에 넣는다.
        #    또한 전체 모델에 root FSDP hook/state_dict/optimizer traversal의 기준점을 만든다.
        #    이미 FSDP인 child block은 nested unit으로 취급되며, 같은 parameter를 중복 shard하려는
        #    목적이 아니다.
        model = FSDP(
            model,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(4321 + rank)
        batch_size, seq_len, vocab_size = 4, 16, 256
        input_ids = torch.randint(vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(vocab_size, (batch_size, seq_len), device=device)

        if rank == 0:
            print("=" * 60)
            print("FSDP GPU Smoke Test")
            print("=" * 60)
            print(f"  world_size={world_size}, local_rank={local_rank}, device={device}")

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            avg_loss = _average_loss_for_logging(loss, world_size)

            if rank == 0:
                print(f"  step {step}: avg_loss={avg_loss.item():.4f}")

        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  rank {rank}: peak CUDA memory = {peak_mb:.1f} MB")

    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# Part 6: 실제 GPU FSDP2 smoke test
# ============================================================

def run_fsdp2_gpu_smoke_test():
    """
    실제 CUDA GPU에서 PyTorch FSDP2(fully_shard) forward/backward/optimizer step을 확인.

    FSDP1은 FSDP(module) wrapper를 새로 만든다.
    FSDP2는 fully_shard(module)을 호출해서 원래 module에 hook을 등록하고,
    parameter를 DTensor 기반 shard로 in-place 변환한다.

    단일 GPU:
      python implementation-practice/07-fsdp/fsdp.py fsdp2

    멀티 GPU:
      torchrun --nproc_per_node=2 implementation-practice/07-fsdp/fsdp.py fsdp2
    """
    rank, local_rank, world_size, device, initialized_here = _init_cuda_dist()

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.cuda.reset_peak_memory_stats(device)

        model = TinyFSDPModel().to(device)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

        # smoke test에서는 dtype 변환 없이 FSDP2 sharding 경로만 확인한다.
        # BF16까지 확인하려면 아래를 MixedPrecisionPolicy(param_dtype=torch.bfloat16,
        # reduce_dtype=torch.float32)로 바꾸면 된다.
        mp_policy = MixedPrecisionPolicy()

        for block in model.blocks:
            # FSDP2는 wrapper 객체를 반환하는 방식이 아니라 module을 in-place로 바꾼다.
            # block 단위로 먼저 fully_shard하면 FSDP1 nested wrap과 같은 unit granularity가 된다.
            fully_shard(block, mesh=mesh, mp_policy=mp_policy)

        # root에도 fully_shard를 적용해서 block 밖 parameter까지 shard하고,
        # 전체 모델의 FSDP hook/state_dict 기준점을 만든다.
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(4321 + rank)
        batch_size, seq_len, vocab_size = 4, 16, 256
        input_ids = torch.randint(vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(vocab_size, (batch_size, seq_len), device=device)

        if rank == 0:
            print("=" * 60)
            print("FSDP2 GPU Smoke Test")
            print("=" * 60)
            print(f"  world_size={world_size}, local_rank={local_rank}, device={device}")

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            avg_loss = _average_loss_for_logging(loss, world_size)

            if rank == 0:
                print(f"  step {step}: avg_loss={avg_loss.item():.4f}")

        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  rank {rank}: peak CUDA memory = {peak_mb:.1f} MB")

    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# Part 7: Megatron-FSDP reference example
# ============================================================

def print_megatron_fsdp_reference():
    """
    Megatron-FSDP를 실제 프로젝트에 붙일 때의 최소 API/flag 예시.

    이 함수는 megatron-core 설치 없이도 읽을 수 있도록 reference snippet만 출력한다.
    실제 실행은 NVIDIA Megatron-LM/Megatron-Core 환경에서 해야 한다.
    """
    print("=" * 60)
    print("Megatron-FSDP Reference")
    print("=" * 60)
    print("""
CLI flags used by Megatron-LM training scripts:

  --use-megatron-fsdp
  --data-parallel-sharding-strategy optim_grads_params
  --use-distributed-optimizer
  --no-gradient-accumulation-fusion

Python API sketch:

  import os
  import torch
  from torch.distributed.device_mesh import init_device_mesh
  from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
      fully_shard,
      fully_shard_model,
      fully_shard_optimizer,
  )
  from megatron.core.transformer.transformer_layer import TransformerLayer

  device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
  mesh = init_device_mesh(
      "cuda",
      (dp_size, tp_size),
      mesh_dim_names=("dp_shard", "tp"),
  )

  model = build_megatron_core_model(config).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  # Option A: shard model and optimizer together.
  model, optimizer = fully_shard(
      model,
      optimizer,
      device_mesh=mesh,
      dp_shard_dim="dp_shard",
      tp_dim="tp",
      fsdp_unit_modules=[TransformerLayer],
      zero_dp_strategy="optim_grads_params",
      device=device,
      overlap_grad_reduce=True,
      overlap_param_gather=True,
  )

  # Option B: do the two steps explicitly.
  model = fully_shard_model(
      model,
      device_mesh=mesh,
      dp_shard_dim="dp_shard",
      tp_dim="tp",
      fsdp_unit_modules=[TransformerLayer],
      zero_dp_strategy="optim_grads_params",
      device=device,
  )
  optimizer = fully_shard_optimizer(torch.optim.AdamW(model.parameters(), lr=lr))

  # Training loop shape stays ordinary PyTorch: forward -> loss -> backward -> step.
  logits = model(input_ids)
  loss = loss_fn(logits, labels)
  loss.backward()
  optimizer.step()

Key points:

  - zero_dp_strategy="optim_grads_params" is the ZeRO-3 style mode:
    parameters, gradients, and optimizer states are sharded.
  - fsdp_unit_modules defines the gather/free unit, similar to wrapping each
    Transformer block in PyTorch FSDP.
  - tp_dim tells Megatron-FSDP how FSDP sharding composes with tensor parallel shards.
  - overlap_grad_reduce and overlap_param_gather are the main performance knobs.
""")


# ============================================================
# Part 8: FSDP + TP 조합 (2D Parallelism)
# ============================================================
#
# 대규모 모델에서는 FSDP와 TP를 함께 사용:
#
#   mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
#
#   # TP 먼저 적용 (intra-layer)
#   parallelize_module(block.ffn, mesh["tp"], {
#       "fc1": ColwiseParallel(),
#       "fc2": RowwiseParallel(),
#   })
#
#   # FSDP 적용 (inter-layer, DP 차원)
#   fully_shard(block, mesh=mesh["dp"])
#
# 예: 32 GPUs = 4 DP × 8 TP
#   - 8 GPUs가 하나의 TP group (layer 내부를 나눔)
#   - 4 TP groups가 FSDP로 data parallel (layer를 나눔)
#
#                     TP group (8 GPUs)
#                  ┌──────────────────┐
#   FSDP group 0: │ GPU0 ... GPU7    │  ← 같은 layer의 weight를 column/row split
#   FSDP group 1: │ GPU8 ... GPU15   │
#   FSDP group 2: │ GPU16 ... GPU23  │
#   FSDP group 3: │ GPU24 ... GPU31  │
#                  └──────────────────┘
#                  각 FSDP group은 동일 데이터의 다른 micro-batch 처리


# ============================================================
# Part 9: 메모리 비교
# ============================================================

def memory_comparison():
    print("\n" + "=" * 60)
    print("Memory per GPU (7B model, 4 GPUs, Adam, BF16)")
    print("=" * 60)

    P = 7  # 7B params
    N = 4  # GPUs
    bf16 = 2  # bytes
    fp32 = 4

    print(f"\n  {'Component':<25} {'DDP':<15} {'FSDP':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")

    ddp_params = P * bf16
    ddp_grads = P * bf16
    ddp_opt = P * fp32 * 2  # m + v in FP32
    ddp_master = P * fp32   # master weights
    ddp_total = ddp_params + ddp_grads + ddp_opt + ddp_master

    fsdp_params = P * bf16 / N
    fsdp_grads = P * bf16 / N
    fsdp_opt = P * fp32 * 2 / N
    fsdp_master = P * fp32 / N
    fsdp_total = fsdp_params + fsdp_grads + fsdp_opt + fsdp_master
    # forward 시 all-gather하면 일시적으로 전체 params 필요
    fsdp_peak = fsdp_total + P * bf16  # shard + 1 layer의 전체 params

    print(f"  {'Parameters (BF16)':<25} {ddp_params:.1f} GB{'':<7} {fsdp_params:.1f} GB")
    print(f"  {'Gradients (BF16)':<25} {ddp_grads:.1f} GB{'':<7} {fsdp_grads:.1f} GB")
    print(f"  {'Optimizer (FP32 m,v)':<25} {ddp_opt:.1f} GB{'':<7} {fsdp_opt:.1f} GB")
    print(f"  {'Master weights (FP32)':<25} {ddp_master:.1f} GB{'':<7} {fsdp_master:.1f} GB")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Total (steady)':<25} {ddp_total:.1f} GB{'':<7} {fsdp_total:.1f} GB")
    print(f"  {'Peak (fwd all-gather)':<25} {'N/A':<15} ~{fsdp_peak:.1f} GB")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    if mode == "rawdist":
        run_raw_dist_fsdp_smoke_test()
    elif mode in ("gpu", "fsdp1"):
        run_fsdp_gpu_smoke_test()
    elif mode == "fsdp2":
        run_fsdp2_gpu_smoke_test()
    elif mode == "megatron":
        print_megatron_fsdp_reference()
    else:
        simulate_fsdp()
        memory_comparison()
