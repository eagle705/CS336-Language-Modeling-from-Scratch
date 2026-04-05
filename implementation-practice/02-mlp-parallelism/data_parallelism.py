"""
Data Parallelism (DP) for MLP
================================
모든 GPU가 동일한 모델을 가지고, 다른 데이터로 학습.

핵심 아이디어:
  - 각 GPU가 전체 모델의 복사본 보유
  - batch를 GPU 수만큼 나눠서 각 GPU에 분배
  - 각 GPU가 독립적으로 forward/backward
  - gradient를 all-reduce로 평균 → 모든 GPU가 동일 update

    Data batch: [B0, B1, B2, B3]
                  |   |   |   |
                  v   v   v   v
    GPU 0: Model(B0)  GPU 1: Model(B1)  GPU 2: Model(B2)  GPU 3: Model(B3)
       ↓ grad_0          ↓ grad_1          ↓ grad_2          ↓ grad_3
       └─────────── All-Reduce (mean) ──────────────┘
                         ↓
                 avg_grad = mean(grad_0..3)
                         ↓
                 모든 GPU가 동일한 update
                 → 모델이 항상 동기화 상태

통신 패턴:
  All-Reduce: 2 * model_size (reduce-scatter + all-gather)
  통신은 backward 중에 overlap 가능 (DDP bucketing)

인터뷰 포인트:
  1. DP는 가장 단순하고 확장성 좋은 parallelism
  2. 한계: 모델이 1개 GPU 메모리에 들어가야 함 → 큰 모델은 FSDP/ZeRO
  3. Effective batch size = micro_batch × num_gpus
  4. gradient all-reduce는 backward과 overlap 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: DDP 동작 시뮬레이션
# ============================================================

def simulate_ddp():
    """DDP의 forward → backward → all-reduce → update를 시뮬레이션."""
    print("=" * 60)
    print("Data Parallelism (DDP) Simulation")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4
    embed_dim = 8
    hidden_dim = 16

    # 동일 모델 (모든 GPU가 같은 weight)
    W1 = torch.randn(embed_dim, hidden_dim)
    W2 = torch.randn(hidden_dim, embed_dim)

    # 전체 batch를 GPU별로 나눔
    total_batch = torch.randn(num_gpus * 2, 4, embed_dim)  # (8, 4, embed_dim)
    micro_batches = total_batch.chunk(num_gpus)  # 각 GPU에 2개씩

    print(f"\n  Config: {num_gpus} GPUs, embed={embed_dim}, ffn={hidden_dim}")
    print(f"  Total batch: {total_batch.shape}")
    print(f"  Micro batch per GPU: {micro_batches[0].shape}")

    # --- 각 GPU에서 독립적으로 forward/backward ---
    all_grads_W1 = []
    all_grads_W2 = []
    all_losses = []

    for gpu_id in range(num_gpus):
        x = micro_batches[gpu_id]

        # Forward
        h = F.gelu(x @ W1)
        out = h @ W2
        loss = out.pow(2).mean()
        all_losses.append(loss.item())

        # Backward (수동 계산)
        # dL/dout = 2 * out / N
        N = out.numel()
        d_out = 2 * out / N
        d_W2 = h.reshape(-1, hidden_dim).T @ d_out.reshape(-1, embed_dim)
        d_h = d_out @ W2.T
        # GELU backward (근사: sigmoid * (1 + x * (1 - sigmoid)))
        z = x @ W1
        gelu_grad = torch.sigmoid(1.702 * z) * (1 + 1.702 * z * (1 - torch.sigmoid(1.702 * z)))
        d_z = d_h * gelu_grad
        d_W1 = x.reshape(-1, embed_dim).T @ d_z.reshape(-1, hidden_dim)

        all_grads_W1.append(d_W1)
        all_grads_W2.append(d_W2)

    print(f"\n  [각 GPU의 독립 계산]")
    for gpu_id in range(num_gpus):
        print(f"    GPU {gpu_id}: loss={all_losses[gpu_id]:.4f}, "
              f"grad_W1 norm={all_grads_W1[gpu_id].norm():.4f}")

    # --- All-Reduce: gradient 평균 ---
    avg_grad_W1 = sum(all_grads_W1) / num_gpus
    avg_grad_W2 = sum(all_grads_W2) / num_gpus

    print(f"\n  [All-Reduce 후]")
    print(f"    avg grad_W1 norm: {avg_grad_W1.norm():.4f}")
    print(f"    avg grad_W2 norm: {avg_grad_W2.norm():.4f}")
    print(f"    → 모든 GPU가 이 동일한 gradient로 update")

    # --- SGD Update ---
    lr = 0.01
    W1_new = W1 - lr * avg_grad_W1
    W2_new = W2 - lr * avg_grad_W2
    print(f"\n  [SGD Update (lr={lr})]")
    print(f"    W1 변화량: {(W1 - W1_new).abs().mean():.6f}")
    print(f"    W2 변화량: {(W2 - W2_new).abs().mean():.6f}")

    # --- 통신량 분석 ---
    model_params = W1.numel() + W2.numel()
    comm_bytes = model_params * 4 * 2  # FP32, all-reduce = 2x model size
    print(f"\n  [통신량]")
    print(f"    Model params: {model_params}")
    print(f"    All-reduce: 2 × {model_params} × 4 bytes = {comm_bytes} bytes")
    print(f"    (reduce-scatter + all-gather = 2 × model_size)")


# ============================================================
# Part 2: DDP vs 단순 DP 비교
# ============================================================

def ddp_vs_dp():
    """nn.DataParallel vs DistributedDataParallel 비교."""
    print("\n" + "=" * 60)
    print("DataParallel vs DistributedDataParallel")
    print("=" * 60)

    print("""
  nn.DataParallel (DP) — 사용하지 마세요!
  ─────────────────────────────────────────
    - 단일 프로세스, multi-thread (GIL bottleneck)
    - GPU 0에서 scatter/gather → GPU 0에 메모리 불균형
    - GPU 0 bottleneck으로 N GPU여도 N배 빨라지지 않음

    model = nn.DataParallel(model)  # ← 쓰지 마세요

  DistributedDataParallel (DDP) — 표준
  ─────────────────────────────────────────
    - 멀티 프로세스 (GPU당 1 프로세스)
    - 각 프로세스가 독립적으로 forward/backward
    - NCCL all-reduce로 gradient 동기화
    - GIL 없음, 균등한 메모리 사용

    model = DDP(model, device_ids=[local_rank])  # ← 이걸 쓰세요

  ┌──────────────┬───────────────────┬──────────────────────┐
  │              │ nn.DataParallel   │ DDP                  │
  ├──────────────┼───────────────────┼──────────────────────┤
  │ 프로세스      │ 1 (multi-thread)  │ N (multi-process)    │
  │ GIL          │ 있음 (bottleneck) │ 없음                 │
  │ GPU 메모리   │ GPU 0에 집중       │ 균등 분배            │
  │ 통신         │ GPU 0 gather      │ NCCL all-reduce      │
  │ 스케일링     │ ~2-3 GPU까지      │ 수천 GPU 가능        │
  │ multi-node   │ 불가              │ 가능                 │
  └──────────────┴───────────────────┴──────────────────────┘
    """)


# ============================================================
# Part 3: Gradient Accumulation 시뮬레이션
# ============================================================

def simulate_gradient_accumulation():
    """
    Gradient Accumulation: micro-batch를 여러 번 backward 후 한 번 update.

    용도: GPU 메모리에 큰 batch가 안 들어갈 때
    effective_batch = micro_batch × accum_steps × num_gpus
    """
    print("\n" + "=" * 60)
    print("Gradient Accumulation Simulation")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(8, 4, bias=False)

    micro_batch_size = 2
    accum_steps = 4
    effective_batch_size = micro_batch_size * accum_steps
    num_gpus = 2

    print(f"\n  micro_batch = {micro_batch_size}")
    print(f"  accum_steps = {accum_steps}")
    print(f"  num_gpus    = {num_gpus}")
    print(f"  effective_batch = {micro_batch_size} × {accum_steps} × {num_gpus}"
          f" = {effective_batch_size * num_gpus}")

    # --- 방법 1: 큰 batch 한번에 (비교 기준) ---
    big_batch = torch.randn(effective_batch_size, 8)
    out = model(big_batch)
    loss = out.pow(2).mean()
    loss.backward()
    baseline_grad = model.weight.grad.clone()
    model.zero_grad()

    # --- 방법 2: Gradient accumulation ---
    micro_batches = big_batch.chunk(accum_steps)

    for i, mb in enumerate(micro_batches):
        out = model(mb)
        # loss를 accum_steps로 나눠서 평균이 맞도록
        loss = out.pow(2).mean() / accum_steps
        loss.backward()  # grad가 누적됨!

    accum_grad = model.weight.grad.clone()

    diff = (baseline_grad - accum_grad).abs().max().item()
    print(f"\n  Big batch grad vs Accumulated grad:")
    print(f"    Max diff: {diff:.2e}")
    print(f"    Result:   {'PASSED' if diff < 1e-5 else 'FAILED'}")

    print(f"""
  코드 패턴:
    for step, batch in enumerate(dataloader):
        loss = model(batch) / accum_steps   # ← accum_steps로 나눔
        loss.backward()                      # gradient 누적

        if (step + 1) % accum_steps == 0:   # ← N번마다 update
            optimizer.step()
            optimizer.zero_grad()
    """)


# ============================================================
# Part 4: DDP + torchrun 코드 템플릿
# ============================================================
#
# 실행: torchrun --nproc_per_node=4 data_parallelism.py ddp
#
# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
#
# def train():
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#
#     model = MyModel().cuda()
#     model = DDP(model, device_ids=[local_rank])
#
#     sampler = DistributedSampler(dataset, rank=rank)
#     loader = DataLoader(dataset, batch_size=32, sampler=sampler)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#
#     for epoch in range(num_epochs):
#         sampler.set_epoch(epoch)      # epoch별 shuffle 패턴 변경
#         for batch in loader:
#             loss = model(batch.cuda())
#             loss.backward()           # ← all-reduce 자동!
#             optimizer.step()
#             optimizer.zero_grad()
#
#     dist.destroy_process_group()


# ============================================================
# Part 5: DTensor로 DP (PyTorch 2.x native)
# ============================================================
#
# from torch.distributed.device_mesh import init_device_mesh
# from torch.distributed.tensor import Replicate
# from torch.distributed.fsdp import fully_shard   # FSDP2 = ZeRO-3 style DP
#
# # 방법 1: DDP (replicate)
# mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
# # DDP는 내부적으로 모든 param이 Replicate() placement
#
# # 방법 2: FSDP2 (sharded DP) — 메모리 효율적
# for block in model.blocks:
#     fully_shard(block, mesh=mesh)
# fully_shard(model, mesh=mesh)
#
# # 방법 3: 2D mesh (DP + TP)
# mesh_2d = init_device_mesh("cuda", (dp_size, tp_size),
#                            mesh_dim_names=("dp", "tp"))
# # TP → mesh_2d["tp"], DP → mesh_2d["dp"]


if __name__ == "__main__":
    simulate_ddp()
    ddp_vs_dp()
    simulate_gradient_accumulation()
