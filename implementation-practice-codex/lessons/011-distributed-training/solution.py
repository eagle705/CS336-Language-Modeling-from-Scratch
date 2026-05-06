"""
Distributed Training
======================
PyTorch 분산 학습의 핵심 개념과 구현.

분산 학습 종류:
  DP (DataParallel):           단일 노드, GIL bottleneck → 비추천
  DDP (DistributedDataParallel): 멀티 프로세스, 가장 기본
  FSDP:                        ZeRO-3 방식, 메모리 효율적
"""

import torch
import torch.nn as nn


# ============================================================
# Part 1: Process Group 핵심 API
# ============================================================
#
# --- 초기화 ---
#
# import torch.distributed as dist
#
# # 방법 1: torchrun이 환경변수 설정 → 가장 간단
# # 실행: torchrun --nproc_per_node=4 train.py
# dist.init_process_group(backend="nccl")  # GPU → nccl, CPU → gloo
#
# rank = dist.get_rank()              # 이 프로세스의 global rank (0~N-1)
# world_size = dist.get_world_size()  # 전체 프로세스 수
# local_rank = int(os.environ["LOCAL_RANK"])  # 노드 내 rank (GPU 할당용)
# torch.cuda.set_device(local_rank)
#
# # 방법 2: 수동 초기화
# dist.init_process_group(
#     backend="nccl",
#     init_method="tcp://master_ip:29500",
#     rank=rank,
#     world_size=world_size,
# )
#
# --- Process Group 생성 ---
#
# # 기본 group: 전체 프로세스
# default_group = dist.group.WORLD
#
# # 서브 group: 특정 rank들만 묶기 (TP/PP에서 사용)
# tp_group = dist.new_group(ranks=[0, 1, 2, 3])  # rank 0-3이 하나의 TP group
#
# # group 내 통신
# dist.all_reduce(tensor, group=tp_group)
#
# --- torchrun 환경변수 ---
# MASTER_ADDR: master 노드 IP
# MASTER_PORT: 통신 포트
# WORLD_SIZE:  전체 프로세스 수
# RANK:        global rank
# LOCAL_RANK:  노드 내 rank


# ============================================================
# Part 2: DDP (DistributedDataParallel)
# ============================================================
#
# DDP 동작:
#   1. 각 프로세스가 동일 모델 보유
#   2. 각 프로세스가 다른 데이터 배치로 forward/backward
#   3. backward 중에 gradient all-reduce (자동)
#   4. 모든 프로세스가 동일한 gradient로 동일한 update
#
# 구현:
#   model = MyModel().cuda(local_rank)
#   model = DDP(model, device_ids=[local_rank])
#
#   # 학습 루프 (일반 PyTorch와 동일!)
#   for batch in dataloader:
#       output = model(batch)
#       loss = loss_fn(output, target)
#       loss.backward()        # ← 여기서 all-reduce 자동 수행
#       optimizer.step()
#       optimizer.zero_grad()
#
# DDP의 핵심 최적화: Gradient Bucketing
#   - gradient를 하나씩 all-reduce하면 통신 overhead 큼
#   - 여러 gradient를 bucket (기본 25MB)으로 묶어서 한번에 all-reduce
#   - backward 계산과 통신을 overlap → 성능 향상
#
#   model = DDP(model, bucket_cap_mb=25)  # bucket 크기 조절


# ============================================================
# Part 3: DDP 시뮬레이션 (single process)
# ============================================================

def simulate_ddp():
    """DDP의 gradient all-reduce를 시뮬레이션."""
    print("=" * 60)
    print("DDP Simulation (4 GPUs)")
    print("=" * 60)

    torch.manual_seed(42)
    num_gpus = 4

    # 동일 모델 (DDP: 모든 GPU가 같은 모델)
    model = nn.Linear(4, 2, bias=False)

    # 각 GPU가 다른 데이터로 계산
    gpu_grads = []
    for gpu_id in range(num_gpus):
        x = torch.randn(2, 4)  # 각 GPU의 mini-batch
        y = model(x)
        loss = y.sum()
        loss.backward()
        gpu_grads.append(model.weight.grad.clone())
        model.zero_grad()

    print(f"\n  각 GPU의 gradient:")
    for i, g in enumerate(gpu_grads):
        print(f"    GPU {i}: mean={g.mean():.4f}, norm={g.norm():.4f}")

    # All-reduce: 평균
    avg_grad = sum(gpu_grads) / num_gpus
    print(f"\n  All-reduce 후 (평균):")
    print(f"    mean={avg_grad.mean():.4f}, norm={avg_grad.norm():.4f}")
    print(f"    → 모든 GPU가 이 동일한 gradient로 update")


# ============================================================
# Part 4: 전체 학습 코드 템플릿
# ============================================================

DDP_TRAINING_TEMPLATE = """
# === DDP Training Template ===
# 실행: torchrun --nproc_per_node=4 --nnodes=1 train.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    # (1) 분산 초기화
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # (2) 모델 (모든 rank에서 동일하게 생성)
    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])

    # (3) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # (4) Data: DistributedSampler로 각 GPU에 다른 데이터
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler,
                            num_workers=4, pin_memory=True)

    # (5) 학습 루프
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # shuffle을 epoch마다 다르게
        model.train()

        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            targets = batch["targets"].cuda()

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            loss.backward()      # all-reduce gradients (DDP 자동)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                print(f"loss: {loss.item():.4f}")

    # (6) 저장 (rank 0만)
    if rank == 0:
        torch.save(model.module.state_dict(), "model.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
"""


# ============================================================
# Part 5: Multi-node 설정
# ============================================================

def multinode_guide():
    print("\n" + "=" * 60)
    print("Multi-Node Training Guide")
    print("=" * 60)

    print("""
  Single node (4 GPUs):
    torchrun --nproc_per_node=4 train.py

  Multi node (2 nodes × 4 GPUs):
    # Node 0 (master):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
             --master_addr=10.0.0.1 --master_port=29500 train.py

    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
             --master_addr=10.0.0.1 --master_port=29500 train.py

  SLURM cluster:
    srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 \\
         torchrun --nproc_per_node=4 train.py

  통신 backend:
    NCCL (GPU):  GPU 간 통신 최적화. InfiniBand, NVLink 지원.
    Gloo (CPU):  CPU 학습용. TCP/IP 기반.
    """)


if __name__ == "__main__":
    simulate_ddp()
    print("\n  DDP Training Template:")
    print(DDP_TRAINING_TEMPLATE)
    multinode_guide()
