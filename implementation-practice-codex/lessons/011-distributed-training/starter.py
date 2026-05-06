"""Distributed Training
======================
PyTorch 분산 학습의 핵심 개념과 구현.

분산 학습 종류:
  DP (DataParallel):           단일 노드, GIL bottleneck → 비추천
  DDP (DistributedDataParallel): 멀티 프로세스, 가장 기본
  FSDP:                        ZeRO-3 방식, 메모리 효율적

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn

def simulate_ddp():
    """DDP의 gradient all-reduce를 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_ddp; compare with solution.py only after trying.')
DDP_TRAINING_TEMPLATE = '\n# === DDP Training Template ===\n# 실행: torchrun --nproc_per_node=4 --nnodes=1 train.py\n\nimport os\nimport torch\nimport torch.distributed as dist\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nfrom torch.utils.data.distributed import DistributedSampler\n\ndef main():\n    # (1) 분산 초기화\n    dist.init_process_group("nccl")\n    rank = dist.get_rank()\n    local_rank = int(os.environ["LOCAL_RANK"])\n    world_size = dist.get_world_size()\n    torch.cuda.set_device(local_rank)\n\n    # (2) 모델 (모든 rank에서 동일하게 생성)\n    model = MyModel().cuda()\n    model = DDP(model, device_ids=[local_rank])\n\n    # (3) Optimizer\n    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)\n\n    # (4) Data: DistributedSampler로 각 GPU에 다른 데이터\n    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)\n    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler,\n                            num_workers=4, pin_memory=True)\n\n    # (5) 학습 루프\n    for epoch in range(num_epochs):\n        sampler.set_epoch(epoch)  # shuffle을 epoch마다 다르게\n        model.train()\n\n        for batch in dataloader:\n            input_ids = batch["input_ids"].cuda()\n            targets = batch["targets"].cuda()\n\n            logits = model(input_ids)\n            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))\n\n            loss.backward()      # all-reduce gradients (DDP 자동)\n\n            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n            optimizer.step()\n            optimizer.zero_grad()\n\n            if rank == 0:\n                print(f"loss: {loss.item():.4f}")\n\n    # (6) 저장 (rank 0만)\n    if rank == 0:\n        torch.save(model.module.state_dict(), "model.pt")\n\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    main()\n'

def multinode_guide():
    raise NotImplementedError('TODO: implement multinode_guide; compare with solution.py only after trying.')
if __name__ == '__main__':
    simulate_ddp()
    print('\n  DDP Training Template:')
    print(DDP_TRAINING_TEMPLATE)
    multinode_guide()
