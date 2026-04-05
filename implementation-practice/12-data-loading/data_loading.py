"""
Data Loading for LLM Training
===============================
대규모 학습 데이터를 효율적으로 로드하는 방법.

핵심 고려사항:
  1. 데이터가 디스크에 있을 때 I/O bottleneck 방지
  2. 분산 학습에서 각 GPU가 다른 데이터를 받도록
  3. 메모리에 전체 데이터를 올리지 않고 streaming
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import os
import struct


# ============================================================
# Part 1: Memory-Mapped Dataset
# ============================================================

class MemmapTokenDataset(Dataset):
    """
    Memory-mapped 파일로 대규모 토큰 데이터 로드.

    원리:
    - 전처리: 텍스트 → 토큰 ID 배열 → .bin 파일로 저장
    - 학습: np.memmap으로 파일을 메모리에 매핑 (실제로 읽지 않음)
    - 접근 시에만 해당 부분을 디스크에서 읽음 (OS가 관리)

    장점:
    - 메모리 사용량: 거의 0 (OS page cache가 관리)
    - 랜덤 접근: O(1) (파일 내 offset 계산)
    - 여러 프로세스가 동일 파일을 공유 가능
    """

    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        # np.memmap: 파일을 메모리에 매핑 (실제 메모리 사용 X)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.num_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # input: tokens[0:seq_len], target: tokens[1:seq_len+1]
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ============================================================
# Part 2: DistributedSampler
# ============================================================
#
# 분산 학습에서 각 GPU가 다른 데이터를 받도록 하는 방법.
#
# from torch.utils.data.distributed import DistributedSampler
#
# sampler = DistributedSampler(
#     dataset,
#     num_replicas=world_size,  # 전체 GPU 수
#     rank=rank,                # 이 GPU의 rank
#     shuffle=True,
#     seed=42,
# )
#
# dataloader = DataLoader(dataset, batch_size=B, sampler=sampler)
#
# for epoch in range(num_epochs):
#     sampler.set_epoch(epoch)  # epoch마다 shuffle 패턴 변경!
#     for batch in dataloader:
#         ...
#
# 동작 원리:
#   전체 인덱스: [0, 1, 2, 3, 4, 5, 6, 7]
#   shuffle:     [3, 7, 1, 5, 0, 4, 2, 6]
#   GPU 0 (rank=0): [3, 1, 0, 2]  (짝수 위치)
#   GPU 1 (rank=1): [7, 5, 4, 6]  (홀수 위치)
#   → 겹치는 데이터 없이 전체 데이터 커버


# ============================================================
# Part 3: Iterable Dataset (Streaming)
# ============================================================

class StreamingTokenDataset(IterableDataset):
    """
    Streaming 방식으로 여러 파일에서 데이터 로드.

    장점:
    - 파일을 순차적으로 읽어서 I/O 효율적
    - 데이터 크기와 무관한 메모리 사용
    - 분산 학습에서 worker별 파일 분배 가능

    Megatron-LM, GPT-NeoX 등에서 사용하는 패턴.
    """

    def __init__(self, file_paths, seq_len, seed=42):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        # 분산 학습: 이 worker가 담당할 파일 결정
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # DataLoader의 num_workers > 0일 때
            per_worker = len(self.file_paths) // worker_info.num_workers
            start = worker_info.id * per_worker
            files = self.file_paths[start:start + per_worker]
        else:
            files = self.file_paths

        rng = np.random.RandomState(self.seed)
        rng.shuffle(files)

        buffer = np.array([], dtype=np.uint16)
        for fpath in files:
            data = np.memmap(fpath, dtype=np.uint16, mode='r')
            buffer = np.concatenate([buffer, data])

            # buffer에서 seq_len+1 단위로 잘라서 yield
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1].astype(np.int64)
                buffer = buffer[self.seq_len:]
                x = torch.from_numpy(chunk[:-1])
                y = torch.from_numpy(chunk[1:])
                yield x, y


# ============================================================
# Part 4: DataLoader 최적화
# ============================================================
#
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     num_workers=4,         # I/O를 별도 프로세스에서 수행
#     pin_memory=True,       # CPU→GPU 전송 속도 향상 (pinned memory)
#     prefetch_factor=2,     # 미리 2 batch 준비
#     persistent_workers=True,  # worker 프로세스 재사용 (초기화 비용 제거)
# )
#
# 최적화 팁:
#   1. num_workers: CPU 코어 수 / GPU 수 정도 (너무 많으면 오히려 느림)
#   2. pin_memory=True: GPU 학습 시 필수
#   3. prefetch_factor: GPU utilization이 낮으면 늘리기
#   4. persistent_workers: epoch 간 worker 유지


# ============================================================
# Part 5: Demo
# ============================================================

def demo():
    print("=" * 60)
    print("Data Loading Demo")
    print("=" * 60)

    # 테스트용 토큰 데이터 생성
    tmp_path = "/tmp/test_tokens.bin"
    num_tokens = 10000
    tokens = np.random.randint(0, 30000, size=num_tokens, dtype=np.uint16)
    tokens.tofile(tmp_path)

    # Memmap dataset
    seq_len = 128
    dataset = MemmapTokenDataset(tmp_path, seq_len)
    print(f"\n  [MemmapTokenDataset]")
    print(f"    File: {tmp_path} ({os.path.getsize(tmp_path) / 1024:.1f} KB)")
    print(f"    Tokens: {num_tokens:,}")
    print(f"    Samples: {len(dataset)} (seq_len={seq_len})")

    x, y = dataset[0]
    print(f"    Sample 0: x={x.shape}, y={y.shape}")
    print(f"    x[:5] = {x[:5].tolist()}")
    print(f"    y[:5] = {y[:5].tolist()} (x를 1칸 shift)")

    # DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\n  [DataLoader]")
    print(f"    Batch: x={batch[0].shape}, y={batch[1].shape}")

    # Streaming dataset
    print(f"\n  [StreamingTokenDataset]")
    stream_ds = StreamingTokenDataset([tmp_path], seq_len=seq_len)
    stream_loader = DataLoader(stream_ds, batch_size=4)
    batch = next(iter(stream_loader))
    print(f"    Batch: x={batch[0].shape}, y={batch[1].shape}")

    # 분산 학습 시뮬레이션
    print(f"\n  [Distributed Sampler 시뮬레이션]")
    indices = list(range(16))
    world_size = 4
    for rank in range(world_size):
        rank_indices = indices[rank::world_size]  # stride 방식
        print(f"    GPU {rank}: samples {rank_indices}")

    # cleanup
    os.remove(tmp_path)


if __name__ == "__main__":
    demo()
