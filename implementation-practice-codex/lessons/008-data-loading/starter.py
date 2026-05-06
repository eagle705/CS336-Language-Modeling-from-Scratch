"""Data Loading for LLM Training
===============================
대규모 학습 데이터를 효율적으로 로드하는 방법.

핵심 고려사항:
  1. 데이터가 디스크에 있을 때 I/O bottleneck 방지
  2. 분산 학습에서 각 GPU가 다른 데이터를 받도록
  3. 메모리에 전체 데이터를 올리지 않고 streaming

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import os
import struct

class MemmapTokenDataset(Dataset):
    """Memory-mapped 파일로 대규모 토큰 데이터 로드.

원리:
- 전처리: 텍스트 → 토큰 ID 배열 → .bin 파일로 저장
- 학습: np.memmap으로 파일을 메모리에 매핑 (실제로 읽지 않음)
- 접근 시에만 해당 부분을 디스크에서 읽음 (OS가 관리)

장점:
- 메모리 사용량: 거의 0 (OS page cache가 관리)
- 랜덤 접근: O(1) (파일 내 offset 계산)
- 여러 프로세스가 동일 파일을 공유 가능"""

    def __init__(self, data_path, seq_len):
        raise NotImplementedError('TODO: implement MemmapTokenDataset.__init__; compare with solution.py only after trying.')

    def __len__(self):
        raise NotImplementedError('TODO: implement MemmapTokenDataset.__len__; compare with solution.py only after trying.')

    def __getitem__(self, idx):
        raise NotImplementedError('TODO: implement MemmapTokenDataset.__getitem__; compare with solution.py only after trying.')

class StreamingTokenDataset(IterableDataset):
    """Streaming 방식으로 여러 파일에서 데이터 로드.

장점:
- 파일을 순차적으로 읽어서 I/O 효율적
- 데이터 크기와 무관한 메모리 사용
- 분산 학습에서 worker별 파일 분배 가능

Megatron-LM, GPT-NeoX 등에서 사용하는 패턴."""

    def __init__(self, file_paths, seq_len, seed=42):
        raise NotImplementedError('TODO: implement StreamingTokenDataset.__init__; compare with solution.py only after trying.')

    def __iter__(self):
        raise NotImplementedError('TODO: implement StreamingTokenDataset.__iter__; compare with solution.py only after trying.')

def demo():
    raise NotImplementedError('TODO: implement demo; compare with solution.py only after trying.')
if __name__ == '__main__':
    demo()
