"""Distributed Checkpointing
===========================
수천 GPU에서 학습한 모델을 저장/로드하는 방법.

문제:
  - TP=8, PP=4 → 각 GPU가 모델의 1/32만 보유
  - 저장: 32개 shard를 각각 파일로? 하나로 합쳐서?
  - 로드: TP=8로 저장 → TP=4로 로드하려면? (resharding)
  - 대규모 모델(수백 GB) 저장/로드 시 I/O bottleneck

해결:
  1. Sharded checkpoint: 각 rank가 자기 shard를 독립적으로 저장
  2. Resharding: 저장 시 parallelism과 다른 설정으로 로드 가능
  3. Async checkpoint: 저장을 background에서 수행하여 학습 중단 최소화

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
import os
import json
import shutil
from pathlib import Path

def naive_checkpoint_demo():
    """단순 torch.save의 문제점."""
    raise NotImplementedError('TODO: implement naive_checkpoint_demo; compare with solution.py only after trying.')

class ShardedCheckpointManager:
    """Sharded checkpoint: 각 rank가 자기 shard를 독립 저장.

디렉토리 구조:
  checkpoint/
    metadata.json       ← 전체 구조 정보 (shape, dtype, sharding 방식)
    rank_0.pt           ← rank 0의 shard
    rank_1.pt
    ..."""

    def __init__(self, checkpoint_dir):
        raise NotImplementedError('TODO: implement ShardedCheckpointManager.__init__; compare with solution.py only after trying.')

    def save(self, model_shards, metadata):
        """각 rank의 shard를 독립적으로 저장.

model_shards: {rank: {param_name: tensor}}
metadata: {param_name: {shape, dtype, tp_size, pp_size, ...}}"""
        raise NotImplementedError('TODO: implement ShardedCheckpointManager.save; compare with solution.py only after trying.')

    def load(self, target_tp_size, target_pp_size):
        """Resharding: 저장 시와 다른 TP/PP 설정으로 로드.

핵심 로직:
  TP 축소 (8→4): 인접한 2개 shard를 concat
  TP 확대 (4→8): 각 shard를 split
  PP 변경: layer 재배분"""
        raise NotImplementedError('TODO: implement ShardedCheckpointManager.load; compare with solution.py only after trying.')

def simulate_sharded_checkpoint():
    """Sharded checkpoint 저장/로드 + resharding 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_sharded_checkpoint; compare with solution.py only after trying.')

def async_checkpoint_concept():
    """비동기 체크포인트: 학습 중단 최소화."""
    raise NotImplementedError('TODO: implement async_checkpoint_concept; compare with solution.py only after trying.')

def megatron_checkpoint_guide():
    raise NotImplementedError('TODO: implement megatron_checkpoint_guide; compare with solution.py only after trying.')

def checkpoint_strategies():
    raise NotImplementedError('TODO: implement checkpoint_strategies; compare with solution.py only after trying.')
if __name__ == '__main__':
    naive_checkpoint_demo()
    simulate_sharded_checkpoint()
    async_checkpoint_concept()
    megatron_checkpoint_guide()
    checkpoint_strategies()
