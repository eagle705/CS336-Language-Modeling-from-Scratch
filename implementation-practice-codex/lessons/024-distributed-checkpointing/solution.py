"""
Distributed Checkpointing
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
"""

import torch
import torch.nn as nn
import os
import json
import shutil
from pathlib import Path


# ============================================================
# Part 1: Naive Checkpoint (문제점)
# ============================================================

def naive_checkpoint_demo():
    """단순 torch.save의 문제점."""
    print("=" * 60)
    print("Naive Checkpoint Problems")
    print("=" * 60)

    print("""
  torch.save(model.state_dict(), "model.pt")  ← 단일 GPU에서는 OK

  분산 학습에서의 문제:
    1. 전체 모델을 rank 0에 모아서 저장?
       → 수백 GB 모델이 rank 0 메모리에 안 들어감 (OOM)

    2. 각 rank가 자기 shard를 각각 저장?
       → 저장 시 TP/PP 설정이 달라지면 로드 불가
       → 예: TP=8로 저장한 weight를 TP=4로 로드하려면?

    3. 저장 시간:
       → 405B 모델 = ~800GB (BF16)
       → 네트워크 스토리지에 저장 시 수 분 소요
       → 학습 중단 = GPU 수천 개가 idle = 엄청난 비용!
    """)


# ============================================================
# Part 2: Sharded Checkpoint 시뮬레이션
# ============================================================

class ShardedCheckpointManager:
    """
    Sharded checkpoint: 각 rank가 자기 shard를 독립 저장.

    디렉토리 구조:
      checkpoint/
        metadata.json       ← 전체 구조 정보 (shape, dtype, sharding 방식)
        rank_0.pt           ← rank 0의 shard
        rank_1.pt
        ...
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(self, model_shards, metadata):
        """
        각 rank의 shard를 독립적으로 저장.

        model_shards: {rank: {param_name: tensor}}
        metadata: {param_name: {shape, dtype, tp_size, pp_size, ...}}
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 메타데이터 저장 (rank 0만)
        meta_path = self.checkpoint_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 각 rank의 shard 저장 (병렬 가능)
        for rank, shard in model_shards.items():
            shard_path = self.checkpoint_dir / f"rank_{rank}.pt"
            torch.save(shard, shard_path)

        print(f"  Saved {len(model_shards)} shards to {self.checkpoint_dir}/")

    def load(self, target_tp_size, target_pp_size):
        """
        Resharding: 저장 시와 다른 TP/PP 설정으로 로드.

        핵심 로직:
          TP 축소 (8→4): 인접한 2개 shard를 concat
          TP 확대 (4→8): 각 shard를 split
          PP 변경: layer 재배분
        """
        meta_path = self.checkpoint_dir / "metadata.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # 모든 shard 로드
        all_shards = {}
        for shard_file in sorted(self.checkpoint_dir.glob("rank_*.pt")):
            rank = int(shard_file.stem.split("_")[1])
            all_shards[rank] = torch.load(shard_file, weights_only=True)

        return all_shards, metadata


def simulate_sharded_checkpoint():
    """Sharded checkpoint 저장/로드 + resharding 시뮬레이션."""
    print("\n" + "=" * 60)
    print("Sharded Checkpoint Simulation")
    print("=" * 60)

    torch.manual_seed(42)

    # --- 모델: 2 layers, TP=2로 학습 ---
    embed_dim = 8
    ffn_hidden = 16
    tp_size = 2
    num_layers = 2

    # 전체 weight
    W1_full = [torch.randn(embed_dim, ffn_hidden) for _ in range(num_layers)]
    W2_full = [torch.randn(ffn_hidden, embed_dim) for _ in range(num_layers)]

    # TP=2로 split된 shard
    half = ffn_hidden // tp_size
    rank0_shard = {}
    rank1_shard = {}
    for i in range(num_layers):
        rank0_shard[f"layer{i}.W1"] = W1_full[i][:, :half]
        rank0_shard[f"layer{i}.W2"] = W2_full[i][:half, :]
        rank1_shard[f"layer{i}.W1"] = W1_full[i][:, half:]
        rank1_shard[f"layer{i}.W2"] = W2_full[i][half:, :]

    metadata = {}
    for i in range(num_layers):
        metadata[f"layer{i}.W1"] = {
            "full_shape": list(W1_full[i].shape),
            "shard_dim": 1,  # column split
            "tp_size": tp_size,
        }
        metadata[f"layer{i}.W2"] = {
            "full_shape": list(W2_full[i].shape),
            "shard_dim": 0,  # row split
            "tp_size": tp_size,
        }

    # --- 저장 ---
    ckpt_dir = "/tmp/test_sharded_ckpt"
    manager = ShardedCheckpointManager(ckpt_dir)
    manager.save({0: rank0_shard, 1: rank1_shard}, metadata)

    print(f"\n  저장 완료 (TP={tp_size}):")
    print(f"    Rank 0 shard: {list(rank0_shard.keys())}")
    for k, v in rank0_shard.items():
        print(f"      {k}: {list(v.shape)}")

    # --- Resharding: TP=2 → TP=1 (single GPU로 합치기) ---
    print(f"\n  Resharding: TP={tp_size} → TP=1")
    shards, meta = manager.load(target_tp_size=1, target_pp_size=1)

    # 합치기
    for i in range(num_layers):
        key_w1 = f"layer{i}.W1"
        key_w2 = f"layer{i}.W2"
        info_w1 = meta[key_w1]
        info_w2 = meta[key_w2]

        # W1: column split → dim=1로 concat
        W1_restored = torch.cat([shards[r][key_w1] for r in range(tp_size)],
                                dim=info_w1["shard_dim"])
        # W2: row split → dim=0으로 concat
        W2_restored = torch.cat([shards[r][key_w2] for r in range(tp_size)],
                                dim=info_w2["shard_dim"])

        diff_w1 = (W1_full[i] - W1_restored).abs().max().item()
        diff_w2 = (W2_full[i] - W2_restored).abs().max().item()
        print(f"    Layer {i}: W1 diff={diff_w1:.2e}, W2 diff={diff_w2:.2e}")

    print(f"    Result: PASSED")

    # --- Resharding: TP=2 → TP=4 (더 잘게 split) ---
    print(f"\n  Resharding: TP={tp_size} → TP=4")
    target_tp = 4
    quarter = ffn_hidden // target_tp
    for i in range(num_layers):
        W1_restored = torch.cat([shards[r][f"layer{i}.W1"] for r in range(tp_size)], dim=1)
        new_shards = [W1_restored[:, j * quarter:(j + 1) * quarter] for j in range(target_tp)]
        print(f"    Layer {i} W1: {list(W1_restored.shape)} → "
              f"{target_tp} shards of {list(new_shards[0].shape)}")

    # Cleanup
    shutil.rmtree(ckpt_dir)


# ============================================================
# Part 3: Async Checkpoint
# ============================================================

def async_checkpoint_concept():
    """비동기 체크포인트: 학습 중단 최소화."""
    print("\n" + "=" * 60)
    print("Async Checkpoint")
    print("=" * 60)

    print("""
  문제: 동기 저장은 학습을 중단시킴
    Step 1000: [train][train][train][ SAVE (10s) ][train][train]
                                     ↑ GPU idle!

  해결: 비동기 저장
    1. State dict를 CPU 메모리에 복사 (빠름, ~1s)
    2. Background thread에서 디스크에 저장 (학습과 병렬)

    Step 1000: [train][copy→CPU][train][train][train]
                         ↓
               [Background: write to disk...........]

  구현 패턴:
    import threading

    def async_save(state_dict, path):
        # 1. GPU → CPU 복사 (동기, 빠름)
        cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}

        # 2. Background에서 저장
        def _save():
            torch.save(cpu_state, path)
        thread = threading.Thread(target=_save)
        thread.start()
        return thread  # 필요시 thread.join()으로 완료 대기

  PyTorch 2.x: torch.distributed.checkpoint (DCP)
    from torch.distributed.checkpoint import save, load
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, set_model_state_dict,
        get_optimizer_state_dict, set_optimizer_state_dict,
    )

    # 저장: 각 rank가 자기 shard를 병렬 저장
    state_dict = get_model_state_dict(model)
    save(state_dict, checkpoint_id="step_1000")

    # 로드: resharding 자동 지원!
    state_dict = get_model_state_dict(model)  # 현재 model의 sharding 기준
    load(state_dict, checkpoint_id="step_1000")
    set_model_state_dict(model, state_dict)
    # → 저장 시 TP=8, 로드 시 TP=4여도 자동 resharding!
    """)


# ============================================================
# Part 4: Megatron-Core Distributed Checkpoint
# ============================================================

def megatron_checkpoint_guide():
    print("\n" + "=" * 60)
    print("Megatron-Core Distributed Checkpoint")
    print("=" * 60)

    print("""
  Megatron-Core의 자체 distributed checkpoint 시스템:

  --- 저장 ---
  from megatron.core.dist_checkpointing import save, load
  from megatron.core.dist_checkpointing.mapping import ShardedTensor

  # ShardedTensor: 각 param의 sharding 정보를 명시
  state_dict = {
      'weight': ShardedTensor.from_rank_offsets(
          key='model.layers.0.mlp.fc1.weight',
          data=local_weight,                    # 이 rank의 shard
          replica_id=dp_rank,                   # DP 차원 (중복)
          prepend_axis_num=0,                   # PP 차원
          *rank_offsets,                        # TP 차원 offset
      ),
  }
  save(state_dict, checkpoint_dir)

  --- 로드 (resharding) ---
  # 새로운 parallelism 설정으로 모델 생성
  new_model = build_model(tp=4, pp=2)  # 다른 설정
  state_dict = new_model.sharded_state_dict()
  # load가 자동으로 shard를 재구성!
  load(state_dict, checkpoint_dir)

  --- 디렉토리 구조 ---
  checkpoint/
    common.pt              # 비-sharded 데이터 (step, lr 등)
    metadata.json          # shard 매핑 정보
    shard_0_of_8.pt       # TP rank 0
    shard_1_of_8.pt       # TP rank 1
    ...

  핵심: ShardedTensor가 (global_shape, local_offset, local_shape)를 저장
        → 어떤 TP/PP 설정으로든 로드 가능
    """)


# ============================================================
# Part 5: Checkpoint 전략 비교
# ============================================================

def checkpoint_strategies():
    print("\n" + "=" * 60)
    print("Checkpoint Strategy Comparison")
    print("=" * 60)

    print("""
  ┌─────────────────┬──────────────┬──────────────┬──────────────┐
  │ 방법             │ 저장 시간     │ Resharding   │ 구현         │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ torch.save      │ 느림 (동기)   │ 불가         │ 1줄          │
  │ (rank 0 수집)   │ OOM 위험     │              │              │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ 각 rank 독립    │ 빠름 (병렬)   │ 수동 구현    │ 간단         │
  │ 저장            │              │ 필요         │              │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ DCP             │ 빠름 (병렬)   │ 자동         │ PyTorch 2.x  │
  │ (torch.dist.ckpt)│ async 지원  │              │              │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Megatron dist   │ 빠름 (병렬)   │ 자동         │ Megatron     │
  │ checkpoint      │ async 지원   │ ShardedTensor│ 전용         │
  └─────────────────┴──────────────┴──────────────┴──────────────┘

  실전 팁:
    1. 매 N step마다 저장 (예: 500 steps)
    2. Async로 학습 중단 최소화
    3. 최신 K개만 유지, 나머지 삭제 (디스크 관리)
    4. Resharding 테스트: 저장한 ckpt를 다른 설정으로 로드 확인
    """)


if __name__ == "__main__":
    naive_checkpoint_demo()
    simulate_sharded_checkpoint()
    async_checkpoint_concept()
    megatron_checkpoint_guide()
    checkpoint_strategies()
