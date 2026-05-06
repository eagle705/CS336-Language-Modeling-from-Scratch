# 024. Distributed Checkpointing

Focus: Sharded checkpointing, resharding, async save concepts

Source: `implementation-practice/21-distributed-checkpointing/distributed_checkpointing.py`
원본 모듈 제목: `Distributed Checkpointing`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/024-distributed-checkpointing/starter.py
python implementation-practice-codex/lessons/024-distributed-checkpointing/solution.py
```

## TODO Surface

- function `naive_checkpoint_demo`
- class `ShardedCheckpointManager` (__init__, save, load)
- function `simulate_sharded_checkpoint`
- function `async_checkpoint_concept`
- function `megatron_checkpoint_guide`
- function `checkpoint_strategies`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
