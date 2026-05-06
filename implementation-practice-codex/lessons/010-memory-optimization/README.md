# 010. Memory Optimization

Focus: Activation accounting, checkpointing, memory reduction tactics

Source: `implementation-practice/11-memory-optimization/memory_optimization.py`
원본 모듈 제목: `Memory Optimization`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/010-memory-optimization/starter.py
python implementation-practice-codex/lessons/010-memory-optimization/solution.py
```

## TODO Surface

- function `activation_memory_analysis`
- class `CheckpointedBlock` (__init__, _inner, forward)
- function `demo_gradient_checkpointing`
- function `other_optimizations`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
