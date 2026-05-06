# 025. Megatron 3D Parallelism

Focus: Megatron-style 3D layout, process groups, strategy guide

Source: `implementation-practice/08-megatron-3d-parallelism/megatron_3d.py`
원본 모듈 제목: `Megatron-LM 3D Parallelism & Codebase Guide`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/025-megatron-3d-parallelism/starter.py
python implementation-practice-codex/lessons/025-megatron-3d-parallelism/solution.py
```

## TODO Surface

- function `simulate_3d_parallelism`
- function `parallelism_strategy_guide`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
