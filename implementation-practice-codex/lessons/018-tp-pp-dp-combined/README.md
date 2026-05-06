# 018. TP + PP + DP Combined

Focus: 3D process groups, communication, memory, strategy selection

Source: `implementation-practice/02-mlp-parallelism/tp_pp_dp_combined.py`
원본 모듈 제목: `TP + PP + DP Combined (3D Parallelism)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/018-tp-pp-dp-combined/starter.py
python implementation-practice-codex/lessons/018-tp-pp-dp-combined/solution.py
```

## TODO Surface

- function `build_process_groups`
- function `simulate_3d_parallelism`
- function `communication_analysis`
- function `memory_analysis`
- function `strategy_guide`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
