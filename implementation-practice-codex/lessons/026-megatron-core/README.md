# 026. Megatron Core Tutorial

Focus: Vanilla to Megatron Core concepts, launch guide, benchmark sketch

Source: `implementation-practice/19-megatron-core/megatron_core_tutorial.py`
원본 모듈 제목: `Megatron-Core Tutorial`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/026-megatron-core/starter.py
python implementation-practice-codex/lessons/026-megatron-core/solution.py
```

## TODO Surface

- class `VanillaMLP` (__init__, forward)
- function `comparison`
- function `codebase_guide`
- function `launch_guide`
- function `simulate_mcore_mlp`
- function `benchmark_vanilla_vs_tp_split`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
