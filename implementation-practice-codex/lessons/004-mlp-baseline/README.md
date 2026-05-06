# 004. MLP Baseline

Focus: Single-process Transformer FFN baseline before parallelism

Source: `implementation-practice/02-mlp-parallelism/mlp_baseline.py`
원본 모듈 제목: `MLP Baseline (Single GPU)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/004-mlp-baseline/starter.py
python implementation-practice-codex/lessons/004-mlp-baseline/solution.py
```

## TODO Surface

- class `MLPBlock` (__init__, forward)
- class `SimpleTransformerMLP` (__init__, forward)
- function `count_params`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
