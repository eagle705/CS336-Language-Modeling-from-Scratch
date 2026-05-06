# 017. Context Parallelism

Focus: Ring attention, CP communication types, 4D parallelism context

Source: `implementation-practice/02-mlp-parallelism/context_parallelism.py`
원본 모듈 제목: `Context Parallelism (CP)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/017-context-parallelism/starter.py
python implementation-practice-codex/lessons/017-context-parallelism/solution.py
```

## TODO Surface

- class `CPCommType`
- function `ring_attention`
- function `_cp_p2p`
- function `_cp_all_gather`
- function `_cp_a2a`
- function `verify_ring_attention`
- function `communication_analysis`
- function `parallelism_4d`
- function `cp_comparison`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
