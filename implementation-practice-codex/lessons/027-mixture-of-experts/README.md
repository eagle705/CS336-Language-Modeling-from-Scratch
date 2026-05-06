# 027. Mixture of Experts

Focus: Top-k routing, expert layers, expert parallelism ideas

Source: `implementation-practice/09-mixture-of-experts/mixture_of_experts.py`
원본 모듈 제목: `Mixture of Experts (MoE)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/027-mixture-of-experts/starter.py
python implementation-practice-codex/lessons/027-mixture-of-experts/solution.py
```

## TODO Surface

- class `TopKRouter` (__init__, forward)
- class `Expert` (__init__, forward)
- class `MoELayer` (__init__, forward)
- function `demo`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
