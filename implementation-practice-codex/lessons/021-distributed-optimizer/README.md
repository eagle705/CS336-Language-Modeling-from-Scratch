# 021. Distributed Optimizer

Focus: Megatron-style distributed optimizer memory and comm tradeoffs

Source: `implementation-practice/06-zero-optimizer/distributed_optimizer.py`
원본 모듈 제목: `Distributed Optimizer`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/021-distributed-optimizer/starter.py
python implementation-practice-codex/lessons/021-distributed-optimizer/solution.py
```

## TODO Surface

- function `simulate_distributed_optimizer`
- function `communication_comparison`
- function `megatron_dist_optimizer`
- function `memory_analysis`
- function `comparison_with_zero`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
