# 020. ZeRO Optimizer

Focus: ZeRO stages 1/2/3 state partitioning simulation

Source: `implementation-practice/06-zero-optimizer/zero_1_2_3.py`
원본 모듈 제목: `ZeRO (Zero Redundancy Optimizer) Stage 1 / 2 / 3`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/020-zero-optimizer/starter.py
python implementation-practice-codex/lessons/020-zero-optimizer/solution.py
```

## TODO Surface

- function `simulate_zero_stage1`
- function `simulate_zero_stage2`
- function `simulate_zero_stage3`
- function `comparison_table`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
