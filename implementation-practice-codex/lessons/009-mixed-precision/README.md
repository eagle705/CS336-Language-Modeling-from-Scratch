# 009. Mixed Precision

Focus: FP32/FP16/BF16/FP8/FP4, AMP, scaling, memory tradeoffs

Source: `implementation-practice/05-mixed-precision/mixed_precision.py`
원본 모듈 제목: `Mixed Precision Training`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/009-mixed-precision/starter.py
python implementation-practice-codex/lessons/009-mixed-precision/solution.py
```

## TODO Surface

- function `explore_dtypes`
- function `manual_mixed_precision`
- function `pytorch_amp_example`
- function `memory_analysis`
- function `fp8_info`
- function `fp4_info`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
