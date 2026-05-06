# 001. PyTorch Fundamentals

Focus: Autograd, custom Function, hooks, compile, initialization, LR schedule

Source: `implementation-practice/14-pytorch-fundamentals/pytorch_dl.py`
원본 모듈 제목: `PyTorch Deep Learning Fundamentals`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/001-pytorch-fundamentals/starter.py
python implementation-practice-codex/lessons/001-pytorch-fundamentals/solution.py
```

## TODO Surface

- function `autograd_internals`
- class `StraightThroughEstimator` (forward, backward)
- function `custom_function_demo`
- function `hooks_demo`
- function `initialization_demo`
- function `lr_schedule_demo`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
