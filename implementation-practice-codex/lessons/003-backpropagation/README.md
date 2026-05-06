# 003. Backpropagation from Scratch

Focus: Manual forward/backward, gradient check, XOR training

Source: `implementation-practice/01-backpropagation/backprop_from_scratch.py`
원본 모듈 제목: `Backpropagation from Scratch (NumPy only)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/003-backpropagation/starter.py
python implementation-practice-codex/lessons/003-backpropagation/solution.py
```

## TODO Surface

- function `step_by_step_example`
- class `ManualMLP` (__init__, forward, mse_loss, backward, update)
- function `gradient_check`
- function `train_xor`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
