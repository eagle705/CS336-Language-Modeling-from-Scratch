# 008. Data Loading

Focus: Memmap, streaming dataset, distributed sampling concepts

Source: `implementation-practice/12-data-loading/data_loading.py`
원본 모듈 제목: `Data Loading for LLM Training`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/008-data-loading/starter.py
python implementation-practice-codex/lessons/008-data-loading/solution.py
```

## TODO Surface

- class `MemmapTokenDataset` (__init__, __len__, __getitem__)
- class `StreamingTokenDataset` (__init__, __iter__)
- function `demo`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
