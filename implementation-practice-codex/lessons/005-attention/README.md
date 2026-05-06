# 005. Attention

Focus: Scaled dot-product attention, MHA, GQA, Flash-style blocking

Source: `implementation-practice/03-attention/attention.py`
원본 모듈 제목: `Attention Mechanisms`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/005-attention/starter.py
python implementation-practice-codex/lessons/005-attention/solution.py
```

## TODO Surface

- function `scaled_dot_product_attention`
- class `MultiHeadAttention` (__init__, forward)
- class `GroupedQueryAttention` (__init__, forward)
- function `flash_attention_minimal`
- function `demo`
- function `benchmark`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
