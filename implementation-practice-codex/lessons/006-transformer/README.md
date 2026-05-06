# 006. Transformer

Focus: Decoder-only GPT block: RMSNorm, RoPE, SwiGLU, tying

Source: `implementation-practice/04-transformer/transformer.py`
원본 모듈 제목: `Transformer Architecture (GPT-style decoder-only)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/006-transformer/starter.py
python implementation-practice-codex/lessons/006-transformer/solution.py
```

## TODO Surface

- class `RMSNorm` (__init__, forward)
- class `RotaryPositionalEmbedding` (__init__, forward)
- class `FeedForward` (__init__, forward)
- class `TransformerBlock` (__init__, forward)
- class `CausalSelfAttention` (__init__, forward)
- class `GPT` (__init__, forward)
- function `demo`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
