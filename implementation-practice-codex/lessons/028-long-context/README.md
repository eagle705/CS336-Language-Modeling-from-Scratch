# 028. Long Context

Focus: RoPE scaling, sliding window attention, ring attention, KV cache

Source: `implementation-practice/10-long-context/long_context.py`
원본 모듈 제목: `Long Context Techniques`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/028-long-context/starter.py
python implementation-practice-codex/lessons/028-long-context/solution.py
```

## TODO Surface

- function `rope_scaling_demo`
- function `sliding_window_attention`
- function `simulate_ring_attention`
- function `kv_cache_analysis`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
