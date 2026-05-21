# 030. Scaling Book Systems Math

Focus: Roofline estimates, Transformer FLOPs/bytes, training parallelism, inference KV cache

Source: [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)
Related chapters: [Rooflines](https://jax-ml.github.io/scaling-book/roofline/), [Transformer Math](https://jax-ml.github.io/scaling-book/transformers/), [Training Parallelism](https://jax-ml.github.io/scaling-book/training/), [Inference](https://jax-ml.github.io/scaling-book/inference/)

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and check each estimate against simple unit reasoning.
4. Open `solution.py` only after you have a working pass, then compare the accounting choices.
5. Write one short note in `../../PROGRESS.md`: which bottleneck or parallelism invariant this lesson clarified.

## Run

```bash
python3 implementation-practice-codex/lessons/030-scaling-book/starter.py
python3 implementation-practice-codex/lessons/030-scaling-book/solution.py
```

## Reading Map

- Roofline analysis: model time is bounded by math time, communication time, and memory capacity.
- Transformer accounting: dot products, matmuls, training FLOPs, activation memory, and KV cache size are all shape bookkeeping.
- Training parallelism: DP, FSDP/ZeRO, TP, and PP mainly differ in which tensors are replicated, sharded, gathered, or reduced.
- Inference: prefill is throughput-oriented and training-like; generation is latency-sensitive and often bound by parameter/KV-cache bandwidth.

## TODO Surface

- dataclass `RooflineEstimate`
- function `roofline_estimate`
- function `matmul_accounting`
- function `training_flops`
- function `kv_cache_bytes`
- function `decode_step_estimate`
- function `parallelism_hint`
- function `demo`

## Checkpoint Questions

- When can communication and math overlap, and when should you model their sum instead?
- Why does larger matmul size usually improve arithmetic intensity?
- Which state dominates inference memory as batch size and context length grow?
- Which parallelism axis would you keep within a node, and which one can tolerate slower inter-node links?
