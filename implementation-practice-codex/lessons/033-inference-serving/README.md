# 033. Inference Serving

Focus: TRT-LLM/vLLM/SGLang-style deployment tradeoffs, KV-cache accounting, and continuous batching

Source: NVIDIA GenAI Frameworks job-post gap exercise

This lesson complements the training-heavy lessons with inference and deployment
reasoning. The code is a CPU-only model of the serving decisions: what fits in
HBM, what bottlenecks prefill vs decode, and which backend shape matches a
workload.

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and sanity-check units: bytes, tokens/sec, and milliseconds.
4. Open `solution.py` only after you have a working pass, then compare the heuristics.
5. Write one short note in `../../PROGRESS.md`: what changes between prefill throughput and decode latency.

## Run

```bash
python implementation-practice-codex/lessons/033-inference-serving/starter.py
python implementation-practice-codex/lessons/033-inference-serving/solution.py
```

## Prior Lessons To Reuse

- `005-attention`: attention score and KV shape intuition
- `009-mixed-precision`: dtype memory tradeoffs
- `028-long-context`: KV cache and long-context implications
- `029-dl-systems`: HBM bandwidth, FLOPS, and GPU architecture basics
- `030-scaling-book`: decode-step scaling estimates

## TODO Surface

- dataclass `ModelServingSpec`
- dataclass `HardwareSpec`
- dataclass `RequestClass`
- function `kv_cache_bytes`
- function `weight_memory_bytes`
- function `max_batch_size_for_kv_cache`
- function `prefill_decode_estimate`
- function `continuous_batching_schedule`
- function `choose_serving_backend`
- function `demo`

## Checkpoint Questions

- Why is prefill usually compute-heavy while decode often becomes memory/KV-cache sensitive?
- How does GQA reduce KV-cache pressure?
- When would you prefer TensorRT-LLM, vLLM, or SGLang?
- Which serving metrics should be reported separately for latency and throughput?
