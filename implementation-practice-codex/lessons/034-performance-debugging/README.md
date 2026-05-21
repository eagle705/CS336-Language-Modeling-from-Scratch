# 034. Performance Debugging

Focus: profiler-driven bottleneck classification, exposed communication, and tuning actions

Source: NVIDIA GenAI Frameworks job-post gap exercise

This lesson turns performance claims into executable checks. Given synthetic
profiler events, it estimates utilization, exposed communication, bottleneck
classes, and concrete tuning actions. It is meant to connect `023`,
`029`, and `030` into an engineering review workflow.

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and check whether the recommendations follow from the numbers.
4. Open `solution.py` only after you have a working pass, then compare the thresholds.
5. Write one short note in `../../PROGRESS.md`: which performance metric you would put in CI.

## Run

```bash
python implementation-practice-codex/lessons/034-performance-debugging/starter.py
python implementation-practice-codex/lessons/034-performance-debugging/solution.py
```

## Prior Lessons To Reuse

- `009-mixed-precision`: Tensor Core and dtype assumptions
- `010-memory-optimization`: activation memory and recompute tradeoffs
- `023-comm-overlaps`: communication overlap patterns
- `029-dl-systems`: MFU, HBM, interconnect, and profiler concepts
- `030-scaling-book`: roofline estimates and training/inference math

## TODO Surface

- dataclass `ProfilerEvent`
- dataclass `StepTrace`
- function `union_interval_length`
- function `exposed_communication`
- function `summarize_step`
- function `classify_bottlenecks`
- function `recommend_tuning_actions`
- function `compare_runs`
- function `demo`

## Checkpoint Questions

- What is the difference between high communication volume and exposed communication?
- When is low MFU caused by input pipeline, kernel launch overhead, HBM bandwidth, or collectives?
- Which recommendations are safe general advice, and which require another measurement?
- How would you keep a performance fix from regressing in CI?
