# 016. Virtual Pipeline Parallelism

Focus: Interleaved 1F1B and bubble analysis

Source: `implementation-practice/02-mlp-parallelism/virtual_pipeline_parallelism.py`
원본 모듈 제목: `Virtual Pipeline Parallelism (VPP)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/016-virtual-pipeline-parallelism/starter.py
python implementation-practice-codex/lessons/016-virtual-pipeline-parallelism/solution.py
```

## TODO Surface

- function `visualize_schedules`
- function `simulate_vpp_layer_assignment`
- function `simulate_vpp_forward`
- function `bubble_analysis`
- function `memory_impact`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
