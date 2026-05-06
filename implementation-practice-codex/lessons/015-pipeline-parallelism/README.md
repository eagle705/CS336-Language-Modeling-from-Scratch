# 015. Pipeline Parallelism

Focus: GPipe, 1F1B, stages, schedules, pipeline APIs

Source: `implementation-practice/02-mlp-parallelism/pipeline_parallelism.py`
원본 모듈 제목: `Pipeline Parallelism (PP) for MLP`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/015-pipeline-parallelism/starter.py
python implementation-practice-codex/lessons/015-pipeline-parallelism/solution.py
```

## TODO Surface

- class `PipelineStage` (__init__, forward)
- class `GPipeSimulator` (__init__, forward_backward, print_schedule)
- class `OneFOneBSimulator` (print_schedule)
- function `manual_p2p_pipeline_example`
- function `pipelining_example`
- function `demo`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
