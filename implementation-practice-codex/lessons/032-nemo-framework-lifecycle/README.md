# 032. NeMo Framework Lifecycle

Focus: NeMo-style lifecycle orchestration, config/API boundaries, and reproducible GenAI training plans

Source: NVIDIA GenAI Frameworks job-post gap exercise

This lesson ties earlier systems pieces into the end-to-end shape expected by a
Megatron Core and NeMo Framework engineer: preprocessing, pretraining, alignment,
evaluation, checkpoint resharding, and deployment handoff.

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and check that invalid configs fail with concrete reasons.
4. Open `solution.py` only after you have a working pass, then compare the API choices.
5. Write one short note in `../../PROGRESS.md`: which lifecycle boundary should be stable across internal users and open-source users.

## Run

```bash
python implementation-practice-codex/lessons/032-nemo-framework-lifecycle/starter.py
python implementation-practice-codex/lessons/032-nemo-framework-lifecycle/solution.py
```

## Prior Lessons To Reuse

- `008-data-loading`: preprocessing, memmap/streaming, sampler ownership
- `009-mixed-precision`: BF16/FP8 recipes and numerical safety checks
- `018-tp-pp-dp-combined`: 3D process group layout
- `019-tp-cp-pp-dp-4d`: adding context parallelism to the plan
- `024-distributed-checkpointing`: sharded checkpoint and resharding contract
- `026-megatron-core`: mapping vanilla modules to Megatron Core concepts

## TODO Surface

- dataclass `ModelRecipe`
- dataclass `ParallelRecipe`
- dataclass `LifecycleStage`
- function `estimate_dense_parameters`
- function `derive_gradient_accumulation_steps`
- function `validate_parallel_recipe`
- function `build_lifecycle_plan`
- function `build_nemo_launcher`
- function `checkpoint_compatibility_report`
- function `demo`

## Checkpoint Questions

- Which fields belong in a stable user-facing recipe, and which should stay internal?
- Which invariants should fail fast before a multi-node launch starts?
- When does a checkpoint need resharding instead of direct resume?
- How would you explain the same plan to a model researcher, a systems engineer, and a framework maintainer?
