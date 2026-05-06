# 011. Distributed Training

Focus: DDP, process groups, torchrun, multi-node concepts

Source: `implementation-practice/16-distributed-training/distributed_training.py`
원본 모듈 제목: `Distributed Training`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/011-distributed-training/starter.py
python implementation-practice-codex/lessons/011-distributed-training/solution.py
```

## TODO Surface

- function `simulate_ddp`
- function `multinode_guide`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
