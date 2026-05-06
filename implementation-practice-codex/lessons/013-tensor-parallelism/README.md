# 013. Tensor Parallelism

Focus: Column/row parallel linear layers and TP simulation

Source: `implementation-practice/02-mlp-parallelism/tensor_parallelism.py`
원본 모듈 제목: `Tensor Parallelism (TP) for MLP`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/013-tensor-parallelism/starter.py
python implementation-practice-codex/lessons/013-tensor-parallelism/solution.py
```

## TODO Surface

- class `_IdentityFwd_AllreduceGradBwd` (forward, backward)
- class `_AllreduceSumFwd_IdentityBwd` (forward, backward)
- class `ColumnParallelLinear` (__init__, forward)
- class `RowParallelLinear` (__init__, forward)
- class `TensorParallelMLP` (__init__, forward)
- function `init_dist_env_or_notebook_single_process`
- function `distributed_tp_example`
- function `simulate_tensor_parallelism`
- function `step_by_step_tensor_parallelism`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
