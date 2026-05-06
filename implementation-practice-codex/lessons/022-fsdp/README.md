# 022. FSDP

Focus: FSDP sharding/gathering and memory comparison

Source: `implementation-practice/07-fsdp/fsdp.py`
원본 모듈 제목: `FSDP (Fully Sharded Data Parallel)`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/022-fsdp/starter.py
python implementation-practice-codex/lessons/022-fsdp/solution.py

# CUDA GPU smoke test
python implementation-practice-codex/lessons/022-fsdp/solution.py fsdp1
python implementation-practice-codex/lessons/022-fsdp/solution.py fsdp2
torchrun --nproc_per_node=2 implementation-practice-codex/lessons/022-fsdp/solution.py fsdp1
torchrun --nproc_per_node=2 implementation-practice-codex/lessons/022-fsdp/solution.py fsdp2

# Megatron-FSDP API/flag reference
python implementation-practice-codex/lessons/022-fsdp/solution.py megatron
```

## TODO Surface

- function `simulate_fsdp`
- class `TinyFSDPBlock` (`__init__`, `forward`)
- class `TinyFSDPModel` (`__init__`, `forward`)
- function `_find_free_port`
- function `_init_cuda_dist`
- function `_average_loss_for_logging`
- function `run_fsdp_gpu_smoke_test`
- function `run_fsdp2_gpu_smoke_test`
- function `print_megatron_fsdp_reference`
- function `memory_comparison`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
