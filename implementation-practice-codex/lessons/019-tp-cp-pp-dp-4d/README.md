# 019. TP + CP + PP + DP 4D

Focus: 4D process groups, CP+TP attention, TP FFN, setup guide

Source: `implementation-practice/02-mlp-parallelism/tp_cp_pp_dp_4d.py`
원본 모듈 제목: `4D Parallelism: TP × CP × PP × DP`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/019-tp-cp-pp-dp-4d/starter.py
python implementation-practice-codex/lessons/019-tp-cp-pp-dp-4d/solution.py
```

## TODO Surface

- function `build_4d_process_groups`
- function `ring_attention_with_tp`
- function `ffn_with_tp`
- function `simulate_4d_forward`
- function `communication_summary`
- function `detailed_analysis`
- function `setup_guide`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
