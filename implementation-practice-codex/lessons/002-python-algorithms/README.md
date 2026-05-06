# 002. Python Algorithms

Focus: Core interview-style algorithms used around model systems work

Source: `implementation-practice/15-python-algorithms/algorithms.py`
원본 모듈 제목: `Python Algorithms for ML Interviews`

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and fix shape, dtype, and numerical issues.
4. Open `solution.py` only after you have a working pass, then compare the implementation choices.
5. Write one short note in `../../PROGRESS.md`: what invariant or Megatron-Core concept this lesson clarified.

## Run

```bash
python implementation-practice-codex/lessons/002-python-algorithms/starter.py
python implementation-practice-codex/lessons/002-python-algorithms/solution.py
```

## TODO Surface

- function `topk_without_sort`
- function `weighted_reservoir_sampling`
- function `softmax`
- function `cross_entropy_loss`
- function `beam_search`
- function `binary_search`
- function `merge_sort`
- class `LRUCache` (__init__, get, put)
- class `Trie` (__init__, insert, search, starts_with)
- function `topological_sort`
- function `demo`

## Checkpoint Questions

- What tensors or states are partitioned, replicated, or recomputed in this lesson?
- Which shapes must stay invariant across the forward/backward path?
- What would break first if this code moved from a single process simulation to real multi-GPU execution?
