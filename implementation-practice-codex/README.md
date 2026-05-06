# Implementation Practice Codex

단계별 구현 연습 폴더입니다. 원본 `implementation-practice`는 완성 예제로 보존하고, 이 폴더는 하나씩 직접 채워 넣는 연습용으로 분리했습니다.

## How To Use

1. `lessons/001-*`부터 순서대로 들어갑니다.
2. 먼저 `starter.py`의 TODO를 직접 구현합니다.
3. 실행해서 shape, dtype, gradient, communication invariant를 확인합니다.
4. 마지막에만 `solution.py`와 비교합니다.
5. 통과한 lesson은 `PROGRESS.md`에 체크하고, 배운 invariant를 한 줄로 남깁니다.

`starter.py`는 의도적으로 `NotImplementedError`를 던집니다. 하나씩 구현하면서 에러를 제거하는 방식으로 쓰면 됩니다.

## Layout

```text
implementation-practice-codex/
  README.md
  PROGRESS.md
  MEGATRON_CORE_MAP.md
  lessons/
    001-pytorch-fundamentals/
      README.md
      starter.py
      solution.py
```

## Learning Path

| # | Lesson | Focus | Source |
|---|--------|-------|--------|
| 001 | [PyTorch Fundamentals](lessons/001-pytorch-fundamentals/) | Autograd, custom Function, hooks, compile, initialization, LR schedule | `implementation-practice/14-pytorch-fundamentals/pytorch_dl.py` |
| 002 | [Python Algorithms](lessons/002-python-algorithms/) | Core interview-style algorithms used around model systems work | `implementation-practice/15-python-algorithms/algorithms.py` |
| 003 | [Backpropagation from Scratch](lessons/003-backpropagation/) | Manual forward/backward, gradient check, XOR training | `implementation-practice/01-backpropagation/backprop_from_scratch.py` |
| 004 | [MLP Baseline](lessons/004-mlp-baseline/) | Single-process Transformer FFN baseline before parallelism | `implementation-practice/02-mlp-parallelism/mlp_baseline.py` |
| 005 | [Attention](lessons/005-attention/) | Scaled dot-product attention, MHA, GQA, Flash-style blocking | `implementation-practice/03-attention/attention.py` |
| 006 | [Transformer](lessons/006-transformer/) | Decoder-only GPT block: RMSNorm, RoPE, SwiGLU, tying | `implementation-practice/04-transformer/transformer.py` |
| 007 | [Tokenization and BPE](lessons/007-tokenization-bpe/) | Tokenizer/BPE practice slot; source file currently reflects repository contents | `implementation-practice/13-tokenization-bpe/bpe.py` |
| 008 | [Data Loading](lessons/008-data-loading/) | Memmap, streaming dataset, distributed sampling concepts | `implementation-practice/12-data-loading/data_loading.py` |
| 009 | [Mixed Precision](lessons/009-mixed-precision/) | FP32/FP16/BF16/FP8/FP4, AMP, scaling, memory tradeoffs | `implementation-practice/05-mixed-precision/mixed_precision.py` |
| 010 | [Memory Optimization](lessons/010-memory-optimization/) | Activation accounting, checkpointing, memory reduction tactics | `implementation-practice/11-memory-optimization/memory_optimization.py` |
| 011 | [Distributed Training](lessons/011-distributed-training/) | DDP, process groups, torchrun, multi-node concepts | `implementation-practice/16-distributed-training/distributed_training.py` |
| 012 | [Data Parallelism](lessons/012-data-parallelism/) | DDP simulation and gradient accumulation | `implementation-practice/02-mlp-parallelism/data_parallelism.py` |
| 013 | [Tensor Parallelism](lessons/013-tensor-parallelism/) | Column/row parallel linear layers and TP simulation | `implementation-practice/02-mlp-parallelism/tensor_parallelism.py` |
| 014 | [Sequence Parallelism](lessons/014-sequence-parallelism/) | TP+SP activation sharding and communication analysis | `implementation-practice/02-mlp-parallelism/sequence_parallelism.py` |
| 015 | [Pipeline Parallelism](lessons/015-pipeline-parallelism/) | GPipe, 1F1B, stages, schedules, pipeline APIs | `implementation-practice/02-mlp-parallelism/pipeline_parallelism.py` |
| 016 | [Virtual Pipeline Parallelism](lessons/016-virtual-pipeline-parallelism/) | Interleaved 1F1B and bubble analysis | `implementation-practice/02-mlp-parallelism/virtual_pipeline_parallelism.py` |
| 017 | [Context Parallelism](lessons/017-context-parallelism/) | Ring attention, CP communication types, 4D parallelism context | `implementation-practice/02-mlp-parallelism/context_parallelism.py` |
| 018 | [TP + PP + DP Combined](lessons/018-tp-pp-dp-combined/) | 3D process groups, communication, memory, strategy selection | `implementation-practice/02-mlp-parallelism/tp_pp_dp_combined.py` |
| 019 | [TP + CP + PP + DP 4D](lessons/019-tp-cp-pp-dp-4d/) | 4D process groups, CP+TP attention, TP FFN, setup guide | `implementation-practice/02-mlp-parallelism/tp_cp_pp_dp_4d.py` |
| 020 | [ZeRO Optimizer](lessons/020-zero-optimizer/) | ZeRO stages 1/2/3 state partitioning simulation | `implementation-practice/06-zero-optimizer/zero_1_2_3.py` |
| 021 | [Distributed Optimizer](lessons/021-distributed-optimizer/) | Megatron-style distributed optimizer memory and comm tradeoffs | `implementation-practice/06-zero-optimizer/distributed_optimizer.py` |
| 022 | [FSDP](lessons/022-fsdp/) | FSDP sharding/gathering and memory comparison | `implementation-practice/07-fsdp/fsdp.py` |
| 023 | [Communication Overlaps](lessons/023-comm-overlaps/) | DDP bucketing, FSDP prefetch, stream overlap concepts | `implementation-practice/17-comm-overlaps/comm_overlaps.py` |
| 024 | [Distributed Checkpointing](lessons/024-distributed-checkpointing/) | Sharded checkpointing, resharding, async save concepts | `implementation-practice/21-distributed-checkpointing/distributed_checkpointing.py` |
| 025 | [Megatron 3D Parallelism](lessons/025-megatron-3d-parallelism/) | Megatron-style 3D layout, process groups, strategy guide | `implementation-practice/08-megatron-3d-parallelism/megatron_3d.py` |
| 026 | [Megatron Core Tutorial](lessons/026-megatron-core/) | Vanilla to Megatron Core concepts, launch guide, benchmark sketch | `implementation-practice/19-megatron-core/megatron_core_tutorial.py` |
| 027 | [Mixture of Experts](lessons/027-mixture-of-experts/) | Top-k routing, expert layers, expert parallelism ideas | `implementation-practice/09-mixture-of-experts/mixture_of_experts.py` |
| 028 | [Long Context](lessons/028-long-context/) | RoPE scaling, sliding window attention, ring attention, KV cache | `implementation-practice/10-long-context/long_context.py` |
| 029 | [DL Systems](lessons/029-dl-systems/) | GPU/interconnect specs, MFU, throughput and cost estimation | `implementation-practice/18-dl-systems/dl_systems.py` |

## Recommended Cadence

- First pass: 구현을 완성하는 것보다 tensor shape와 state ownership을 말로 설명하는 데 집중합니다.
- Second pass: `solution.py`를 보고 더 짧고 안정적인 구현으로 고칩니다.
- Third pass: Megatron Core 용어로 다시 설명합니다. 예를 들어 TP shard, PP stage, CP ring, DP replica, distributed optimizer state처럼 이름 붙입니다.

## Notes

- 원본 파일은 그대로 `solution.py`에 복사했습니다.
- `13-tokenization-bpe/bpe.py`는 현재 원본 내용 기준으로 복사됩니다. 파일 경로명과 모듈 내용이 다르면 이 폴더도 그 상태를 그대로 반영합니다.
- GPU 없이 읽고 실행할 수 있는 시뮬레이션 중심이지만, 실제 multi-process API가 필요한 lesson은 README의 checkpoint 질문을 먼저 답한 뒤 실행하세요.
