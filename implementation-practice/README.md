# Implementation Practice

ML/DL 핵심 개념을 바닥부터 구현하는 연습 코드 모음.
모든 파일은 **GPU 없이 실행 가능**하며, 주석과 예시 입력으로 동작 원리를 단계별로 확인할 수 있습니다.

## How to Run

```bash
# venv 활성화
source .venv/bin/activate

# 아무 파일이나 바로 실행
python implementation-practice/01-backpropagation/backprop_from_scratch.py
```

## Topics

### Fundamentals

| # | Topic | Files | Description |
|---|-------|-------|-------------|
| 01 | [Backpropagation](01-backpropagation/) | `backprop_from_scratch.py` | NumPy MLP forward/backward, gradient check, XOR 학습 |
| 03 | [Attention](03-attention/) | `attention.py` | Scaled dot-product, MHA, GQA, Flash Attention, 성능 벤치마크 |
| 04 | [Transformer](04-transformer/) | `transformer.py` | GPT decoder-only (RMSNorm, RoPE, SwiGLU, weight tying) |
| 13 | [Tokenization & BPE](13-tokenization-bpe/) | `bpe.py` | BPE from scratch, byte-level BPE, SentencePiece 설명 |
| 14 | [PyTorch Fundamentals](14-pytorch-fundamentals/) | `pytorch_dl.py` | Autograd, hooks, torch.compile, initialization, LR schedule |
| 15 | [Python Algorithms](15-python-algorithms/) | `algorithms.py` | Top-K, softmax, beam search, LRU cache, Trie, topological sort |

### Parallelism

| # | Topic | Files | Description |
|---|-------|-------|-------------|
| 02 | [MLP Parallelism](02-mlp-parallelism/) | `mlp_baseline.py` | Single GPU baseline MLP |
| | | `tensor_parallelism.py` | TP: Column/Row Parallel + DTensor API + 시뮬레이션 |
| | | `pipeline_parallelism.py` | PP: GPipe, 1F1B schedule + torch.distributed.pipelining API |
| | | `data_parallelism.py` | DP: DDP 시뮬레이션, gradient accumulation |
| | | `context_parallelism.py` | CP: Ring Attention 구현 + causal mask 최적화 |
| | | `sequence_parallelism.py` | SP: TP+SP 통합, activation 메모리 절약 |
| | | `virtual_pipeline_parallelism.py` | VPP: Interleaved 1F1B, bubble 분석 |
| | | `tp_pp_dp_combined.py` | 3D parallelism (TP+PP+DP) 시뮬레이션 |
| | | `tp_cp_pp_dp_4d.py` | 4D parallelism (TP+CP+PP+DP) 전체 시뮬레이션 |

### Training Systems

| # | Topic | Files | Description |
|---|-------|-------|-------------|
| 05 | [Mixed Precision](05-mixed-precision/) | `mixed_precision.py` | FP32/FP16/BF16/FP8/NVFP4, AMP, loss scaling |
| 06 | [ZeRO & Dist Optimizer](06-zero-optimizer/) | `zero_1_2_3.py` | ZeRO Stage 1/2/3 시뮬레이션 |
| | | `distributed_optimizer.py` | Megatron Distributed Optimizer (ZeRO-1 최적화) |
| 07 | [FSDP](07-fsdp/) | `fsdp.py` | FSDP1/FSDP2 API, shard/gather 시뮬레이션 |
| 08 | [Megatron 3D](08-megatron-3d-parallelism/) | `megatron_3d.py` | 3D parallelism GPU 배치, process group, 전략 가이드 |
| 11 | [Memory Optimization](11-memory-optimization/) | `memory_optimization.py` | Activation 메모리 계산, gradient checkpointing |
| 16 | [Distributed Training](16-distributed-training/) | `distributed_training.py` | DDP, process groups, torchrun, multi-node |
| 17 | [Comm Overlaps](17-comm-overlaps/) | `comm_overlaps.py` | DDP bucketing, FSDP prefetch, CUDA streams |
| 19 | [Megatron-Core](19-megatron-core/) | `megatron_core_tutorial.py` | Vanilla→Megatron-Core 전환, API 가이드, 성능 벤치마크 |
| 21 | [Dist Checkpointing](21-distributed-checkpointing/) | `distributed_checkpointing.py` | Sharded checkpoint, resharding, async save |

### Model Architecture & Data

| # | Topic | Files | Description |
|---|-------|-------|-------------|
| 09 | [Mixture of Experts](09-mixture-of-experts/) | `mixture_of_experts.py` | Router, Expert, MoE layer, expert parallelism |
| 10 | [Long Context](10-long-context/) | `long_context.py` | RoPE scaling, sliding window, Ring Attention, KV cache |
| 12 | [Data Loading](12-data-loading/) | `data_loading.py` | Memmap dataset, streaming, DistributedSampler |
| 18 | [DL Systems](18-dl-systems/) | `dl_systems.py` | GPU specs, interconnect, MFU, training cost 추정 |

## Structure

```
implementation-practice/
├── 01-backpropagation/          # NumPy backprop
├── 02-mlp-parallelism/          # TP, PP, DP, CP, SP, VPP, 3D, 4D
├── 03-attention/                # MHA, GQA, Flash Attention
├── 04-transformer/              # GPT architecture
├── 05-mixed-precision/          # FP16/BF16/FP8/FP4
├── 06-zero-optimizer/           # ZeRO 1/2/3 + Distributed Optimizer
├── 07-fsdp/                     # FSDP1/FSDP2
├── 08-megatron-3d-parallelism/  # 3D parallelism overview
├── 09-mixture-of-experts/       # MoE
├── 10-long-context/             # RoPE, Ring Attention
├── 11-memory-optimization/      # Gradient checkpointing
├── 12-data-loading/             # DataLoader, streaming
├── 13-tokenization-bpe/         # BPE from scratch
├── 14-pytorch-fundamentals/     # Autograd, hooks, compile
├── 15-python-algorithms/        # Interview algorithms
├── 16-distributed-training/     # DDP, torchrun
├── 17-comm-overlaps/            # Compute-communication overlap
├── 18-dl-systems/               # GPU, MFU, cost
├── 19-megatron-core/            # Megatron-Core tutorial
└── 21-distributed-checkpointing/ # Sharded checkpoint
```
