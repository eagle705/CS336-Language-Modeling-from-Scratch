# Megatron Core Map

이 연습 폴더를 Megatron Core 엔지니어링 역량으로 다시 묶은 지도입니다.

## Core PyTorch and Numerics

- `001-pytorch-fundamentals`: autograd, hooks, custom backward, compile boundaries
- `003-backpropagation`: chain rule and gradient checking
- `009-mixed-precision`: dtype, AMP, loss scaling, low precision tradeoffs

## Model Architecture

- `004-mlp-baseline`: FFN baseline used before model parallel decomposition
- `005-attention`: MHA/GQA/Flash-style attention mechanics
- `006-transformer`: GPT block composition and parameter tying
- `027-mixture-of-experts`: router, expert dispatch, expert parallel intuition
- `028-long-context`: RoPE scaling, sliding window, ring attention, KV cache

## Parallelism

- `011-distributed-training`: process groups and DDP mental model
- `012-data-parallelism`: replicated parameters, gradient synchronization, accumulation
- `013-tensor-parallelism`: column/row parallel linear layers
- `014-sequence-parallelism`: activation sharding along sequence dimension
- `015-pipeline-parallelism`: stage ownership and microbatch scheduling
- `016-virtual-pipeline-parallelism`: interleaving and bubble reduction
- `017-context-parallelism`: sequence/context partitioning and ring communication
- `018-tp-pp-dp-combined`: 3D parallel process group layout
- `019-tp-cp-pp-dp-4d`: 4D parallel composition
- `025-megatron-3d-parallelism`: Megatron-style strategy selection
- `026-megatron-core`: mapping vanilla modules to Megatron Core concepts

## Optimizer, Memory, and Checkpointing

- `010-memory-optimization`: activation memory and recomputation
- `020-zero-optimizer`: optimizer/gradient/parameter state partitioning
- `021-distributed-optimizer`: Megatron distributed optimizer tradeoffs
- `022-fsdp`: parameter sharding and gather/scatter model
- `023-comm-overlaps`: compute/communication overlap patterns
- `024-distributed-checkpointing`: sharded checkpoint and resharding

## Data and Systems

- `002-python-algorithms`: implementation fluency for reviews/interviews
- `007-tokenization-bpe`: tokenizer practice slot from the current source file
- `008-data-loading`: memmap, streaming, sampler ownership
- `029-dl-systems`: GPU specs, interconnect, MFU, throughput, cost
