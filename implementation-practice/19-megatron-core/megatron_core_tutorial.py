"""
Megatron-Core Tutorial
========================
기존 PyTorch MLP를 Megatron-Core의 TP 적용 MLP로 대체하는 과정.

Megatron-Core: NVIDIA의 대규모 모델 학습 라이브러리.
Megatron-LM에서 핵심 로직을 분리한 재사용 가능한 라이브러리.

핵심 특징:
  - ColumnParallelLinear / RowParallelLinear: TP가 내장된 Linear
  - TransformerConfig: 모델 설정을 하나의 config 객체로 관리
  - ModuleSpec 패턴: 어떤 구현체(mcore local / Transformer Engine)를 쓸지 선택
  - 입력 텐서 형태: [seq, batch, hidden] (HuggingFace와 다름!)

설치:
  pip install megatron-core   # 또는 NVIDIA/Megatron-LM repo에서 직접
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: 기존 PyTorch MLP (대체 대상)
# ============================================================

class VanillaMLP(nn.Module):
    """
    일반 PyTorch MLP. 이걸 Megatron-Core 버전으로 바꿀 것.

    구조: Linear(hidden → 4*hidden) → GELU → Linear(4*hidden → hidden)
    """

    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, seq, hidden) — PyTorch 일반 형태
        return self.fc2(self.act(self.fc1(x)))


# ============================================================
# Part 2: Megatron-Core TP Linear 핵심 API
# ============================================================
#
# --- ColumnParallelLinear ---
# FC1에 해당. output dim을 TP rank들에 나눔.
#
# from megatron.core.tensor_parallel import ColumnParallelLinear
#
# col_linear = ColumnParallelLinear(
#     input_size=1024,         # 입력 차원
#     output_size=4096,        # 출력 차원 (TP로 자동 분할됨)
#     config=config,           # ModelParallelConfig (TP size 등)
#     init_method=init_method_normal(0.02),  # weight 초기화
#     bias=False,
#     gather_output=False,     # True면 all-gather로 전체 output 복원
#                              # False면 split 상태 유지 (→ RowParallel에 전달)
#     skip_bias_add=True,      # bias를 별도 반환 (kernel fusion용)
# )
#
# # forward: (seq, batch, hidden) → ((seq, batch, hidden//tp), bias)
# output, bias = col_linear(x)
#
#
# --- RowParallelLinear ---
# FC2에 해당. input dim을 TP rank들에 나눔 + all-reduce.
#
# from megatron.core.tensor_parallel import RowParallelLinear
#
# row_linear = RowParallelLinear(
#     input_size=4096,         # 입력 차원 (TP로 자동 분할됨)
#     output_size=1024,        # 출력 차원
#     config=config,
#     init_method=init_method_normal(0.02),
#     bias=False,
#     input_is_parallel=True,  # 입력이 이미 TP rank에 split된 상태
#                              # (ColumnParallel의 gather_output=False 출력)
#     skip_bias_add=True,
# )
#
# # forward: (seq, batch, hidden//tp) → ((seq, batch, hidden), bias)
# # 내부에서 all-reduce 수행!
# output, bias = row_linear(x)
#
#
# 핵심 포인트:
#   - gather_output=False + input_is_parallel=True 조합이 표준 패턴
#     → Column과 Row 사이에서 activation이 split 상태로 유지
#     → all-reduce는 RowParallel 끝에서 1번만!
#   - 반환값이 (output, bias) 튜플. skip_bias_add=True면 bias 별도 반환.
#   - 입력 shape: [seq, batch, hidden] (batch first 아님!)


# ============================================================
# Part 3: Megatron-Core MLP 사용법
# ============================================================
#
# megatron.core.transformer.mlp.MLP: TP가 내장된 MLP 모듈.
# 직접 ColumnParallel/RowParallel을 조립할 필요 없이, config만 넘기면 됨.
#
# from megatron.core.transformer.mlp import MLP, MLPSubmodules
# from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
# from megatron.core.transformer.transformer_config import TransformerConfig
#
# config = TransformerConfig(
#     hidden_size=1024,
#     ffn_hidden_size=4096,        # 보통 4 * hidden_size
#     num_layers=1,                # MLP만 쓸 때도 필요
#     num_attention_heads=16,      # MLP만 쓸 때도 필요 (config 유효성)
#     tensor_model_parallel_size=2,
#     add_bias_linear=False,
#     activation_func=F.gelu,
# )
#
# mlp = MLP(
#     config=config,
#     submodules=MLPSubmodules(
#         linear_fc1=ColumnParallelLinear,  # FC1 구현체 지정
#         linear_fc2=RowParallelLinear,     # FC2 구현체 지정
#     ),
# )
#
# # forward: [seq, batch, hidden] → (output, bias)
# hidden_states = torch.randn(128, 8, 1024, device="cuda")
# output, output_bias = mlp(hidden_states)


# ============================================================
# Part 4: SwiGLU MLP (LLaMA style)
# ============================================================
#
# SwiGLU = gated linear unit. FC1이 2개의 projection을 동시에 수행.
#
# config = TransformerConfig(
#     hidden_size=1024,
#     ffn_hidden_size=4096,
#     gated_linear_unit=True,      # ← SwiGLU 활성화
#     activation_func=F.silu,      # ← SiLU (= Swish)
#     ...
# )
#
# 내부 동작:
#   ColumnParallelLinear의 output_size가 2 * ffn_hidden_size가 됨
#   output을 반으로 나눠서 gate, up projection으로 사용:
#     gate, up = fc1_output.chunk(2, dim=-1)
#     intermediate = silu(gate) * up    # gating mechanism
#     output = fc2(intermediate)


# ============================================================
# Part 5: parallel_state 초기화
# ============================================================
#
# Megatron-Core 사용 전에 반드시 process group 초기화 필요.
#
# import megatron.core.parallel_state as mpu
#
# # Step 1: PyTorch distributed 초기화
# torch.distributed.init_process_group(backend="nccl")
#
# # Step 2: Megatron parallel groups 초기화
# mpu.initialize_model_parallel(
#     tensor_model_parallel_size=8,     # TP 크기
#     pipeline_model_parallel_size=4,   # PP 크기
#     # DP는 자동 계산: world_size / (TP * PP)
# )
#
# # 각종 group/rank 조회
# tp_rank = mpu.get_tensor_model_parallel_rank()
# tp_size = mpu.get_tensor_model_parallel_world_size()
# tp_group = mpu.get_tensor_model_parallel_group()
# pp_rank = mpu.get_pipeline_model_parallel_rank()
# dp_group = mpu.get_data_parallel_group()


# ============================================================
# Part 6: GPTModel 전체 조립
# ============================================================
#
# from megatron.core.models.gpt import GPTModel
# from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
#
# config = TransformerConfig(
#     num_layers=32,
#     hidden_size=4096,
#     num_attention_heads=32,
#     num_query_groups=8,              # GQA (8 KV heads)
#     ffn_hidden_size=11008,
#     tensor_model_parallel_size=8,
#     pipeline_model_parallel_size=4,
#     sequence_parallel=True,          # LayerNorm/Dropout도 TP 적용
#     add_bias_linear=False,
#     normalization='RMSNorm',
#     activation_func=F.silu,
#     gated_linear_unit=True,
# )
#
# # layer_spec: 어떤 구현체를 사용할지 지정
# # local_spec: 순수 megatron-core 구현 (기본)
# # te_spec: Transformer Engine 사용 (FP8 지원)
# layer_spec = get_gpt_layer_local_spec()
#
# model = GPTModel(
#     config=config,
#     transformer_layer_spec=layer_spec,
#     vocab_size=32000,
#     max_sequence_length=4096,
#     position_embedding_type='rope',
#     share_embeddings_and_output_weights=True,
# )
#
# # Pipeline parallelism: pre_process / post_process로 stage 구분
# # Stage 0: pre_process=True (embedding 포함)
# # Stage N-1: post_process=True (output head 포함)
# # 중간 stage: 둘 다 False (transformer blocks만)


# ============================================================
# Part 7: 전환 비교 (Vanilla → Megatron-Core)
# ============================================================

def comparison():
    """기존 PyTorch 코드와 Megatron-Core 코드를 나란히 비교."""
    print("=" * 70)
    print("Vanilla PyTorch vs Megatron-Core 비교")
    print("=" * 70)

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    Vanilla PyTorch                                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │  class MLP(nn.Module):                                             │
  │      def __init__(self, hidden, ffn_hidden):                       │
  │          self.fc1 = nn.Linear(hidden, ffn_hidden)                  │
  │          self.fc2 = nn.Linear(ffn_hidden, hidden)                  │
  │          self.act = nn.GELU()                                      │
  │                                                                    │
  │      def forward(self, x):          # x: (batch, seq, hidden)     │
  │          return self.fc2(self.act(self.fc1(x)))                    │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                   Megatron-Core TP MLP                             │
  ├─────────────────────────────────────────────────────────────────────┤
  │  # 방법 1: 직접 조립                                                │
  │  class TP_MLP(MegatronModule):                                     │
  │      def __init__(self, config):                                   │
  │          self.fc1 = ColumnParallelLinear(                          │
  │              hidden, ffn_hidden,                                    │
  │              config=config,                                        │
  │              init_method=init_method_normal(0.02),                 │
  │              gather_output=False,    # split 유지                   │
  │          )                                                         │
  │          self.fc2 = RowParallelLinear(                             │
  │              ffn_hidden, hidden,                                    │
  │              config=config,                                        │
  │              init_method=init_method_normal(0.02),                 │
  │              input_is_parallel=True, # 이미 split된 입력            │
  │          )                                                         │
  │          self.act = nn.GELU()                                      │
  │                                                                    │
  │      def forward(self, x):          # x: (seq, batch, hidden)     │
  │          x, _ = self.fc1(x)         # → (seq, batch, ffn//tp)     │
  │          x = self.act(x)                                           │
  │          x, _ = self.fc2(x)         # → (seq, batch, hidden)      │
  │          return x                   # all-reduce는 fc2 내부에서!   │
  │                                                                    │
  │  # 방법 2: MLP 클래스 사용 (권장)                                    │
  │  mlp = MLP(                                                        │
  │      config=config,                                                │
  │      submodules=MLPSubmodules(                                     │
  │          linear_fc1=ColumnParallelLinear,                          │
  │          linear_fc2=RowParallelLinear,                             │
  │      ),                                                            │
  │  )                                                                 │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    print("  핵심 차이점:")
    diffs = [
        ("텐서 shape",      "(batch, seq, hidden)",   "(seq, batch, hidden)"),
        ("Linear 반환값",    "output",                 "(output, bias) 튜플"),
        ("TP 통신",          "수동 구현 필요",          "ColumnParallel/RowParallel 내장"),
        ("weight 분할",      "수동 split",             "자동 (config의 tp_size 기반)"),
        ("초기화",           "nn.init.*",              "config + init_method callable"),
        ("Config",          "생성자 인자",             "TransformerConfig dataclass"),
    ]

    print(f"  {'항목':<18} {'Vanilla':<25} {'Megatron-Core':<30}")
    print(f"  {'-'*18} {'-'*25} {'-'*30}")
    for item, vanilla, mcore in diffs:
        print(f"  {item:<18} {vanilla:<25} {mcore:<30}")


# ============================================================
# Part 8: Megatron-Core 코드베이스 가이드
# ============================================================

def codebase_guide():
    print("\n" + "=" * 70)
    print("Megatron-Core Codebase Guide")
    print("=" * 70)

    print("""
  megatron/core/
  ├── parallel_state.py              # TP/PP/DP process group 관리
  │                                    → initialize_model_parallel()
  │                                    → get_tensor_model_parallel_group()
  │
  ├── tensor_parallel/
  │   ├── layers.py                  # ColumnParallelLinear, RowParallelLinear
  │   │                                → 가장 핵심. TP Linear 구현
  │   ├── mappings.py                # 통신 primitives
  │   │                                → _CopyToModelParallelRegion
  │   │                                → _ReduceFromModelParallelRegion
  │   └── cross_entropy.py           # vocab parallel cross entropy
  │
  ├── transformer/
  │   ├── transformer_config.py      # TransformerConfig (모든 설정)
  │   ├── mlp.py                     # MLP (ColumnParallel + RowParallel)
  │   ├── attention.py               # Self-attention (TP 적용)
  │   ├── transformer_block.py       # N개 TransformerLayer 쌓기
  │   └── transformer_layer.py       # 단일 layer (attn + mlp + norm)
  │
  ├── models/
  │   └── gpt/
  │       ├── gpt_model.py           # GPTModel (전체 모델)
  │       └── gpt_layer_specs.py     # layer 구현체 지정 (local / TE)
  │
  └── model_parallel_config.py       # ModelParallelConfig (base config)

  읽는 순서 (추천):
    1. parallel_state.py         → process group 이해
    2. tensor_parallel/layers.py → TP Linear 핵심
    3. transformer/mlp.py        → MLP 조립 패턴
    4. gpt_layer_specs.py        → ModuleSpec 패턴 이해
    5. gpt_model.py              → 전체 모델 조립
    """)


# ============================================================
# Part 9: 실행 방법
# ============================================================

def launch_guide():
    print("\n" + "=" * 70)
    print("Megatron-Core 실행 방법")
    print("=" * 70)

    print("""
  # 설치
  pip install megatron-core

  # 또는 소스에서 (최신 기능)
  git clone https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM && pip install -e .

  # 실행 (TP=2로 MLP 학습 예시)
  torchrun --nproc_per_node=2 my_training_script.py

  # Megatron-LM 학습 스크립트 예시
  torchrun --nproc_per_node=8 --nnodes=4 \\
    pretrain_gpt.py \\
    --tensor-model-parallel-size 8 \\
    --pipeline-model-parallel-size 4 \\
    --num-layers 32 \\
    --hidden-size 4096 \\
    --num-attention-heads 32 \\
    --seq-length 4096 \\
    --micro-batch-size 4 \\
    --global-batch-size 512 \\
    --lr 3e-4 \\
    --train-iters 100000 \\
    --bf16
    """)


# ============================================================
# Part 10: 시뮬레이션 (GPU 없이 구조 확인)
# ============================================================

def simulate_mcore_mlp():
    """Megatron-Core MLP의 weight 분할을 시뮬레이션."""
    print("=" * 70)
    print("Megatron-Core MLP Simulation (no GPUs needed)")
    print("=" * 70)

    torch.manual_seed(42)
    hidden_size = 8
    ffn_hidden_size = 16
    tp_size = 2

    # 전체 weight (single GPU 기준)
    W1 = torch.randn(hidden_size, ffn_hidden_size)   # fc1: (hidden, ffn)
    W2 = torch.randn(ffn_hidden_size, hidden_size)   # fc2: (ffn, hidden)
    x = torch.randn(4, 2, hidden_size)               # (seq, batch, hidden)

    # --- Single GPU (baseline) ---
    out_single = F.gelu(x @ W1) @ W2

    # --- Megatron-Core TP 시뮬레이션 ---
    # ColumnParallelLinear: W1을 column 방향으로 split
    #   GPU 0: W1[:, :ffn//2]  → output의 앞 절반
    #   GPU 1: W1[:, ffn//2:]  → output의 뒷 절반
    half = ffn_hidden_size // tp_size
    W1_rank0 = W1[:, :half]          # (hidden, ffn//2)
    W1_rank1 = W1[:, half:]

    # RowParallelLinear: W2를 row 방향으로 split
    #   GPU 0: W2[:ffn//2, :]
    #   GPU 1: W2[ffn//2:, :]
    W2_rank0 = W2[:half, :]          # (ffn//2, hidden)
    W2_rank1 = W2[half:, :]

    # 각 rank에서 독립 계산
    # Rank 0: x → fc1_rank0 → GELU → fc2_rank0 → partial output
    a1_rank0 = F.gelu(x @ W1_rank0)              # (seq, batch, ffn//2)
    partial_rank0 = a1_rank0 @ W2_rank0           # (seq, batch, hidden)

    # Rank 1: 동일
    a1_rank1 = F.gelu(x @ W1_rank1)
    partial_rank1 = a1_rank1 @ W2_rank1

    # RowParallelLinear 내부의 all-reduce
    out_tp = partial_rank0 + partial_rank1

    diff = (out_single - out_tp).abs().max().item()

    print(f"\n  Config: hidden={hidden_size}, ffn={ffn_hidden_size}, tp={tp_size}")
    print(f"\n  Weight shapes:")
    print(f"    Single GPU:  W1={list(W1.shape)}, W2={list(W2.shape)}")
    print(f"    Per TP rank: W1={list(W1_rank0.shape)}, W2={list(W2_rank0.shape)}")
    print(f"\n  Activation flow (per rank):")
    print(f"    Input:        {list(x.shape)}")
    print(f"    After fc1:    {list(a1_rank0.shape)}  (ffn split across ranks)")
    print(f"    After fc2:    {list(partial_rank0.shape)}  (partial sum)")
    print(f"    After reduce: {list(out_tp.shape)}  (final output)")
    print(f"\n  Single GPU vs TP diff: {diff:.2e}")
    print(f"  Result: {'PASSED' if diff < 1e-5 else 'FAILED'}")

    # (output, bias) 튜플 반환 패턴 설명
    print(f"\n  참고: Megatron-Core의 forward는 (output, bias) 튜플 반환.")
    print(f"  skip_bias_add=True → bias를 output에 더하지 않고 별도 반환.")
    print(f"  이유: LayerNorm fusion 등에서 bias add를 묶어서 처리하면 더 빠름.")


# ============================================================
# Part 11: 성능 벤치마크 (Vanilla vs TP-split MLP)
# ============================================================
#
# 실제 Megatron-Core는 multi-GPU 환경이 필요하므로,
# 여기서는 single GPU에서 "TP split된 weight로 연산하면 얼마나 빨라지는지"
# 를 측정. 실제 TP의 이점은 multi-GPU에서 통신 숨기기까지 포함.

def benchmark_vanilla_vs_tp_split():
    """Vanilla MLP vs TP-split MLP의 single-device 성능 비교."""
    print("\n" + "=" * 70)
    print("Performance Benchmark: Vanilla vs TP-split MLP")
    print("=" * 70)

    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = device == "cuda"

    configs = [
        # (hidden, ffn, seq, batch, label)
        (1024,  4096,   512, 8,  "Small  (1K→4K)"),
        (4096,  16384,  512, 8,  "Medium (4K→16K)"),
        (8192,  32768,  512, 4,  "Large  (8K→32K)"),
    ]

    tp_sizes = [1, 2, 4, 8]

    print(f"\n  Device: {device}")
    print(f"  측정: forward + backward 시간 (ms)")
    if not is_cuda:
        print(f"  (CPU에서는 TP split의 이점이 제한적. GPU에서 실행 권장.)")

    for hidden, ffn, seq, batch, label in configs:
        print(f"\n  --- {label}: hidden={hidden}, ffn={ffn}, seq={seq}, batch={batch} ---")
        print(f"  {'TP size':<10} {'Matmul size (per rank)':<30} {'Fwd+Bwd (ms)':>14} {'Speedup':>10}")
        print(f"  {'-'*10} {'-'*30} {'-'*14} {'-'*10}")

        baseline_time = None

        for tp in tp_sizes:
            if ffn % tp != 0:
                continue

            ffn_per_rank = ffn // tp

            # Megatron-Core style: W1 column split, W2 row split
            W1 = torch.randn(hidden, ffn_per_rank, device=device, requires_grad=True)
            W2 = torch.randn(ffn_per_rank, hidden, device=device, requires_grad=True)
            x = torch.randn(seq, batch, hidden, device=device, requires_grad=True)

            # Warmup
            for _ in range(3):
                out = F.gelu(x @ W1) @ W2
                out.sum().backward()

            if is_cuda:
                torch.cuda.synchronize()

            # Benchmark
            n_iters = 20
            start = time.perf_counter()
            for _ in range(n_iters):
                out = F.gelu(x @ W1) @ W2
                out.sum().backward()
            if is_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) / n_iters * 1000

            if baseline_time is None:
                baseline_time = elapsed_ms
            speedup = baseline_time / elapsed_ms

            matmul_desc = f"({seq*batch}×{hidden}) @ ({hidden}×{ffn_per_rank})"
            print(f"  {tp:<10} {matmul_desc:<30} {elapsed_ms:>11.2f} ms {speedup:>9.2f}x")

    print(f"""
  해석:
    - TP=1 (no split): 전체 ffn_hidden_size로 matmul → baseline
    - TP=N: ffn_hidden_size/N으로 matmul → 각 rank의 연산량 1/N

    Single GPU에서는 순차 실행이라 TP split의 이점이 제한적이지만,
    실제 multi-GPU에서는:
      1. 각 rank가 동시에 1/N 크기 matmul 실행 → ~1/N 시간
      2. All-reduce 통신을 backward과 overlap → 추가 비용 최소화
      3. 결과: 거의 선형 speedup (NVLink 대역폭 충분 시)

  실제 multi-GPU 기대 성능 (NVLink 기준):
    TP=2: ~1.8-1.9x speedup (통신 overhead 5-10%)
    TP=4: ~3.5-3.7x speedup
    TP=8: ~6.5-7.0x speedup (노드 내 NVLink)
    """)


if __name__ == "__main__":
    simulate_mcore_mlp()
    comparison()
    benchmark_vanilla_vs_tp_split()
    codebase_guide()
    launch_guide()
