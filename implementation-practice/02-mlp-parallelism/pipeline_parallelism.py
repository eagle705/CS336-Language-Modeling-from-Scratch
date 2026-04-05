"""
Pipeline Parallelism (PP) for MLP
===================================
모델의 layer들을 여러 GPU에 순차적으로 배치.

핵심 아이디어:
- 각 GPU가 모델의 일부 layer만 담당
- Micro-batching으로 pipeline bubble 최소화

    GPU 0: [Layer 0, Layer 1]  ─── activation ──→  GPU 1: [Layer 2, Layer 3]
                                  (send/recv)

Naive PP (큰 bubble):
    Time →
    GPU 0: [  Forward  ][   idle   ][  Backward  ]
    GPU 1: [   idle    ][  Forward  ][  Backward  ]

GPipe (micro-batch로 bubble 축소):
    Time →
    GPU 0: [F_m0][F_m1][F_m2][F_m3][    ][B_m3][B_m2][B_m1][B_m0]
    GPU 1: [    ][F_m0][F_m1][F_m2][F_m3][B_m3][B_m2][B_m1][B_m0]

1F1B (메모리 효율적):
    Time →
    GPU 0: [F_m0][F_m1][B_m0][F_m2][B_m1][F_m3][B_m2][B_m3]
    GPU 1: [    ][F_m0][F_m1][B_m0][F_m2][B_m1][F_m3][B_m2][B_m3]

인터뷰 포인트:
1. Bubble ratio = (pp_size - 1) / num_microbatches
2. 통신: stage 경계에서 activation/gradient의 point-to-point 전송 (send/recv)
3. 1F1B가 GPipe보다 메모리 효율적 (activation 보관 수: pp_size vs num_microbatches)
"""

import torch
import torch.nn as nn
from typing import List


# ============================================================
# Part 1: torch.distributed P2P 통신 API 정리
# ============================================================
#
# PP에서는 all-reduce가 아닌 point-to-point(P2P) 통신 사용:
#
# dist.send(tensor, dst=1)     # 현재 GPU → GPU 1로 tensor 전송 (blocking)
# dist.recv(tensor, src=0)     # GPU 0으로부터 tensor 수신 (blocking)
#
# dist.isend(tensor, dst=1)    # 비동기 전송 → Work 객체 반환
# dist.irecv(tensor, src=0)    # 비동기 수신 → Work 객체 반환
# work.wait()                  # 비동기 작업 완료 대기
#
# PP 통신 패턴:
#   Forward:  GPU 0 → send(activation) → GPU 1 → send(activation) → GPU 2
#   Backward: GPU 0 ← recv(gradient)  ← GPU 1 ← recv(gradient)  ← GPU 2
#
# 예시:
#   # GPU 0 (stage 0)
#   output = stage_0(input)
#   dist.send(output, dst=1)          # activation을 다음 stage로
#   dist.recv(grad, src=1)            # gradient를 다음 stage로부터
#   output.backward(grad)
#
#   # GPU 1 (stage 1)
#   dist.recv(input, src=0)           # activation을 이전 stage로부터
#   output = stage_1(input)
#   loss = loss_fn(output, target)
#   loss.backward()
#   dist.send(input.grad, dst=0)      # gradient를 이전 stage로


# ============================================================
# Part 2: 수동 PP 구현 (GPipe style)
# ============================================================

class PipelineStage(nn.Module):
    """하나의 GPU에 배치되는 layer 묶음."""

    def __init__(self, layers: List[nn.Module], embed_dim: int,
                 is_first=False, is_last=False, vocab_size=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.is_first = is_first
        self.is_last = is_last

        if is_first:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        if is_last:
            self.ln = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        if self.is_first:
            x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)  # residual
        if self.is_last:
            x = self.ln(x)
            x = self.head(x)
        return x


class GPipeSimulator:
    """
    GPipe 시뮬레이터 (single process에서 동작).

    동작 순서:
    1. 모든 micro-batch forward (stage 0 → N-1)
    2. 모든 micro-batch backward (stage N-1 → 0)
    3. gradient accumulation 후 update
    """

    def __init__(self, stages: List[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.num_stages = len(stages)

    def forward_backward(self, input_ids, targets, loss_fn):
        M = self.num_microbatches
        S = self.num_stages

        input_chunks = input_ids.chunk(M)
        target_chunks = targets.chunk(M)

        # activations[stage][mb] = tensor. backward에서 gradient 계산에 필요.
        activations = [[None] * M for _ in range(S + 1)]
        losses = []

        # --- Forward: 모든 micro-batch를 stage 순서대로 ---
        for mb in range(M):
            activations[0][mb] = input_chunks[mb]
            for s in range(S):
                inp = activations[s][mb]
                if s > 0:
                    # stage 경계: detach로 그래프 끊고 requires_grad 설정
                    # (실제 PP에서는 send/recv가 이 역할)
                    inp = inp.detach().requires_grad_(True)
                out = self.stages[s](inp)
                activations[s][mb] = inp      # backward용 input 보관
                activations[s + 1][mb] = out  # 다음 stage의 input

            logits = activations[S][mb]
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_chunks[mb].view(-1)
            ) / M  # gradient accumulation: M으로 나눠서 평균
            losses.append(loss.item())

        # --- Backward: 모든 micro-batch를 역순으로 ---
        for mb in reversed(range(M)):
            logits = activations[S][mb]
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_chunks[mb].view(-1)
            ) / M
            loss.backward()

        return sum(losses)

    def print_schedule(self):
        M = self.num_microbatches
        S = self.num_stages
        total_steps = 2 * (S + M - 1)
        forward_steps = S + M - 1

        print(f"\nGPipe Schedule (stages={S}, microbatches={M}):")
        print("-" * 60)
        for s in range(S):
            timeline = []
            for t in range(total_steps):
                if t < forward_steps:
                    mb = t - s
                    timeline.append(f"F{mb}" if 0 <= mb < M else "  ")
                else:
                    bt = t - forward_steps
                    mb = M - 1 - (bt - s)
                    timeline.append(f"B{mb}" if 0 <= mb < M and bt >= s else "  ")
            print(f"  GPU {s}: [{'|'.join(timeline)}]")

        print(f"\n  Bubble ratio = (pp-1)/M = {(S-1)/M:.2f}")


# ============================================================
# Part 3: 1F1B Schedule 시각화
# ============================================================

class OneFOneBSimulator:
    """
    1F1B Schedule.

    GPipe와 차이:
    - Warmup: pp_size개 forward만 먼저
    - Steady state: 1 backward + 1 forward 번갈아
    - Cooldown: 남은 backward
    장점: peak activation memory = pp_size (GPipe는 num_microbatches)
    """

    def print_schedule(self, num_stages, num_microbatches):
        S = num_stages
        M = num_microbatches
        assert M >= S

        print(f"\n1F1B Schedule (stages={S}, microbatches={M}):")
        print("-" * 60)
        for s in range(S):
            timeline = []
            f_done = 0
            b_done = 0

            # Delay
            for _ in range(s):
                timeline.append("  ")

            # Warmup forwards
            for _ in range(S - s):
                if f_done < M:
                    timeline.append(f"F{f_done}")
                    f_done += 1

            # Steady state: 1B + 1F
            while f_done < M or b_done < M:
                if b_done < M:
                    timeline.append(f"B{b_done}")
                    b_done += 1
                if f_done < M:
                    timeline.append(f"F{f_done}")
                    f_done += 1

            # Cooldown
            while b_done < M:
                timeline.append(f"B{b_done}")
                b_done += 1

            max_len = 2 * M + S - 1
            while len(timeline) < max_len:
                timeline.append("  ")
            print(f"  GPU {s}: [{'|'.join(timeline[:max_len])}]")

        print(f"\n  Peak activation memory: {S} micro-batches (GPipe: {M})")
        print(f"  Bubble ratio = (pp-1)/M = {(S-1)/M:.2f}")


# ============================================================
# Part 4: torch.distributed.pipelining (PyTorch 2.x native)
# ============================================================
#
# --- 핵심 API ---
#
# from torch.distributed.pipelining import (
#     pipeline,          # tracer로 모델 자동 분할
#     SplitPoint,        # 어디서 자를지 지정
#     PipelineStage,     # 하나의 stage
#     ScheduleGPipe,     # GPipe 스케줄러
#     Schedule1F1B,      # 1F1B 스케줄러
# )
#
# --- 사용법 (3단계) ---
#
# 1) 모델을 stage로 분할:
#
#   pipe = pipeline(
#       module=model,
#       mb_args=(example_input,),       # micro-batch 예시 (shape 추론용)
#       split_spec={
#           "layers.2": SplitPoint.BEGINNING,   # layers.2 앞에서 자름
#           "layers.4": SplitPoint.BEGINNING,   # layers.4 앞에서 자름
#       },
#   )
#   # → 3개 stage: [layers.0-1], [layers.2-3], [layers.4-...]
#
# 2) 이 rank의 stage 빌드:
#
#   stage = pipe.build_stage(
#       stage_index=rank,
#       device=torch.device(f"cuda:{rank}"),
#   )
#
# 3) 스케줄 실행:
#
#   schedule = ScheduleGPipe(stage, n_microbatches=8, loss_fn=loss_fn)
#   # 또는
#   schedule = Schedule1F1B(stage, n_microbatches=8, loss_fn=loss_fn)
#
#   if rank == 0:
#       schedule.step(input_batch)   # 첫 stage만 input 넣음
#   else:
#       schedule.step()              # 나머지는 이전 stage에서 recv

def pipelining_example():
    """
    torch.distributed.pipelining으로 PP 적용하는 전체 코드.
    실행: torchrun --nproc_per_node=2 pipeline_parallelism.py pipelining
    """
    import torch.distributed as dist
    from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 1) 일반 모델 정의
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(256, 256)
            self.layer1 = nn.Linear(256, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.layer0(x))
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)

    model = SimpleModel().cuda()

    # 2) pipeline()으로 자동 분할
    example_input = torch.randn(4, 256, device="cuda")  # micro-batch 예시
    pipe = pipeline(
        module=model,
        mb_args=(example_input,),
        split_spec={
            "layer2": SplitPoint.BEGINNING,  # layer2 앞에서 자름 → 2 stages
        },
    )

    # 3) 이 rank의 stage 빌드
    stage = pipe.build_stage(stage_index=rank, device=torch.device(f"cuda:{rank}"))

    # 4) 스케줄 생성 및 실행
    loss_fn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, n_microbatches=4, loss_fn=loss_fn)

    # 실행
    input_batch = torch.randn(16, 256, device="cuda")  # 4 micro-batches x 4
    target = torch.randint(0, 10, (16,), device="cuda")

    if rank == 0:
        schedule.step(input_batch)
    else:
        losses = []
        output = schedule.step(target=target, losses=losses)
        print(f"Loss: {losses}")

    dist.destroy_process_group()


# ============================================================
# Part 5: Demo
# ============================================================

def demo():
    """schedule 시각화 + 메모리 비교."""
    print("=" * 60)
    print("Pipeline Parallelism Demo")
    print("=" * 60)

    torch.manual_seed(42)

    # 모델 세팅
    from mlp_baseline import MLPBlock
    embed_dim, hidden_dim, vocab_size = 64, 256, 100
    num_layers, pp_size, num_microbatches = 4, 2, 4

    layers = [MLPBlock(embed_dim, hidden_dim, dropout=0.0) for _ in range(num_layers)]
    layers_per_stage = num_layers // pp_size

    stages = [
        PipelineStage(layers[:layers_per_stage], embed_dim,
                       is_first=True, vocab_size=vocab_size),
        PipelineStage(layers[layers_per_stage:], embed_dim,
                       is_last=True, vocab_size=vocab_size),
    ]

    # GPipe 시뮬레이션
    input_ids = torch.randint(0, vocab_size, (8, 16))
    targets = torch.randint(0, vocab_size, (8, 16))

    sim = GPipeSimulator(stages, num_microbatches)
    loss = sim.forward_backward(input_ids, targets, nn.functional.cross_entropy)
    print(f"\n  GPipe loss: {loss:.4f}")
    sim.print_schedule()

    # 1F1B schedule
    ofob = OneFOneBSimulator()
    ofob.print_schedule(num_stages=2, num_microbatches=4)
    ofob.print_schedule(num_stages=4, num_microbatches=8)

    # 메모리 비교
    print("\n" + "=" * 60)
    print("Memory Comparison")
    print("=" * 60)
    print(f"  {'M (micro-batches)':<22} {'GPipe peak':<15} {'1F1B peak':<15} {'Bubble'}")
    print(f"  {'-'*22} {'-'*15} {'-'*15} {'-'*10}")
    for M in [4, 8, 16]:
        print(f"  {M:<22} {M:<15} {pp_size:<15} {(pp_size-1)/M:.1%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pipelining":
        pipelining_example()
    else:
        demo()
