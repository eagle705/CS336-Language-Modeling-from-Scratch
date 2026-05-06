"""Pipeline Parallelism (PP) for MLP
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

실제 P2P send/recv 예제 (GPU 8장 가정):
    torchrun --nproc_per_node=8 pipeline_parallelism.py p2p

Jupyter 단일 커널에서는 보통 rank가 1개만 잡히므로 실제 P2P pipeline을 보기 어렵다.
노트북에서는 demo() / GPipeSimulator / OneFOneBSimulator로 스케줄을 보고,
진짜 dist.send/recv는 torchrun으로 여러 프로세스를 띄워 실행하는 것이 정석.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn
from typing import List

class PipelineStage(nn.Module):
    """하나의 GPU에 배치되는 layer 묶음."""

    def __init__(self, layers: List[nn.Module], embed_dim: int, is_first=False, is_last=False, vocab_size=None):
        raise NotImplementedError('TODO: implement PipelineStage.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement PipelineStage.forward; compare with solution.py only after trying.')

class GPipeSimulator:
    """GPipe 시뮬레이터 (single process에서 동작).

동작 순서:
1. 모든 micro-batch forward (stage 0 → N-1)
2. 모든 micro-batch backward (stage N-1 → 0)
3. gradient accumulation 후 update"""

    def __init__(self, stages: List[PipelineStage], num_microbatches: int):
        raise NotImplementedError('TODO: implement GPipeSimulator.__init__; compare with solution.py only after trying.')

    def forward_backward(self, input_ids, targets, loss_fn):
        raise NotImplementedError('TODO: implement GPipeSimulator.forward_backward; compare with solution.py only after trying.')

    def print_schedule(self):
        raise NotImplementedError('TODO: implement GPipeSimulator.print_schedule; compare with solution.py only after trying.')

class OneFOneBSimulator:
    """1F1B Schedule.

GPipe와 차이:
- Warmup: pp_size개 forward만 먼저
- Steady state: 1 backward + 1 forward 번갈아
- Cooldown: 남은 backward
장점: peak activation memory = pp_size (GPipe는 num_microbatches)"""

    def print_schedule(self, num_stages, num_microbatches):
        raise NotImplementedError('TODO: implement OneFOneBSimulator.print_schedule; compare with solution.py only after trying.')

def manual_p2p_pipeline_example():
    """8-GPU 기준 수동 pipeline parallel 예제.

각 rank는 작은 Linear+ReLU stage 하나만 가진다.
forward에서는 activation을 다음 rank로 보내고,
backward에서는 input gradient를 이전 rank로 보낸다.

이 예제는 교육용으로 blocking send/recv를 사용한다.
실제 고성능 PP는 non-blocking P2P, micro-batch 스케줄링(1F1B), overlap 등을 더한다."""
    raise NotImplementedError('TODO: implement manual_p2p_pipeline_example; compare with solution.py only after trying.')

def pipelining_example():
    """torch.distributed.pipelining으로 PP 적용하는 전체 코드.
실행: torchrun --nproc_per_node=2 pipeline_parallelism.py pipelining"""
    raise NotImplementedError('TODO: implement pipelining_example; compare with solution.py only after trying.')

def demo():
    """schedule 시각화 + 메모리 비교."""
    raise NotImplementedError('TODO: implement demo; compare with solution.py only after trying.')
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'p2p':
        manual_p2p_pipeline_example()
    elif len(sys.argv) > 1 and sys.argv[1] == 'pipelining':
        pipelining_example()
    else:
        demo()
