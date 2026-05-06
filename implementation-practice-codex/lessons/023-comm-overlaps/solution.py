"""
Communication Overlaps
========================
통신과 연산을 겹쳐서 학습 throughput 극대화.

문제: GPU가 통신 기다리는 동안 idle → GPU utilization 저하
해결: 비동기 통신으로 연산과 통신을 동시에 수행

    Without overlap:
    GPU:  [compute][  wait  ][compute][  wait  ][compute]
    NIC:  [  idle  ][ comm  ][  idle  ][ comm  ][  idle ]

    With overlap:
    GPU:  [compute][compute][compute]
    NIC:  [ comm  ][ comm  ][ comm  ]
    → GPU와 NIC가 동시에 일함!
"""

import torch
import torch.nn as nn
import time


# ============================================================
# Part 1: DDP Gradient Bucketing + Overlap
# ============================================================
#
# DDP의 핵심 최적화: backward 계산 중에 gradient 통신을 겹침.
#
# 동작 원리:
#   1. Backward는 마지막 layer부터 시작 (layer N → layer 0)
#   2. Layer N의 gradient 계산 완료 → 즉시 all-reduce 시작 (비동기)
#   3. Layer N-1의 gradient 계산 시작 (동시에 layer N 통신 진행 중)
#   4. ...
#
#   Layer N:   [backward][ all-reduce  ]
#   Layer N-1: [  wait   ][backward][ all-reduce  ]
#   Layer N-2: [         ][  wait   ][backward][ all-reduce  ]
#
# Bucket 단위로 all-reduce:
#   - gradient를 개별 all-reduce하면 overhead 큼 (launch cost)
#   - 여러 gradient를 bucket (기본 25MB)으로 모아서 한번에
#   - bucket이 차면 all-reduce 시작 → 나머지 backward와 overlap
#
# PyTorch DDP 설정:
#   model = DDP(model,
#       bucket_cap_mb=25,           # bucket 크기 (작을수록 overlap 기회 많음)
#       gradient_as_bucket_view=True,  # 메모리 절약 (grad가 bucket의 view)
#   )


# ============================================================
# Part 2: FSDP Prefetching
# ============================================================
#
# FSDP에서 all-gather와 연산 겹치기.
#
# 문제: FSDP는 각 layer forward 전에 all-gather 필요
#       all-gather 기다리면 GPU가 idle
#
# 해결: 다음 layer의 all-gather를 미리 시작 (prefetch)
#
#   Layer 0: [all-gather][forward][  free  ]
#   Layer 1:      [all-gather    ][forward][  free  ]
#   Layer 2:            [all-gather       ][forward]
#
# PyTorch FSDP 설정:
#   model = FSDP(model,
#       forward_prefetch=True,   # forward에서 다음 FSDP unit prefetch
#       backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # backward에서도 prefetch
#       limit_all_gathers=True,  # 동시 all-gather 수 제한 (메모리 관리)
#   )
#
# FSDP2:
#   fully_shard(model, mesh=mesh,
#       reshard_after_forward=True,   # forward 후 shard로 돌아감
#   )


# ============================================================
# Part 3: TP의 Compute-Communication Overlap
# ============================================================
#
# TP에서 all-reduce와 다음 layer 연산 겹치기.
#
# 기본 TP:
#   Layer 0: [column matmul][GELU][row matmul][all-reduce]
#   Layer 1:                                   [column matmul][GELU]...
#
# Overlap TP:
#   Layer 0: [column matmul][GELU][row matmul][all-reduce      ]
#   Layer 1:                                   [column matmul   ][GELU]...
#                                              ↑ all-reduce와 동시!
#
# 구현: CUDA stream 분리
#   compute_stream = torch.cuda.Stream()
#   comm_stream = torch.cuda.Stream()
#
#   with torch.cuda.stream(comm_stream):
#       dist.all_reduce(output)  # 통신 stream에서 all-reduce
#
#   with torch.cuda.stream(compute_stream):
#       next_layer_out = next_layer(input)  # 계산 stream에서 다음 layer
#
#   comm_stream.synchronize()  # 통신 완료 대기


# ============================================================
# Part 4: Pipeline Parallelism Overlap
# ============================================================
#
# 1F1B schedule 자체가 overlap의 한 형태:
#   - Forward와 backward를 번갈아 수행
#   - send/recv를 계산과 겹침
#
# 구현:
#   # 비동기 send/recv로 overlap
#   send_work = dist.isend(activation, dst=next_stage)  # 비동기 전송
#   output = current_stage_backward(...)                  # 동시에 backward
#   send_work.wait()                                      # 전송 완료 확인


# ============================================================
# Part 5: Async 통신 시뮬레이션
# ============================================================

def simulate_overlap():
    """비동기 통신으로 overlap하는 효과를 시뮬레이션."""
    print("=" * 60)
    print("Communication Overlap Simulation")
    print("=" * 60)

    # 시뮬레이션 파라미터
    compute_time = 10   # ms
    comm_time = 8       # ms
    num_layers = 4

    # Without overlap: sequential
    total_no_overlap = num_layers * (compute_time + comm_time)

    # With overlap: 첫 layer만 sequential, 이후 overlap
    total_overlap = compute_time + comm_time  # 첫 layer
    for _ in range(num_layers - 1):
        total_overlap += max(compute_time, comm_time)  # overlap

    print(f"\n  Per-layer: compute={compute_time}ms, comm={comm_time}ms")
    print(f"  Layers: {num_layers}")
    print(f"\n  Without overlap:")
    print(f"    Total: {total_no_overlap}ms")
    for i in range(num_layers):
        c = "C" * compute_time
        m = "M" * comm_time
        print(f"    Layer {i}: [{c}][{m}]")

    print(f"\n  With overlap:")
    print(f"    Total: {total_overlap}ms ({(1-total_overlap/total_no_overlap)*100:.0f}% faster)")
    for i in range(num_layers):
        c = "C" * compute_time
        m = "M" * comm_time
        if i == 0:
            print(f"    Layer {i}: [{c}][{m}]")
        else:
            overlap_part = min(compute_time, comm_time)
            print(f"    Layer {i}: [{c}]")
            print(f"    Comm {i}:  {'':>{compute_time-overlap_part}}[{m}]  ← overlapped!")


# ============================================================
# Part 6: CUDA Stream 사용법
# ============================================================

def cuda_streams_demo():
    """CUDA stream으로 연산 overlap (GPU 필요)."""
    print("\n" + "=" * 60)
    print("CUDA Streams (concept)")
    print("=" * 60)

    print("""
  CUDA Stream: GPU 연산의 순서를 보장하는 큐.
  같은 stream 내 → 순차 실행
  다른 stream 간 → 병렬 실행 가능

  사용 패턴:
    default_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()

    # 기본 stream에서 연산
    y = model_layer(x)

    # 통신 stream에서 all-reduce (y의 연산 완료 대기 후 시작)
    comm_stream.wait_stream(default_stream)  # 의존성 명시
    with torch.cuda.stream(comm_stream):
        dist.all_reduce(y)

    # 기본 stream에서 다음 연산 (all-reduce와 병렬!)
    z = next_layer(x2)

    # all-reduce 결과 필요한 시점에서 동기화
    default_stream.wait_stream(comm_stream)
    final = z + y  # 이 시점에서 y의 all-reduce 완료 보장

  주의:
    - stream 간 의존성을 명시하지 않으면 race condition
    - event로 더 세밀한 동기화 가능:
        event = comm_stream.record_event()
        default_stream.wait_event(event)
    """)


# ============================================================
# Part 7: Performance Tips
# ============================================================

def performance_tips():
    print("\n" + "=" * 60)
    print("Performance Optimization Checklist")
    print("=" * 60)

    tips = [
        ("DDP bucket size",
         "bucket_cap_mb 조정. 작으면 overlap↑ but launch overhead↑. 기본 25MB가 보통 적절."),
        ("NCCL 환경변수",
         "NCCL_IB_DISABLE=0 (InfiniBand 사용), NCCL_SOCKET_IFNAME=eth0 (네트워크 인터페이스)"),
        ("torch.compile",
         "operator fusion으로 kernel launch overhead 감소 + 메모리 최적화"),
        ("Pin memory",
         "DataLoader(pin_memory=True)로 CPU→GPU 전송 속도 향상"),
        ("Mixed precision",
         "BF16으로 연산 2x + 통신량 2x 감소"),
        ("Gradient accumulation",
         "effective batch 늘려서 통신 빈도 감소"),
        ("Profiling",
         "torch.profiler로 bottleneck 파악: GPU util, 통신 대기 시간"),
    ]

    for name, desc in tips:
        print(f"\n  {name}:")
        print(f"    {desc}")


if __name__ == "__main__":
    simulate_overlap()
    cuda_streams_demo()
    performance_tips()
