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
import torch.distributed as dist
import torch.nn as nn
import os
import socket
import time


# ============================================================
# Part 1: DDP Gradient Bucketing + Overlap
# ============================================================
#
# DDP의 핵심 최적화: backward 계산 중에 gradient 통신을 겹침.
# Megatron 옵션으로는 --overlap-grad-reduce에 해당:
#   backward 중 어떤 layer/bucket의 gradient가 준비되면
#   다음 backward compute를 계속하면서 grad reduce-scatter/all-reduce를 비동기로 진행.
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
# 개별 all-reduce vs bucket:
#   개별 all-reduce:
#     layer1.weight.grad → all-reduce 1번
#     layer1.bias.grad   → all-reduce 1번
#     layer2.weight.grad → all-reduce 1번
#     ...
#     작은 tensor마다 NCCL collective를 시작하므로 launch/latency overhead가 반복된다.
#
#   bucket all-reduce:
#     bucket0 = [layer1.weight.grad, layer1.bias.grad, layer2.weight.grad, ...]  # 예: 25MB
#     bucket0 전체에 대해 all-reduce 1번
#     여러 작은 gradient 택배를 상자에 모아서 한 번에 보내는 느낌.
#
#   trade-off:
#     bucket이 작으면 overlap 기회는 많지만 launch overhead가 늘고,
#     bucket이 크면 launch overhead는 줄지만 bucket이 다 찰 때까지 기다려야 한다.
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
# Megatron 옵션으로는 --overlap-param-gather에 해당:
#   현재 layer를 계산하는 동안 다음 layer/FSDP unit의 parameter all-gather를 미리 시작.
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
# Megatron 옵션으로는 --tp-comm-overlap에 해당:
#   tensor-parallel row/column parallel linear 주변의 all-reduce/reduce-scatter/all-gather를
#   CUDA communication stream에 걸고 다음 compute와 겹치게 한다.
#
# 기본 TP:
#   Layer 0: [column matmul][GELU][row matmul][all-reduce]
#   Layer 1:                                              [column matmul][GELU]...
#            ↑ Layer 0의 all-reduce가 끝날 때까지 Layer 1 시작 못 함
#
# Overlap TP:
#   Layer 0: [column matmul][GELU][row matmul][all-reduce      ]
#   Layer 1:                                   [column matmul   ][GELU]...
#                                              ↑ Layer 0 all-reduce 중에 Layer 1 일부 계산
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
# Part 6: 실제 dist API overlap demo
# ============================================================

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_dist_for_overlap():
    """
    torchrun 멀티프로세스와 단일 프로세스 실행을 모두 지원하는 초기화 helper.

    멀티 GPU:
      torchrun --nproc_per_node=2 implementation-practice-codex/lessons/023-comm-overlaps/solution.py dist

    단일 프로세스:
      python implementation-practice-codex/lessons/023-comm-overlaps/solution.py dist
      world_size=1이라 실제 rank 간 통신은 없지만 API 흐름은 확인 가능.
    """
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _find_free_port())

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        initialized_here = True

    return rank, local_rank, world_size, device, initialized_here


def _compute_work(device, size=512, iters=8):
    """통신과 겹칠 수 있는 dummy compute workload."""
    x = torch.randn(size, size, device=device)
    w = torch.randn(size, size, device=device)
    for _ in range(iters):
        x = torch.relu(x @ w)
    return x


def _time_block(device, fn):
    """CUDA면 synchronize를 포함해 wall-clock 시간을 잰다."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, result


def run_dist_overlap_demo():
    """
    실제 torch.distributed all-reduce와 compute overlap 데모.

    핵심 API:
      work = dist.all_reduce(tensor, async_op=True)
      ... 다른 compute 수행 ...
      work.wait()

    CUDA에서는 comm_stream에서 all_reduce를 enqueue하고 default stream에서 compute를 수행해
    NCCL 통신과 CUDA kernel compute가 서로 다른 stream에서 진행되도록 한다.
    CPU/Gloo에서는 async_op=True가 Work handle을 반환하므로, compute 후 wait하는 흐름을 보여준다.
    """
    rank, local_rank, world_size, device, initialized_here = _init_dist_for_overlap()

    try:
        torch.manual_seed(1234 + rank)
        comm_tensor = torch.ones(8 * 1024 * 1024, device=device)  # 약 32MB FP32

        # Warmup
        _compute_work(device, size=128, iters=2)
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

        def no_overlap():
            # 순차 실행: compute가 끝난 뒤 blocking all_reduce.
            _compute_work(device)
            dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

        def with_overlap():
            if device.type == "cuda":
                # CUDA stream 두 개 사용:
                #   comm_stream: NCCL all_reduce enqueue
                #   default stream: dummy compute 실행
                #
                # async_op=True는 통신 완료를 기다리지 않고 Work handle을 즉시 반환한다.
                # compute를 진행한 뒤 work.wait()/stream synchronize로 통신 완료를 보장한다.
                comm_stream = torch.cuda.Stream(device=device)
                default_stream = torch.cuda.current_stream(device)

                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(default_stream)
                    work = dist.all_reduce(
                        comm_tensor,
                        op=dist.ReduceOp.SUM,
                        async_op=True,
                    )

                _compute_work(device)
                work.wait()
                default_stream.wait_stream(comm_stream)
            else:
                # CPU/Gloo에서도 async_op=True는 Work handle을 반환한다.
                # 실제 overlap 정도는 backend/threading에 따라 다르지만 API 패턴은 동일하다.
                work = dist.all_reduce(
                    comm_tensor,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
                _compute_work(device, size=256, iters=4)
                work.wait()

        no_overlap_ms, _ = _time_block(device, no_overlap)
        overlap_ms, _ = _time_block(device, with_overlap)

        # rank별 측정값 중 가장 느린 시간을 전체 step time으로 보는 것이 보통 더 현실적이다.
        timing = torch.tensor([no_overlap_ms, overlap_ms], device=device)
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)

        if rank == 0:
            print("=" * 60)
            print("Real dist Communication Overlap Demo")
            print("=" * 60)
            print(f"  backend={dist.get_backend()}, world_size={world_size}, device={device}")
            print(f"  no overlap: {timing[0].item():.2f} ms")
            print(f"  overlap:    {timing[1].item():.2f} ms")
            if timing[0].item() > 0:
                speedup = (1 - timing[1].item() / timing[0].item()) * 100
                print(f"  speedup:    {speedup:.1f}%")
            print("\n  읽는 법:")
            print("    no overlap = compute 후 blocking all_reduce")
            print("    overlap    = async all_reduce를 먼저 걸고 compute 중에 통신 진행")
            print("    실제 speedup은 tensor 크기, compute량, NCCL/Gloo backend, 네트워크에 따라 달라짐")

    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# Part 7: CUDA Stream 사용법
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
# Part 8: Performance Tips
# ============================================================

def performance_tips():
    print("\n" + "=" * 60)
    print("Performance Optimization Checklist")
    print("=" * 60)

    tips = [
        ("Megatron --overlap-grad-reduce",
         "Part 1에 해당. backward 중 gradient bucket이 준비되는 즉시 DP grad reduce를 비동기로 시작."),
        ("Megatron --overlap-param-gather",
         "Part 2에 해당. FSDP/ZeRO-3에서 다음 layer parameter all-gather를 미리 시작."),
        ("Megatron --tp-comm-overlap",
         "Part 3에 해당. TP all-reduce/reduce-scatter/all-gather를 compute stream과 겹침."),
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
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    if mode == "dist":
        run_dist_overlap_demo()
    else:
        simulate_overlap()
        cuda_streams_demo()
        performance_tips()
