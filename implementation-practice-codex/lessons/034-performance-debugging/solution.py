"""Performance Debugging
=======================
Practice profiler-driven performance triage for distributed training.

This is a synthetic trace analyzer. It does not need CUDA. It helps turn profiler
events into bottleneck classes and concrete tuning actions.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ProfilerEvent:
    name: str
    category: str
    start_ms: float
    duration_ms: float
    flops: float = 0.0
    bytes_moved: float = 0.0
    stream: str = "default"


@dataclass(frozen=True)
class StepTrace:
    step_time_ms: float
    events: Sequence[ProfilerEvent]
    dataloader_ms: float = 0.0
    cpu_launch_ms: float = 0.0


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _event_interval(event: ProfilerEvent) -> Tuple[float, float]:
    _require_non_negative("start_ms", event.start_ms)
    _require_non_negative("duration_ms", event.duration_ms)
    return event.start_ms, event.start_ms + event.duration_ms


def union_interval_length(intervals: Sequence[Tuple[float, float]]) -> float:
    """Return total length covered by possibly overlapping intervals."""

    if not intervals:
        return 0.0
    normalized = []
    for start, end in intervals:
        if end < start:
            raise ValueError("interval end must be >= start")
        normalized.append((start, end))

    normalized.sort()
    total = 0.0
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _overlap_length(interval: Tuple[float, float], blockers: Sequence[Tuple[float, float]]) -> float:
    start, end = interval
    overlaps = []
    for blocker_start, blocker_end in blockers:
        overlap_start = max(start, blocker_start)
        overlap_end = min(end, blocker_end)
        if overlap_end > overlap_start:
            overlaps.append((overlap_start, overlap_end))
    return union_interval_length(overlaps)


def exposed_communication(trace: StepTrace) -> Dict[str, float]:
    """Return total, overlapped, and exposed communication time."""

    compute_intervals = [
        _event_interval(event)
        for event in trace.events
        if event.category in {"compute", "memory"}
    ]
    comm_events = [event for event in trace.events if event.category == "comm"]
    total_comm = sum(event.duration_ms for event in comm_events)
    overlapped = sum(_overlap_length(_event_interval(event), compute_intervals) for event in comm_events)
    exposed = max(0.0, total_comm - overlapped)
    overlap_fraction = overlapped / total_comm if total_comm else 1.0
    return {
        "total_comm_ms": total_comm,
        "overlapped_comm_ms": overlapped,
        "exposed_comm_ms": exposed,
        "comm_overlap_fraction": overlap_fraction,
    }


def summarize_step(trace: StepTrace, peak_tflops: float, hbm_bandwidth_gbps: float) -> Dict[str, object]:
    """Compute utilization-style summary metrics for one training step."""

    if trace.step_time_ms <= 0:
        raise ValueError("step_time_ms must be positive")
    if peak_tflops <= 0 or hbm_bandwidth_gbps <= 0:
        raise ValueError("peak_tflops and hbm_bandwidth_gbps must be positive")

    total_flops = sum(event.flops for event in trace.events)
    total_bytes = sum(event.bytes_moved for event in trace.events)
    compute_ms = sum(event.duration_ms for event in trace.events if event.category == "compute")
    memory_ms = sum(event.duration_ms for event in trace.events if event.category == "memory")
    input_ms = trace.dataloader_ms + sum(event.duration_ms for event in trace.events if event.category == "input")
    tiny_kernel_count = sum(1 for event in trace.events if event.duration_ms < 0.05 and event.category != "comm")
    comm = exposed_communication(trace)

    achieved_tflops = total_flops / (trace.step_time_ms / 1000.0) / 1e12
    mfu = achieved_tflops / peak_tflops
    achieved_bandwidth_gbps = total_bytes / (trace.step_time_ms / 1000.0) / 1e9
    hbm_utilization = achieved_bandwidth_gbps / hbm_bandwidth_gbps
    top_events = sorted(trace.events, key=lambda event: event.duration_ms, reverse=True)[:5]

    return {
        "step_time_ms": trace.step_time_ms,
        "achieved_tflops": achieved_tflops,
        "mfu": mfu,
        "achieved_bandwidth_gbps": achieved_bandwidth_gbps,
        "hbm_utilization": hbm_utilization,
        "compute_fraction": compute_ms / trace.step_time_ms,
        "memory_fraction": memory_ms / trace.step_time_ms,
        "input_fraction": input_ms / trace.step_time_ms,
        "cpu_launch_fraction": trace.cpu_launch_ms / trace.step_time_ms,
        "tiny_kernel_count": tiny_kernel_count,
        "top_events": [(event.name, event.category, event.duration_ms) for event in top_events],
        **comm,
    }


def classify_bottlenecks(summary: Dict[str, object]) -> List[str]:
    """Turn summary metrics into coarse bottleneck labels."""

    labels: List[str] = []
    if summary["input_fraction"] > 0.10:
        labels.append("input_pipeline")
    if summary["cpu_launch_fraction"] > 0.05 or summary["tiny_kernel_count"] >= 10:
        labels.append("kernel_launch_or_fusion")
    if summary["exposed_comm_ms"] / summary["step_time_ms"] > 0.08:
        labels.append("exposed_communication")
    if summary["hbm_utilization"] > 0.60 and summary["mfu"] < 0.35:
        labels.append("memory_bandwidth")
    if summary["mfu"] < 0.30 and not labels:
        labels.append("low_compute_utilization")
    if not labels:
        labels.append("no_obvious_single_bottleneck")
    return labels


def recommend_tuning_actions(summary: Dict[str, object]) -> List[str]:
    """Return concrete next actions from measured bottleneck classes."""

    actions: List[str] = []
    labels = classify_bottlenecks(summary)
    if "input_pipeline" in labels:
        actions.append("profile dataset transforms; add workers, pinned memory, prefetch, or offline preprocessing")
    if "kernel_launch_or_fusion" in labels:
        actions.append("try torch.compile, fused LayerNorm/RMSNorm, fused optimizer, or larger fused kernels")
    if "exposed_communication" in labels:
        actions.append("inspect bucket sizes and schedule overlap; move TP inside NVLink and overlap DP/FSDP collectives")
    if "memory_bandwidth" in labels:
        actions.append("check activation recompute, sequence parallelism, kernel fusion, and tensor-core-friendly shapes")
    if "low_compute_utilization" in labels:
        actions.append("verify mixed precision, matmul dimensions, global batch, and profiler warmup")
    if "no_obvious_single_bottleneck" in labels:
        actions.append("compare against a nearby baseline and inspect top events before changing the recipe")
    return actions


def compare_runs(baseline: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, float]:
    """Compare two summarized runs and report percentage changes."""

    base_step = baseline["step_time_ms"]
    cand_step = candidate["step_time_ms"]
    if base_step <= 0 or cand_step <= 0:
        raise ValueError("step times must be positive")

    return {
        "step_time_change_pct": 100.0 * (cand_step - base_step) / base_step,
        "throughput_change_pct": 100.0 * ((base_step / cand_step) - 1.0),
        "mfu_change_points": 100.0 * (candidate["mfu"] - baseline["mfu"]),
        "exposed_comm_change_ms": candidate["exposed_comm_ms"] - baseline["exposed_comm_ms"],
    }


def demo() -> None:
    trace = StepTrace(
        step_time_ms=120.0,
        dataloader_ms=6.0,
        cpu_launch_ms=8.0,
        events=[
            ProfilerEvent("qkv_matmul", "compute", 0.0, 24.0, flops=90e12, bytes_moved=80e9),
            ProfilerEvent("attention_softmax", "memory", 24.0, 10.0, flops=4e12, bytes_moved=220e9),
            ProfilerEvent("tp_all_reduce", "comm", 28.0, 18.0, bytes_moved=64e9),
            ProfilerEvent("mlp_matmul", "compute", 46.0, 34.0, flops=130e12, bytes_moved=120e9),
            ProfilerEvent("dp_reduce_scatter", "comm", 82.0, 20.0, bytes_moved=96e9),
            ProfilerEvent("optimizer", "memory", 102.0, 8.0, flops=2e12, bytes_moved=180e9),
            *[
                ProfilerEvent(f"tiny_elementwise_{idx}", "memory", 110.0 + idx * 0.03, 0.03, bytes_moved=1e9)
                for idx in range(12)
            ],
        ],
    )
    summary = summarize_step(trace, peak_tflops=8 * 990, hbm_bandwidth_gbps=8 * 3350)

    print("=" * 72)
    print("Step Summary")
    print("=" * 72)
    for key in (
        "step_time_ms",
        "mfu",
        "hbm_utilization",
        "input_fraction",
        "cpu_launch_fraction",
        "exposed_comm_ms",
        "comm_overlap_fraction",
        "tiny_kernel_count",
    ):
        print(f"{key:<24} {summary[key]}")
    print("top events:", summary["top_events"])

    print("\n" + "=" * 72)
    print("Bottlenecks")
    print("=" * 72)
    print(classify_bottlenecks(summary))
    for action in recommend_tuning_actions(summary):
        print("-", action)

    improved = dict(summary)
    improved["step_time_ms"] = 102.0
    improved["mfu"] = 0.36
    improved["exposed_comm_ms"] = 7.0
    print("\n" + "=" * 72)
    print("Run Comparison")
    print("=" * 72)
    print(compare_runs(summary, improved))


if __name__ == "__main__":
    demo()
