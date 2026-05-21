"""Performance Debugging
=======================
Practice profiler-driven performance triage for distributed training.

This is a synthetic trace analyzer. It does not need CUDA. It helps turn profiler
events into bottleneck classes and concrete tuning actions.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
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


def union_interval_length(intervals: Sequence[Tuple[float, float]]) -> float:
    """Return total length covered by possibly overlapping intervals."""
    raise NotImplementedError("TODO: implement union_interval_length; compare with solution.py only after trying.")


def exposed_communication(trace: StepTrace) -> Dict[str, float]:
    """Return total, overlapped, and exposed communication time."""
    raise NotImplementedError("TODO: implement exposed_communication; compare with solution.py only after trying.")


def summarize_step(trace: StepTrace, peak_tflops: float, hbm_bandwidth_gbps: float) -> Dict[str, object]:
    """Compute utilization-style summary metrics for one training step."""
    raise NotImplementedError("TODO: implement summarize_step; compare with solution.py only after trying.")


def classify_bottlenecks(summary: Dict[str, object]) -> List[str]:
    """Turn summary metrics into coarse bottleneck labels."""
    raise NotImplementedError("TODO: implement classify_bottlenecks; compare with solution.py only after trying.")


def recommend_tuning_actions(summary: Dict[str, object]) -> List[str]:
    """Return concrete next actions from measured bottleneck classes."""
    raise NotImplementedError("TODO: implement recommend_tuning_actions; compare with solution.py only after trying.")


def compare_runs(baseline: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, float]:
    """Compare two summarized runs and report percentage changes."""
    raise NotImplementedError("TODO: implement compare_runs; compare with solution.py only after trying.")


def demo() -> None:
    raise NotImplementedError("TODO: implement demo; compare with solution.py only after trying.")


if __name__ == "__main__":
    demo()
