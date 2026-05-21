"""Inference Serving
===================
Practice deployment-side accounting for LLM inference backends.

The exercise models the ideas behind TRT-LLM, vLLM, and SGLang without requiring
those packages: KV-cache memory, prefill/decode cost, continuous batching, and
backend selection.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ModelServingSpec:
    name: str
    parameter_count: float
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    dtype_bytes: int = 2


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    gpu_count: int
    hbm_gb_per_gpu: float
    peak_tflops_per_gpu: float
    hbm_bandwidth_gbps_per_gpu: float


@dataclass(frozen=True)
class RequestClass:
    name: str
    prompt_tokens: int
    output_tokens: int
    requests_per_second: float
    max_concurrency: int


def kv_cache_bytes(spec: ModelServingSpec, batch_size: int, sequence_length: int) -> int:
    """Return total KV-cache bytes for batch_size cached sequences."""
    raise NotImplementedError("TODO: implement kv_cache_bytes; compare with solution.py only after trying.")


def weight_memory_bytes(spec: ModelServingSpec, tensor_parallel_size: int = 1) -> float:
    """Return per-rank model weight bytes under tensor parallelism."""
    raise NotImplementedError("TODO: implement weight_memory_bytes; compare with solution.py only after trying.")


def max_batch_size_for_kv_cache(
    spec: ModelServingSpec,
    hardware: HardwareSpec,
    sequence_length: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    safety_gb_per_gpu: float = 4.0,
) -> int:
    """Estimate max batch after reserving HBM for weights and safety margin."""
    raise NotImplementedError("TODO: implement max_batch_size_for_kv_cache; compare with solution.py only after trying.")


def prefill_decode_estimate(
    spec: ModelServingSpec,
    hardware: HardwareSpec,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    tensor_parallel_size: int = 1,
) -> Dict[str, float]:
    """Estimate prefill/decode latency and throughput with a first-order model."""
    raise NotImplementedError("TODO: implement prefill_decode_estimate; compare with solution.py only after trying.")


def continuous_batching_schedule(
    requests: Sequence[Dict[str, int]],
    token_budget_per_step: int,
) -> List[Dict[str, object]]:
    """Simulate a token-budgeted continuous batching loop."""
    raise NotImplementedError("TODO: implement continuous_batching_schedule; compare with solution.py only after trying.")


def choose_serving_backend(workload: RequestClass, needs_structured_generation: bool = False) -> Dict[str, str]:
    """Choose a serving backend shape from workload requirements."""
    raise NotImplementedError("TODO: implement choose_serving_backend; compare with solution.py only after trying.")


def demo() -> None:
    raise NotImplementedError("TODO: implement demo; compare with solution.py only after trying.")


if __name__ == "__main__":
    demo()
