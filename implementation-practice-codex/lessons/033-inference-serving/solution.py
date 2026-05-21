"""Inference Serving
===================
Practice deployment-side accounting for LLM inference backends.

The exercise models the ideas behind TRT-LLM, vLLM, and SGLang without requiring
those packages: KV-cache memory, prefill/decode cost, continuous batching, and
backend selection.
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


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def kv_cache_bytes(spec: ModelServingSpec, batch_size: int, sequence_length: int) -> int:
    """Return total KV-cache bytes for batch_size cached sequences."""

    for name, value in (
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("num_layers", spec.num_layers),
        ("hidden_size", spec.hidden_size),
        ("num_attention_heads", spec.num_attention_heads),
        ("num_kv_heads", spec.num_kv_heads),
        ("dtype_bytes", spec.dtype_bytes),
    ):
        _require_positive(name, value)

    if spec.hidden_size % spec.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if spec.num_attention_heads % spec.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")

    head_dim = spec.hidden_size // spec.num_attention_heads
    return int(
        2
        * batch_size
        * sequence_length
        * spec.num_layers
        * spec.num_kv_heads
        * head_dim
        * spec.dtype_bytes
    )


def weight_memory_bytes(spec: ModelServingSpec, tensor_parallel_size: int = 1) -> float:
    """Return per-rank model weight bytes under tensor parallelism."""

    _require_positive("parameter_count", spec.parameter_count)
    _require_positive("dtype_bytes", spec.dtype_bytes)
    _require_positive("tensor_parallel_size", tensor_parallel_size)
    return spec.parameter_count * spec.dtype_bytes / tensor_parallel_size


def max_batch_size_for_kv_cache(
    spec: ModelServingSpec,
    hardware: HardwareSpec,
    sequence_length: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    safety_gb_per_gpu: float = 4.0,
) -> int:
    """Estimate max batch after reserving HBM for weights and safety margin."""

    for name, value in (
        ("gpu_count", hardware.gpu_count),
        ("hbm_gb_per_gpu", hardware.hbm_gb_per_gpu),
        ("sequence_length", sequence_length),
        ("tensor_parallel_size", tensor_parallel_size),
        ("gpu_memory_utilization", gpu_memory_utilization),
    ):
        _require_positive(name, value)
    if tensor_parallel_size > hardware.gpu_count:
        raise ValueError("tensor_parallel_size cannot exceed gpu_count")

    usable_hbm = hardware.hbm_gb_per_gpu * 1e9 * gpu_memory_utilization
    reserved = safety_gb_per_gpu * 1e9
    weights = weight_memory_bytes(spec, tensor_parallel_size)
    available_per_gpu = usable_hbm - reserved - weights
    if available_per_gpu <= 0:
        return 0

    bytes_per_sequence = kv_cache_bytes(spec, batch_size=1, sequence_length=sequence_length)
    # KV cache is sharded across tensor-parallel ranks for this coarse estimate.
    per_gpu_kv = bytes_per_sequence / tensor_parallel_size
    return int(available_per_gpu // per_gpu_kv)


def prefill_decode_estimate(
    spec: ModelServingSpec,
    hardware: HardwareSpec,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    tensor_parallel_size: int = 1,
) -> Dict[str, float]:
    """Estimate prefill/decode latency and throughput with a first-order model."""

    for name, value in (
        ("batch_size", batch_size),
        ("prompt_tokens", prompt_tokens),
        ("decode_tokens", decode_tokens),
        ("gpu_count", hardware.gpu_count),
        ("peak_tflops_per_gpu", hardware.peak_tflops_per_gpu),
        ("hbm_bandwidth_gbps_per_gpu", hardware.hbm_bandwidth_gbps_per_gpu),
    ):
        _require_positive(name, value)

    active_gpus = min(hardware.gpu_count, tensor_parallel_size)
    peak_flops = active_gpus * hardware.peak_tflops_per_gpu * 1e12
    hbm_bandwidth = active_gpus * hardware.hbm_bandwidth_gbps_per_gpu * 1e9
    flops_per_token = 2 * spec.parameter_count

    prefill_tokens = batch_size * prompt_tokens
    prefill_flops = flops_per_token * prefill_tokens
    prefill_latency_s = prefill_flops / peak_flops

    # During decode, each generated token repeatedly touches weights and grows KV reads.
    decode_flops = flops_per_token * batch_size * decode_tokens
    decode_compute_s = decode_flops / peak_flops
    per_step_weight_read_s = weight_memory_bytes(spec, tensor_parallel_size) / hbm_bandwidth
    average_cached_length = prompt_tokens + max(0, decode_tokens - 1) / 2
    kv_read_bytes = kv_cache_bytes(spec, batch_size, int(average_cached_length))
    kv_read_s_per_step = (kv_read_bytes / tensor_parallel_size) / hbm_bandwidth
    decode_latency_s = decode_tokens * max(
        decode_compute_s / decode_tokens,
        per_step_weight_read_s + kv_read_s_per_step,
    )

    total_output_tokens = batch_size * decode_tokens
    total_latency_s = prefill_latency_s + decode_latency_s
    return {
        "prefill_latency_s": prefill_latency_s,
        "decode_latency_s": decode_latency_s,
        "total_latency_s": total_latency_s,
        "prefill_tokens_per_s": prefill_tokens / prefill_latency_s,
        "decode_tokens_per_s": total_output_tokens / decode_latency_s,
        "overall_output_tokens_per_s": total_output_tokens / total_latency_s,
        "decode_bottleneck": "compute" if decode_compute_s / decode_tokens > per_step_weight_read_s + kv_read_s_per_step else "memory",
    }


def continuous_batching_schedule(
    requests: Sequence[Dict[str, int]],
    token_budget_per_step: int,
) -> List[Dict[str, object]]:
    """Simulate a token-budgeted continuous batching loop.

    Each request dict needs id, arrival_step, prompt_tokens, and output_tokens.
    Prefill can be chunked. Decode emits at most one token per active request per step.
    """

    _require_positive("token_budget_per_step", token_budget_per_step)
    pending = [
        {
            "id": request["id"],
            "arrival_step": request["arrival_step"],
            "prompt_remaining": request["prompt_tokens"],
            "decode_remaining": request["output_tokens"],
        }
        for request in requests
    ]
    active: List[Dict[str, int]] = []
    completed: List[int] = []
    timeline: List[Dict[str, object]] = []
    step = 0

    while pending or active:
        arriving = [request for request in pending if request["arrival_step"] <= step]
        pending = [request for request in pending if request["arrival_step"] > step]
        active.extend(arriving)

        budget = token_budget_per_step
        prefill_tokens = 0
        decode_tokens = 0

        for request in active:
            if budget == 0:
                break
            if request["prompt_remaining"] > 0:
                used = min(budget, request["prompt_remaining"])
                request["prompt_remaining"] -= used
                budget -= used
                prefill_tokens += used

        for request in active:
            if budget == 0:
                break
            if request["prompt_remaining"] == 0 and request["decode_remaining"] > 0:
                request["decode_remaining"] -= 1
                budget -= 1
                decode_tokens += 1

        newly_completed = [
            request["id"]
            for request in active
            if request["prompt_remaining"] == 0 and request["decode_remaining"] == 0
        ]
        completed.extend(newly_completed)
        active = [
            request
            for request in active
            if not (request["prompt_remaining"] == 0 and request["decode_remaining"] == 0)
        ]

        timeline.append(
            {
                "step": step,
                "arrived": [request["id"] for request in arriving],
                "active": [request["id"] for request in active],
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "completed": list(completed),
            }
        )
        step += 1

    return timeline


def choose_serving_backend(workload: RequestClass, needs_structured_generation: bool = False) -> Dict[str, str]:
    """Choose a serving backend shape from workload requirements."""

    if needs_structured_generation:
        return {
            "backend": "SGLang",
            "reason": "structured generation and programmatic control are first-class concerns",
        }
    if workload.max_concurrency >= 64 or workload.requests_per_second >= 20:
        return {
            "backend": "vLLM",
            "reason": "continuous batching and paged KV cache matter most for this traffic shape",
        }
    if workload.prompt_tokens + workload.output_tokens <= 2048 and workload.max_concurrency <= 16:
        return {
            "backend": "TensorRT-LLM",
            "reason": "latency-oriented optimized kernels and engine builds are a good fit",
        }
    return {
        "backend": "NeMo export plus benchmark both TRT-LLM and vLLM",
        "reason": "mixed workload; measure latency target and batching efficiency before committing",
    }


def _fmt_gb(value: float) -> str:
    return f"{value / 1e9:.2f} GB"


def _fmt_ms(value: float) -> str:
    return f"{value * 1000:.2f} ms"


def demo() -> None:
    model = ModelServingSpec(
        name="gqa-7b",
        parameter_count=7e9,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
    )
    hardware = HardwareSpec(
        name="4xH100",
        gpu_count=4,
        hbm_gb_per_gpu=80,
        peak_tflops_per_gpu=990,
        hbm_bandwidth_gbps_per_gpu=3350,
    )

    print("=" * 72)
    print("Serving Memory")
    print("=" * 72)
    print(f"weights per TP rank: {_fmt_gb(weight_memory_bytes(model, tensor_parallel_size=4))}")
    print(f"KV cache batch=32 seq=8192: {_fmt_gb(kv_cache_bytes(model, 32, 8192))}")
    print(
        "max batch @ 8k context:",
        max_batch_size_for_kv_cache(model, hardware, sequence_length=8192, tensor_parallel_size=4),
    )

    print("\n" + "=" * 72)
    print("Prefill vs Decode")
    print("=" * 72)
    estimate = prefill_decode_estimate(
        model,
        hardware,
        batch_size=16,
        prompt_tokens=2048,
        decode_tokens=256,
        tensor_parallel_size=4,
    )
    print(f"prefill: {_fmt_ms(estimate['prefill_latency_s'])}")
    print(f"decode:  {_fmt_ms(estimate['decode_latency_s'])}")
    print(f"decode bottleneck: {estimate['decode_bottleneck']}")
    print(f"output tok/s: {estimate['overall_output_tokens_per_s']:.0f}")

    print("\n" + "=" * 72)
    print("Continuous Batching")
    print("=" * 72)
    requests = [
        {"id": 1, "arrival_step": 0, "prompt_tokens": 8, "output_tokens": 3},
        {"id": 2, "arrival_step": 1, "prompt_tokens": 4, "output_tokens": 4},
        {"id": 3, "arrival_step": 2, "prompt_tokens": 2, "output_tokens": 2},
    ]
    for row in continuous_batching_schedule(requests, token_budget_per_step=8):
        print(row)

    print("\n" + "=" * 72)
    print("Backend Choice")
    print("=" * 72)
    workload = RequestClass(
        name="chat-high-throughput",
        prompt_tokens=1024,
        output_tokens=256,
        requests_per_second=30,
        max_concurrency=128,
    )
    print(choose_serving_backend(workload))


if __name__ == "__main__":
    demo()
