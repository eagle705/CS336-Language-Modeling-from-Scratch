"""Scaling Book Systems Math
===========================
Practice estimates inspired by "How To Scale Your Model".

The goal is not an exact performance model. The goal is to make the first-order
quantities explicit: FLOPs, bytes, lower/upper roofline bounds, and which state is
replicated or sharded.
"""

from dataclasses import dataclass


BYTES_PER_BF16 = 2
BYTES_PER_FP32 = 4


@dataclass(frozen=True)
class RooflineEstimate:
    """Timing bounds for one operation or communication phase."""

    math_time_s: float
    communication_time_s: float
    lower_bound_s: float
    upper_bound_s: float
    bottleneck: str


def _require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def roofline_estimate(
    compute_flops,
    communication_bytes,
    peak_flops_per_s,
    bandwidth_bytes_per_s,
):
    """Estimate math/comms time and lower/upper bounds.

    If math and communication overlap perfectly, the lower bound is max(math,
    comms). If nothing overlaps, the upper bound is math + comms.
    """

    _require_positive("peak_flops_per_s", peak_flops_per_s)
    _require_positive("bandwidth_bytes_per_s", bandwidth_bytes_per_s)
    if compute_flops < 0 or communication_bytes < 0:
        raise ValueError("compute_flops and communication_bytes must be non-negative")

    math_time_s = compute_flops / peak_flops_per_s
    communication_time_s = communication_bytes / bandwidth_bytes_per_s
    lower_bound_s = max(math_time_s, communication_time_s)
    upper_bound_s = math_time_s + communication_time_s

    if math_time_s > communication_time_s:
        bottleneck = "compute"
    elif communication_time_s > math_time_s:
        bottleneck = "bandwidth"
    else:
        bottleneck = "balanced"

    return RooflineEstimate(
        math_time_s=math_time_s,
        communication_time_s=communication_time_s,
        lower_bound_s=lower_bound_s,
        upper_bound_s=upper_bound_s,
        bottleneck=bottleneck,
    )


def matmul_accounting(m, n, k, bytes_per_element=BYTES_PER_BF16):
    """Return FLOPs, bytes, and arithmetic intensity for [m, k] @ [k, n]."""

    for name, value in (("m", m), ("n", n), ("k", k), ("bytes_per_element", bytes_per_element)):
        _require_positive(name, value)

    flops = 2 * m * n * k
    bytes_read = (m * k + k * n) * bytes_per_element
    bytes_written = m * n * bytes_per_element
    total_bytes = bytes_read + bytes_written
    arithmetic_intensity = flops / total_bytes

    return {
        "flops": flops,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "total_bytes": total_bytes,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
    }


def training_flops(parameter_count, token_count, activation_checkpointing=False):
    """Estimate dense Transformer training FLOPs with the common 6 * P * T rule."""

    _require_positive("parameter_count", parameter_count)
    _require_positive("token_count", token_count)

    forward_flops = 2 * parameter_count * token_count
    backward_flops = 4 * parameter_count * token_count
    recompute_flops = forward_flops if activation_checkpointing else 0
    total_flops = forward_flops + backward_flops + recompute_flops

    return {
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "recompute_flops": recompute_flops,
        "total_flops": total_flops,
        "flops_per_token": total_flops / token_count,
    }


def kv_cache_bytes(
    batch_size,
    sequence_length,
    num_layers,
    num_kv_heads,
    head_dim,
    bytes_per_element=BYTES_PER_BF16,
):
    """Return KV-cache bytes for a decoder-only Transformer.

    KV cache stores both keys and values for every layer and cached token.
    """

    for name, value in (
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("num_layers", num_layers),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("bytes_per_element", bytes_per_element),
    ):
        _require_positive(name, value)

    return 2 * batch_size * sequence_length * num_layers * num_kv_heads * head_dim * bytes_per_element


def decode_step_estimate(
    parameter_count,
    batch_size,
    sequence_length,
    num_layers,
    num_kv_heads,
    head_dim,
    num_chips,
    peak_flops_per_chip,
    hbm_bandwidth_per_chip,
    bytes_per_param=BYTES_PER_BF16,
    bytes_per_kv=BYTES_PER_BF16,
):
    """Estimate one-token generation latency from parameter loading, FLOPs, and KV reads."""

    for name, value in (
        ("parameter_count", parameter_count),
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("num_chips", num_chips),
        ("peak_flops_per_chip", peak_flops_per_chip),
        ("hbm_bandwidth_per_chip", hbm_bandwidth_per_chip),
    ):
        _require_positive(name, value)

    total_peak_flops = num_chips * peak_flops_per_chip
    total_hbm_bandwidth = num_chips * hbm_bandwidth_per_chip

    parameter_bytes = parameter_count * bytes_per_param
    linear_flops = 2 * parameter_count * batch_size
    kv_bytes = kv_cache_bytes(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_kv,
    )

    linear_math_time = linear_flops / total_peak_flops
    parameter_load_time = parameter_bytes / total_hbm_bandwidth
    attention_kv_time = kv_bytes / total_hbm_bandwidth
    linear_time = max(linear_math_time, parameter_load_time)
    latency_s = linear_time + attention_kv_time
    throughput_tokens_per_s = batch_size / latency_s

    return {
        "parameter_bytes": parameter_bytes,
        "kv_cache_bytes_read": kv_bytes,
        "linear_math_time_s": linear_math_time,
        "parameter_load_time_s": parameter_load_time,
        "attention_kv_time_s": attention_kv_time,
        "latency_s": latency_s,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "linear_bottleneck": "compute" if linear_math_time > parameter_load_time else "bandwidth",
    }


def parallelism_hint(
    parameter_count,
    global_tokens_per_step,
    num_chips,
    hbm_bytes_per_chip,
    node_size=8,
):
    """Suggest a coarse parallelism strategy from model size and hardware shape."""

    for name, value in (
        ("parameter_count", parameter_count),
        ("global_tokens_per_step", global_tokens_per_step),
        ("num_chips", num_chips),
        ("hbm_bytes_per_chip", hbm_bytes_per_chip),
        ("node_size", node_size),
    ):
        _require_positive(name, value)

    param_bytes = parameter_count * BYTES_PER_BF16
    training_state_bytes = parameter_count * (BYTES_PER_BF16 + BYTES_PER_BF16 + 2 * BYTES_PER_FP32)
    fits_one_chip_for_training = training_state_bytes < 0.8 * hbm_bytes_per_chip
    chips_per_node = min(num_chips, node_size)
    enough_tokens_per_chip = (global_tokens_per_step / num_chips) >= 1024

    if fits_one_chip_for_training and enough_tokens_per_chip:
        strategy = "pure data parallelism"
        reason = "model state and per-chip work are both large enough for simple replication"
    elif num_chips <= chips_per_node:
        strategy = "tensor parallelism inside one node"
        reason = "state is too large for clean replication, but fast intra-node links can carry TP collectives"
    elif parameter_count >= 70e9 and num_chips >= 64:
        strategy = "TP within nodes + PP across model depth + FSDP/ZeRO across replicas"
        reason = "large dense models usually need model sharding plus optimizer-state sharding"
    else:
        strategy = "FSDP/ZeRO across data replicas, with optional small TP group"
        reason = "optimizer and parameter states need sharding, while DP communication can use slower links"

    return {
        "strategy": strategy,
        "reason": reason,
        "param_gb": param_bytes / 1e9,
        "training_state_gb": training_state_bytes / 1e9,
        "tokens_per_chip": global_tokens_per_step / num_chips,
    }


def _fmt_ms(seconds):
    return f"{seconds * 1000:.3f} ms"


def demo():
    print("=" * 72)
    print("Roofline Estimate")
    print("=" * 72)
    h100_peak = 9.89e14
    h100_hbm = 3.35e12
    mm = matmul_accounting(m=4096, n=4096, k=4096)
    estimate = roofline_estimate(
        compute_flops=mm["flops"],
        communication_bytes=mm["total_bytes"],
        peak_flops_per_s=h100_peak,
        bandwidth_bytes_per_s=h100_hbm,
    )
    print(f"Matmul FLOPs: {mm['flops'] / 1e12:.3f} TFLOPs")
    print(f"Arithmetic intensity: {mm['arithmetic_intensity_flops_per_byte']:.1f} FLOPs/byte")
    print(f"Lower bound: {_fmt_ms(estimate.lower_bound_s)} ({estimate.bottleneck}-bound)")

    print("\n" + "=" * 72)
    print("Training FLOPs")
    print("=" * 72)
    train = training_flops(parameter_count=7e9, token_count=140e9, activation_checkpointing=True)
    print(f"7B x 140B tokens with recompute: {train['total_flops'] / 1e21:.2f} ZFLOPs")
    print(f"FLOPs/token: {train['flops_per_token'] / 1e9:.1f}B")

    print("\n" + "=" * 72)
    print("Inference Decode Step")
    print("=" * 72)
    decode = decode_step_estimate(
        parameter_count=13e9,
        batch_size=32,
        sequence_length=8192,
        num_layers=40,
        num_kv_heads=40,
        head_dim=128,
        num_chips=8,
        peak_flops_per_chip=h100_peak,
        hbm_bandwidth_per_chip=h100_hbm,
        bytes_per_param=1,
        bytes_per_kv=2,
    )
    print(f"Latency: {_fmt_ms(decode['latency_s'])}")
    print(f"Throughput: {decode['throughput_tokens_per_s']:.0f} tokens/s")
    print(f"Linear bottleneck: {decode['linear_bottleneck']}")
    print(f"KV read: {decode['kv_cache_bytes_read'] / 1e9:.1f} GB")

    print("\n" + "=" * 72)
    print("Parallelism Hint")
    print("=" * 72)
    hint = parallelism_hint(
        parameter_count=70e9,
        global_tokens_per_step=4_000_000,
        num_chips=256,
        hbm_bytes_per_chip=80 * 2**30,
    )
    print(hint["strategy"])
    print(hint["reason"])
    print(f"Training state: {hint['training_state_gb']:.0f} GB, tokens/chip: {hint['tokens_per_chip']:.0f}")


if __name__ == "__main__":
    demo()
