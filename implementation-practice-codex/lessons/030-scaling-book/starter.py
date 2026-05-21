"""Scaling Book Systems Math
===========================
Practice estimates inspired by "How To Scale Your Model".

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass.
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


def roofline_estimate(
    compute_flops,
    communication_bytes,
    peak_flops_per_s,
    bandwidth_bytes_per_s,
):
    """Estimate math/comms time and lower/upper bounds."""
    raise NotImplementedError("TODO: implement roofline_estimate; compare with solution.py only after trying.")


def matmul_accounting(m, n, k, bytes_per_element=BYTES_PER_BF16):
    """Return FLOPs, bytes, and arithmetic intensity for [m, k] @ [k, n]."""
    raise NotImplementedError("TODO: implement matmul_accounting; compare with solution.py only after trying.")


def training_flops(parameter_count, token_count, activation_checkpointing=False):
    """Estimate dense Transformer training FLOPs with the common 6 * P * T rule."""
    raise NotImplementedError("TODO: implement training_flops; compare with solution.py only after trying.")


def kv_cache_bytes(
    batch_size,
    sequence_length,
    num_layers,
    num_kv_heads,
    head_dim,
    bytes_per_element=BYTES_PER_BF16,
):
    """Return KV-cache bytes for a decoder-only Transformer."""
    raise NotImplementedError("TODO: implement kv_cache_bytes; compare with solution.py only after trying.")


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
    raise NotImplementedError("TODO: implement decode_step_estimate; compare with solution.py only after trying.")


def parallelism_hint(
    parameter_count,
    global_tokens_per_step,
    num_chips,
    hbm_bytes_per_chip,
    node_size=8,
):
    """Suggest a coarse parallelism strategy from model size and hardware shape."""
    raise NotImplementedError("TODO: implement parallelism_hint; compare with solution.py only after trying.")


def demo():
    raise NotImplementedError("TODO: implement demo; compare with solution.py only after trying.")


if __name__ == "__main__":
    demo()
