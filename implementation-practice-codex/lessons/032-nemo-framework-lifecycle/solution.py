"""NeMo Framework Lifecycle
==========================
Practice a NeMo-like end-to-end training lifecycle plan around Megatron Core.

The goal is not to reimplement NeMo. The goal is to make the API boundaries
explicit: model recipe, parallel recipe, lifecycle stages, launch arguments, and
checkpoint compatibility.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ModelRecipe:
    model_name: str
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    tie_word_embeddings: bool = True
    precision: str = "bf16"


@dataclass(frozen=True)
class ParallelRecipe:
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    data_parallel_size: int
    micro_batch_size: int
    global_batch_size: int
    sequence_length: int
    virtual_pipeline_model_parallel_size: int = 1


@dataclass(frozen=True)
class LifecycleStage:
    name: str
    inputs: Sequence[str]
    outputs: Sequence[str]
    owner: str
    acceptance_checks: Sequence[str]


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _world_size(parallel: ParallelRecipe) -> int:
    return (
        parallel.tensor_model_parallel_size
        * parallel.pipeline_model_parallel_size
        * parallel.context_parallel_size
        * parallel.data_parallel_size
    )


def estimate_dense_parameters(recipe: ModelRecipe) -> Dict[str, float]:
    """Estimate dense decoder-only parameter counts by component.

    This is a first-order accounting model. It captures GQA by making K/V
    projections smaller when num_kv_heads < num_attention_heads.
    """

    for name, value in (
        ("num_layers", recipe.num_layers),
        ("hidden_size", recipe.hidden_size),
        ("ffn_hidden_size", recipe.ffn_hidden_size),
        ("num_attention_heads", recipe.num_attention_heads),
        ("num_kv_heads", recipe.num_kv_heads),
        ("vocab_size", recipe.vocab_size),
    ):
        _require_positive(name, value)

    if recipe.hidden_size % recipe.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if recipe.num_attention_heads % recipe.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")

    head_dim = recipe.hidden_size // recipe.num_attention_heads
    q_projection = recipe.hidden_size * recipe.hidden_size
    kv_projection = 2 * recipe.hidden_size * recipe.num_kv_heads * head_dim
    output_projection = recipe.hidden_size * recipe.hidden_size
    attention_per_layer = q_projection + kv_projection + output_projection

    # SwiGLU-style MLP has gate, up, and down projections.
    mlp_per_layer = 3 * recipe.hidden_size * recipe.ffn_hidden_size
    norm_per_layer = 2 * recipe.hidden_size
    transformer_layers = recipe.num_layers * (attention_per_layer + mlp_per_layer + norm_per_layer)

    embeddings = recipe.vocab_size * recipe.hidden_size
    lm_head = 0 if recipe.tie_word_embeddings else embeddings
    final_norm = recipe.hidden_size
    total = transformer_layers + embeddings + lm_head + final_norm

    return {
        "attention": float(recipe.num_layers * attention_per_layer),
        "mlp": float(recipe.num_layers * mlp_per_layer),
        "norms": float(recipe.num_layers * norm_per_layer + final_norm),
        "embeddings": float(embeddings),
        "lm_head": float(lm_head),
        "total": float(total),
        "total_billions": total / 1e9,
    }


def derive_gradient_accumulation_steps(parallel: ParallelRecipe) -> int:
    """Return global_batch / (micro_batch * data_parallel)."""

    for name, value in (
        ("micro_batch_size", parallel.micro_batch_size),
        ("global_batch_size", parallel.global_batch_size),
        ("data_parallel_size", parallel.data_parallel_size),
    ):
        _require_positive(name, value)

    denominator = parallel.micro_batch_size * parallel.data_parallel_size
    if parallel.global_batch_size % denominator != 0:
        raise ValueError(
            "global_batch_size must be divisible by micro_batch_size * data_parallel_size"
        )
    return parallel.global_batch_size // denominator


def validate_parallel_recipe(recipe: ModelRecipe, parallel: ParallelRecipe) -> Dict[str, object]:
    """Validate divisibility and launch invariants before a distributed run."""

    issues: List[str] = []
    warnings: List[str] = []

    for name, value in (
        ("tensor_model_parallel_size", parallel.tensor_model_parallel_size),
        ("pipeline_model_parallel_size", parallel.pipeline_model_parallel_size),
        ("context_parallel_size", parallel.context_parallel_size),
        ("data_parallel_size", parallel.data_parallel_size),
        ("micro_batch_size", parallel.micro_batch_size),
        ("global_batch_size", parallel.global_batch_size),
        ("sequence_length", parallel.sequence_length),
        ("virtual_pipeline_model_parallel_size", parallel.virtual_pipeline_model_parallel_size),
    ):
        if value <= 0:
            issues.append(f"{name} must be positive")

    positive_parallel = not any(
        value <= 0
        for value in (
            parallel.tensor_model_parallel_size,
            parallel.pipeline_model_parallel_size,
            parallel.context_parallel_size,
            parallel.data_parallel_size,
            parallel.micro_batch_size,
            parallel.global_batch_size,
            parallel.sequence_length,
            parallel.virtual_pipeline_model_parallel_size,
        )
    )

    if recipe.hidden_size % recipe.num_attention_heads != 0:
        issues.append("hidden_size must be divisible by num_attention_heads")
    if positive_parallel:
        if recipe.num_attention_heads % parallel.tensor_model_parallel_size != 0:
            issues.append("num_attention_heads must be divisible by tensor_model_parallel_size")
        if recipe.num_kv_heads % parallel.tensor_model_parallel_size != 0:
            warnings.append("num_kv_heads is not divisible by TP; GQA kernels may need special handling")
        if recipe.num_layers % (
            parallel.pipeline_model_parallel_size * parallel.virtual_pipeline_model_parallel_size
        ) != 0:
            issues.append("num_layers must divide evenly across PP * virtual PP")
        if parallel.sequence_length % parallel.context_parallel_size != 0:
            issues.append("sequence_length must be divisible by context_parallel_size")

    try:
        gradient_accumulation_steps = derive_gradient_accumulation_steps(parallel)
    except ValueError as exc:
        gradient_accumulation_steps = None
        issues.append(str(exc))

    if parallel.pipeline_model_parallel_size > 1 and gradient_accumulation_steps == 1:
        warnings.append("PP with one microbatch has a large bubble; increase gradient accumulation")
    if recipe.precision.lower() not in {"fp32", "fp16", "bf16", "fp8"}:
        issues.append("precision must be one of fp32, fp16, bf16, fp8")

    return {
        "valid": not issues,
        "world_size": _world_size(parallel),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "issues": issues,
        "warnings": warnings,
    }


def build_lifecycle_plan(recipe: ModelRecipe, parallel: ParallelRecipe) -> List[LifecycleStage]:
    """Return an end-to-end training and deployment lifecycle plan."""

    validation = validate_parallel_recipe(recipe, parallel)
    if not validation["valid"]:
        raise ValueError(f"invalid recipe: {validation['issues']}")

    return [
        LifecycleStage(
            name="data_preprocess",
            inputs=["raw text or multimodal manifests", "tokenizer config"],
            outputs=["indexed dataset", "sampling weights", "data quality report"],
            owner="data pipeline",
            acceptance_checks=["deterministic shards", "documented filtering", "resume-safe offsets"],
        ),
        LifecycleStage(
            name="pretrain",
            inputs=["indexed dataset", "model recipe", "parallel recipe"],
            outputs=["distributed checkpoints", "training metrics", "loss-scale logs"],
            owner="Megatron Core trainer",
            acceptance_checks=["stable MFU", "no overflow storm", "checkpoint resume works"],
        ),
        LifecycleStage(
            name="alignment_or_customization",
            inputs=["base checkpoint", "SFT or preference data", "adapter/full-finetune policy"],
            outputs=["customized checkpoint", "regression evals"],
            owner="NeMo alignment workflow",
            acceptance_checks=["eval improvement", "no format drift", "reproducible config diff"],
        ),
        LifecycleStage(
            name="evaluation",
            inputs=["candidate checkpoint", "task harness", "safety and robustness tests"],
            outputs=["scorecard", "failure examples", "promotion decision"],
            owner="evaluation pipeline",
            acceptance_checks=["versioned benchmark set", "human-readable error slices"],
        ),
        LifecycleStage(
            name="checkpoint_reshard",
            inputs=["training checkpoint", "target serving or finetune parallel recipe"],
            outputs=["resharded checkpoint", "load manifest"],
            owner="distributed checkpointing",
            acceptance_checks=["tensor shapes match target", "optimizer state policy is explicit"],
        ),
        LifecycleStage(
            name="deployment_handoff",
            inputs=["approved checkpoint", "tokenizer", "serving constraints"],
            outputs=["export bundle", "latency/throughput target", "rollback plan"],
            owner="inference platform",
            acceptance_checks=["KV-cache budget fits", "backend selected", "canary metrics defined"],
        ),
    ]


def build_nemo_launcher(recipe: ModelRecipe, parallel: ParallelRecipe, config_path: str) -> List[str]:
    """Build a dry-run torchrun command for a NeMo/Megatron-style job."""

    validation = validate_parallel_recipe(recipe, parallel)
    if not validation["valid"]:
        raise ValueError(f"invalid recipe: {validation['issues']}")

    return [
        "torchrun",
        f"--nproc_per_node={min(_world_size(parallel), 8)}",
        "train.py",
        f"--config-path={config_path}",
        f"model.name={recipe.model_name}",
        f"model.num_layers={recipe.num_layers}",
        f"model.hidden_size={recipe.hidden_size}",
        f"model.ffn_hidden_size={recipe.ffn_hidden_size}",
        f"model.num_attention_heads={recipe.num_attention_heads}",
        f"model.num_kv_heads={recipe.num_kv_heads}",
        f"trainer.precision={recipe.precision}",
        f"model.tensor_model_parallel_size={parallel.tensor_model_parallel_size}",
        f"model.pipeline_model_parallel_size={parallel.pipeline_model_parallel_size}",
        f"model.context_parallel_size={parallel.context_parallel_size}",
        f"model.micro_batch_size={parallel.micro_batch_size}",
        f"model.global_batch_size={parallel.global_batch_size}",
        f"model.sequence_length={parallel.sequence_length}",
        f"model.virtual_pipeline_model_parallel_size={parallel.virtual_pipeline_model_parallel_size}",
    ]


def checkpoint_compatibility_report(
    saved_parallel: ParallelRecipe,
    target_parallel: ParallelRecipe,
) -> Dict[str, object]:
    """Describe whether a checkpoint can resume directly or needs resharding."""

    compared_fields = [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "virtual_pipeline_model_parallel_size",
    ]
    mismatches = [
        field
        for field in compared_fields
        if getattr(saved_parallel, field) != getattr(target_parallel, field)
    ]

    optimizer_state_compatible = saved_parallel.data_parallel_size == target_parallel.data_parallel_size
    reshard_required = bool(mismatches)

    reasons = []
    if mismatches:
        reasons.append("model-state shard layout changed: " + ", ".join(mismatches))
    if not optimizer_state_compatible:
        reasons.append("data_parallel_size changed; optimizer state may need repartitioning")
    if saved_parallel.sequence_length != target_parallel.sequence_length:
        reasons.append("sequence_length changed; validate RoPE/cache metadata before resume")

    return {
        "direct_model_resume": not reshard_required,
        "optimizer_state_compatible": optimizer_state_compatible,
        "reshard_required": reshard_required,
        "reasons": reasons or ["parallel layout is compatible"],
    }


def _fmt_billions(value: float) -> str:
    return f"{value / 1e9:.2f}B"


def demo() -> None:
    recipe = ModelRecipe(
        model_name="gqa-7b-practice",
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=128000,
        precision="bf16",
    )
    parallel = ParallelRecipe(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        data_parallel_size=8,
        micro_batch_size=2,
        global_batch_size=256,
        sequence_length=8192,
        virtual_pipeline_model_parallel_size=2,
    )

    print("=" * 72)
    print("Model Accounting")
    print("=" * 72)
    params = estimate_dense_parameters(recipe)
    for key in ("attention", "mlp", "embeddings", "lm_head", "total"):
        print(f"{key:<12} {_fmt_billions(params[key])}")

    print("\n" + "=" * 72)
    print("Parallel Recipe Validation")
    print("=" * 72)
    validation = validate_parallel_recipe(recipe, parallel)
    print(f"world size: {validation['world_size']}")
    print(f"grad accumulation: {validation['gradient_accumulation_steps']}")
    print(f"valid: {validation['valid']}")
    print(f"warnings: {validation['warnings']}")

    print("\n" + "=" * 72)
    print("Lifecycle Plan")
    print("=" * 72)
    for stage in build_lifecycle_plan(recipe, parallel):
        print(f"{stage.name:<26} owner={stage.owner:<28} output={stage.outputs[0]}")

    print("\n" + "=" * 72)
    print("Launcher")
    print("=" * 72)
    print(" ".join(build_nemo_launcher(recipe, parallel, "conf/pretrain_gqa.yaml")))

    print("\n" + "=" * 72)
    print("Checkpoint Compatibility")
    print("=" * 72)
    target = ParallelRecipe(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=2,
        data_parallel_size=8,
        micro_batch_size=2,
        global_batch_size=256,
        sequence_length=8192,
        virtual_pipeline_model_parallel_size=1,
    )
    print(checkpoint_compatibility_report(parallel, target))


if __name__ == "__main__":
    demo()
