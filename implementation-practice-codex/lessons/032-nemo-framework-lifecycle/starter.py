"""NeMo Framework Lifecycle
==========================
Practice a NeMo-like end-to-end training lifecycle plan around Megatron Core.

The goal is not to reimplement NeMo. The goal is to make the API boundaries
explicit: model recipe, parallel recipe, lifecycle stages, launch arguments, and
checkpoint compatibility.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
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


def estimate_dense_parameters(recipe: ModelRecipe) -> Dict[str, float]:
    """Estimate dense decoder-only parameter counts by component."""
    raise NotImplementedError("TODO: implement estimate_dense_parameters; compare with solution.py only after trying.")


def derive_gradient_accumulation_steps(parallel: ParallelRecipe) -> int:
    """Return global_batch / (micro_batch * data_parallel)."""
    raise NotImplementedError("TODO: implement derive_gradient_accumulation_steps; compare with solution.py only after trying.")


def validate_parallel_recipe(recipe: ModelRecipe, parallel: ParallelRecipe) -> Dict[str, object]:
    """Validate divisibility and launch invariants before a distributed run."""
    raise NotImplementedError("TODO: implement validate_parallel_recipe; compare with solution.py only after trying.")


def build_lifecycle_plan(recipe: ModelRecipe, parallel: ParallelRecipe) -> List[LifecycleStage]:
    """Return an end-to-end training and deployment lifecycle plan."""
    raise NotImplementedError("TODO: implement build_lifecycle_plan; compare with solution.py only after trying.")


def build_nemo_launcher(recipe: ModelRecipe, parallel: ParallelRecipe, config_path: str) -> List[str]:
    """Build a dry-run torchrun command for a NeMo/Megatron-style job."""
    raise NotImplementedError("TODO: implement build_nemo_launcher; compare with solution.py only after trying.")


def checkpoint_compatibility_report(
    saved_parallel: ParallelRecipe,
    target_parallel: ParallelRecipe,
) -> Dict[str, object]:
    """Describe whether a checkpoint can resume directly or needs resharding."""
    raise NotImplementedError("TODO: implement checkpoint_compatibility_report; compare with solution.py only after trying.")


def demo() -> None:
    raise NotImplementedError("TODO: implement demo; compare with solution.py only after trying.")


if __name__ == "__main__":
    demo()
