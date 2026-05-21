"""Lilian Weng LLM Patterns
==========================
Practice utilities distilled from Lil'Log posts on Transformers, prompts, agents,
test-time compute, and reward hacking.

This file keeps the ideas executable: count an attention graph, choose a prompt
strategy, sketch an agent spec, budget test-time compute, and flag reward proxies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    task: str
    examples_available: int
    needs_reasoning: bool
    has_retrieval: bool
    has_tools: bool
    risk_level: str


def _require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def attention_complexity(seq_len, hidden_dim, pattern, window=None, global_tokens=0):
    """Estimate attention edges, FLOPs, and score-memory for common attention patterns."""

    _require_positive("seq_len", seq_len)
    _require_positive("hidden_dim", hidden_dim)
    pattern = pattern.lower().replace("-", "_")

    if pattern == "full":
        edges = seq_len * seq_len
    elif pattern == "local":
        if window is None:
            raise ValueError("window is required for local attention")
        _require_positive("window", window)
        edges = seq_len * min(seq_len, 2 * window + 1)
    elif pattern in {"global_local", "local_global"}:
        if window is None:
            raise ValueError("window is required for global_local attention")
        if global_tokens < 0:
            raise ValueError("global_tokens must be non-negative")
        local_edges = seq_len * min(seq_len, 2 * window + 1)
        global_edges = 2 * seq_len * global_tokens
        global_to_global_edges = global_tokens * global_tokens
        edges = min(seq_len * seq_len, local_edges + global_edges + global_to_global_edges)
    elif pattern == "linear":
        edges = seq_len
    else:
        raise ValueError(f"unknown attention pattern: {pattern!r}")

    score_memory_bytes = edges * 4
    qk_flops = 2 * edges * hidden_dim
    av_flops = 2 * edges * hidden_dim

    return {
        "pattern": pattern,
        "edges": edges,
        "qk_flops": qk_flops,
        "av_flops": av_flops,
        "total_flops": qk_flops + av_flops,
        "score_memory_bytes": score_memory_bytes,
        "edge_ratio_vs_full": edges / (seq_len * seq_len),
    }


def select_prompt_strategy(context):
    """Choose a prompt/system strategy from task shape and risk."""

    if context.examples_available < 0:
        raise ValueError("examples_available must be non-negative")
    if context.risk_level not in {"low", "medium", "high"}:
        raise ValueError("risk_level must be one of: low, medium, high")

    strategy = []
    notes = []

    if context.examples_available >= 3:
        strategy.append("few_shot")
        notes.append("use representative examples to define input/output format")
    else:
        strategy.append("zero_shot_or_instruction")
        notes.append("state the task and output schema explicitly")

    if context.needs_reasoning:
        strategy.append("decompose_then_answer")
        notes.append("spend extra test-time compute on subgoals, samples, or verification")

    if context.has_retrieval:
        strategy.append("retrieval_augmented")
        notes.append("ground claims in retrieved context before generation")

    if context.has_tools:
        strategy.append("tool_augmented")
        notes.append("route current facts, code execution, or private data through tools")

    if context.risk_level == "high":
        strategy.append("verification_required")
        notes.append("separate generation from checking and prefer auditable evidence")
    elif context.risk_level == "medium":
        strategy.append("lightweight_self_check")

    return {
        "task": context.task,
        "strategy": strategy,
        "notes": notes,
    }


def build_agent_spec(goal, tools, memory_items):
    """Return a minimal planning/memory/tool-use spec for an LLM agent."""

    if not goal:
        raise ValueError("goal must be non-empty")

    tool_names = [tool["name"] if isinstance(tool, dict) else str(tool) for tool in tools]
    short_term = list(memory_items[-4:])
    long_term = list(memory_items[:-4])

    return {
        "goal": goal,
        "planning": [
            "state success criteria",
            "decompose the goal into subgoals",
            "choose the next action from available tools",
            "observe results and update state",
            "reflect before final output",
        ],
        "memory": {
            "short_term_context": short_term,
            "long_term_store_candidates": long_term,
        },
        "tool_use": {
            "available_tools": tool_names,
            "policy": "call a tool when model weights are stale, private data is needed, or execution is required",
        },
        "failure_checks": [
            "stale or missing memory",
            "tool result not verified",
            "plan continues after success criteria are already met",
        ],
    }


def thinking_compute_budget(
    answer_tokens,
    model_parameters,
    reasoning_samples=1,
    revision_rounds=0,
    tool_calls=0,
    average_tool_latency_s=0.0,
):
    """Estimate added test-time compute and latency from sampling, revision, and tools."""

    for name, value in (
        ("answer_tokens", answer_tokens),
        ("model_parameters", model_parameters),
        ("reasoning_samples", reasoning_samples),
    ):
        _require_positive(name, value)
    if revision_rounds < 0 or tool_calls < 0 or average_tool_latency_s < 0:
        raise ValueError("revision_rounds, tool_calls, and average_tool_latency_s must be non-negative")

    passes = reasoning_samples * (1 + revision_rounds)
    generated_tokens = answer_tokens * passes
    flops_per_token = 2 * model_parameters
    model_flops = generated_tokens * flops_per_token
    external_latency_s = tool_calls * average_tool_latency_s

    return {
        "passes": passes,
        "generated_tokens": generated_tokens,
        "flops_per_token": flops_per_token,
        "model_flops": model_flops,
        "external_latency_s": external_latency_s,
        "compute_multiplier_vs_direct": passes,
    }


def reward_hacking_red_flags(reward_spec):
    """Inspect a reward/evaluation spec for obvious hackable proxies."""

    flags = []
    if reward_spec.get("single_proxy_metric"):
        flags.append("single proxy metric may diverge from intended behavior")
    if reward_spec.get("evaluator_visible_to_agent"):
        flags.append("agent can optimize to the evaluator instead of the task")
    if reward_spec.get("editable_tests_or_labels"):
        flags.append("agent can change the measurement surface")
    if reward_spec.get("ambiguous_success_criteria"):
        flags.append("ambiguous success criteria invite shortcut behavior")
    if reward_spec.get("no_ood_checks"):
        flags.append("missing OOD checks can hide spurious correlations")
    if reward_spec.get("preference_only_without_rubric"):
        flags.append("preference labels need a rubric to reduce exploitable taste matching")

    return flags or ["no obvious red flags in this simple checklist"]


def _fmt_big(number):
    if number >= 1e12:
        return f"{number / 1e12:.2f}T"
    if number >= 1e9:
        return f"{number / 1e9:.2f}B"
    if number >= 1e6:
        return f"{number / 1e6:.2f}M"
    return f"{number:.0f}"


def demo():
    print("=" * 72)
    print("Attention Pattern Accounting")
    print("=" * 72)
    full = attention_complexity(seq_len=8192, hidden_dim=128, pattern="full")
    local = attention_complexity(seq_len=8192, hidden_dim=128, pattern="local", window=256)
    mixed = attention_complexity(
        seq_len=8192,
        hidden_dim=128,
        pattern="global_local",
        window=256,
        global_tokens=16,
    )
    for item in (full, local, mixed):
        print(
            f"{item['pattern']:<12} edges={_fmt_big(item['edges'])} "
            f"ratio={item['edge_ratio_vs_full']:.4f}"
        )

    print("\n" + "=" * 72)
    print("Prompt Strategy")
    print("=" * 72)
    strategy = select_prompt_strategy(
        PromptContext(
            task="summarize a private research thread with citations",
            examples_available=4,
            needs_reasoning=True,
            has_retrieval=True,
            has_tools=True,
            risk_level="high",
        )
    )
    print(" -> ".join(strategy["strategy"]))

    print("\n" + "=" * 72)
    print("Agent Spec")
    print("=" * 72)
    agent = build_agent_spec(
        goal="prepare a model-scaling experiment plan",
        tools=[{"name": "calendar"}, {"name": "code_runner"}, {"name": "paper_search"}],
        memory_items=[
            "previous GPU budget",
            "baseline throughput",
            "failed run note",
            "cluster quota",
            "current model config",
        ],
    )
    print(f"tools={agent['tool_use']['available_tools']}")
    print(f"short_term={agent['memory']['short_term_context']}")

    print("\n" + "=" * 72)
    print("Test-Time Compute Budget")
    print("=" * 72)
    budget = thinking_compute_budget(
        answer_tokens=512,
        model_parameters=7e9,
        reasoning_samples=8,
        revision_rounds=1,
        tool_calls=3,
        average_tool_latency_s=0.8,
    )
    print(f"generated tokens={budget['generated_tokens']}")
    print(f"model FLOPs={budget['model_flops'] / 1e12:.1f} TFLOPs")
    print(f"external latency={budget['external_latency_s']:.1f}s")

    print("\n" + "=" * 72)
    print("Reward Hacking Checklist")
    print("=" * 72)
    flags = reward_hacking_red_flags(
        {
            "single_proxy_metric": True,
            "evaluator_visible_to_agent": True,
            "editable_tests_or_labels": False,
            "ambiguous_success_criteria": True,
            "no_ood_checks": True,
        }
    )
    for flag in flags:
        print(f"- {flag}")


if __name__ == "__main__":
    demo()
