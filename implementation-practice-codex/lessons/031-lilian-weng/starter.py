"""Lilian Weng LLM Patterns
==========================
Practice utilities distilled from Lil'Log posts on Transformers, prompts, agents,
test-time compute, and reward hacking.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass.
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


def attention_complexity(seq_len, hidden_dim, pattern, window=None, global_tokens=0):
    """Estimate attention edges, FLOPs, and score-memory for common attention patterns."""
    raise NotImplementedError("TODO: implement attention_complexity; compare with solution.py only after trying.")


def select_prompt_strategy(context):
    """Choose a prompt/system strategy from task shape and risk."""
    raise NotImplementedError("TODO: implement select_prompt_strategy; compare with solution.py only after trying.")


def build_agent_spec(goal, tools, memory_items):
    """Return a minimal planning/memory/tool-use spec for an LLM agent."""
    raise NotImplementedError("TODO: implement build_agent_spec; compare with solution.py only after trying.")


def thinking_compute_budget(
    answer_tokens,
    model_parameters,
    reasoning_samples=1,
    revision_rounds=0,
    tool_calls=0,
    average_tool_latency_s=0.0,
):
    """Estimate added test-time compute and latency from sampling, revision, and tools."""
    raise NotImplementedError("TODO: implement thinking_compute_budget; compare with solution.py only after trying.")


def reward_hacking_red_flags(reward_spec):
    """Inspect a reward/evaluation spec for obvious hackable proxies."""
    raise NotImplementedError("TODO: implement reward_hacking_red_flags; compare with solution.py only after trying.")


def demo():
    raise NotImplementedError("TODO: implement demo; compare with solution.py only after trying.")


if __name__ == "__main__":
    demo()
