# 031. Lilian Weng LLM Patterns

Focus: Transformer variants, prompt strategies, agent architecture, test-time compute, reward hacking checks

Source: [Lil'Log](https://lilianweng.github.io/)
Related posts: [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/), [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/), [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)

## Practice Loop

1. Read this README and skim the TODO list below.
2. Implement `starter.py` from top to bottom without opening `solution.py`.
3. Run the starter and inspect the returned dictionaries, not just the printed text.
4. Open `solution.py` only after you have a working pass, then compare the taxonomy and edge-case choices.
5. Write one short note in `../../PROGRESS.md`: which LLM system pattern you would now recognize in another codebase.

## Run

```bash
python3 implementation-practice-codex/lessons/031-lilian-weng/starter.py
python3 implementation-practice-codex/lessons/031-lilian-weng/solution.py
```

## Reading Map

- Transformer family: attention, positional encoding, long-context memory, sparse/local/global patterns, and adaptive computation.
- Prompt engineering: zero-shot, few-shot, instruction prompts, self-consistency, CoT-style decomposition, retrieval, code, and external APIs.
- Agents: planning, memory, and tool use are separate design surfaces and need separate failure checks.
- Test-time compute: extra samples, revisions, tool calls, or search trade cost and latency for higher quality.
- Reward hacking: optimize against intended behavior, not only a proxy metric that an agent can exploit.

## TODO Surface

- function `attention_complexity`
- function `select_prompt_strategy`
- function `build_agent_spec`
- function `thinking_compute_budget`
- function `reward_hacking_red_flags`
- function `demo`

## Checkpoint Questions

- What changes in the attention graph when full attention becomes local, sparse, or global-local?
- When is a prompt strategy a task-interface choice versus a test-time compute choice?
- Which agent state belongs in short-term context, and which belongs in external memory?
- What proxy signal could be exploited if the model is optimized too directly against it?
