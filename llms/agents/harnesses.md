# Agent Harnesses

> Stub — to be filled in. (Replaced the older 2024-era agents note, which was deleted as dated.)

- ToDo: Add notes from [Harness Engineering for Self-Improvement (Lilian Weng, 2026-07)](https://lilianweng.github.io/posts/2026-07-04-harness/) — harness design patterns (workflows, file systems as memory, sub-agents), context engineering, self-improving mechanisms / evolutionary search, joint harness+weights optimization, open challenges (weak evaluators, reward hacking)

The harness is everything wrapped around the model that turns it into an agent: the loop, the tools, and the resource management. Topics to cover:

## The agent loop

- Model ↔ tool-execution loop; when to stop; streaming and interruption
- Planning vs acting; how much structure the harness imposes vs leaves to the model

## Tools

- Tool schemas/definitions; tool-call parsing and retries
- MCP (Model Context Protocol) and tool ecosystems

## Context management

- Context-window budgeting, compaction/summarization, memory (short-term vs persistent)
- Sub-agents / orchestration: fan-out, isolation, result aggregation

## Safety & execution environment

- Sandboxing, permission models, human-in-the-loop approval
- Verifiability of actions; rollbacks

## Examples to study

- Claude Code / Claude Agent SDK, OpenAI Agents SDK, open-source harnesses

Related: [Computer Use](../computer_use/notes.md) for harnesses with GUI action spaces.
