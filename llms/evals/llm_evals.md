# LLM Evaluation

> Draft — seeded from reading notes, to expand. Primary source: [Raschka — LLM evaluation: 4 approaches](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches). Governance-flavored evals (RSPs, science of evals, statistics of evals) are in [governance_and_statistics.md](governance_and_statistics.md).

## The four approaches

- Multiple-choice benchmarks (e.g. MMLU) — cheap, contamination-prone, measures knowledge not generation
- Verifiers — check the output programmatically (math answers, unit tests); the foundation of RLVR-style training (see [RL / Reasoning](../rl/reasoning.md))
- Leaderboards / arenas — pairwise human preference (e.g. LMArena-style Elo)
- LLM judges — model-graded eval; cheap and scalable, but judge bias and calibration matter

## To write up

- Judge rubric design, pass@k for verifiable tasks, contamination checks, agentic/task-based evals
