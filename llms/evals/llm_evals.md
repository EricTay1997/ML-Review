# LLM Evaluation

> Draft — seeded from reading notes, to expand. Primary source: [Raschka — LLM evaluation: 4 approaches](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches). Governance-flavored evals (RSPs, science of evals, statistics of evals) are in [governance_and_statistics.md](governance_and_statistics.md).

## The four approaches

- Multiple-choice benchmarks (e.g. MMLU) — cheap, contamination-prone, measures knowledge not generation
- Verifiers — check the output programmatically (math answers, unit tests); the foundation of RLVR-style training (see [RL / Reasoning](../rl/reasoning.md)); for hard-to-verify domains, verifiers can themselves be trained models — see DeepSeekMath-V2 below
- Leaderboards / arenas — pairwise human preference (e.g. LMArena-style Elo)
- LLM judges — model-graded eval; cheap and scalable, but judge bias and calibration matter

## DeepSeekMath V2: Self-Verification and Self-Refinement

Sources: [Raschka's DeepSeek technical deep-dive](https://magazine.sebastianraschka.com/p/technical-deepseek), [DeepSeekMath-V2 paper (arXiv 2511.22570)](https://arxiv.org/abs/2511.22570) — see also the reward-hacking angle in [Post-Training §Reward Hacking](../post_training/notes.md)

- Train a verifier with RL and labelled data
- Train a **meta-verifier** with RL and labelled data
  - Why do we need a meta-verifier? The verifier can receive full reward by predicting the correct scores while hallucinating non-existent issues, undermining its trustworthiness
- Retrain the verifier with meta-verifier scores
- Self-Verification and Self-Refinement
  - Ask the model to self-verify, and train this bit with the meta-verifier
- Training the verifier
  - Use majority voting to determine what is correct
  - When the verifier (as deemed by the meta-verifier) is unsure, send to humans to label

## To write up

- Judge rubric design, pass@k for verifiable tasks, contamination checks, agentic/task-based evals
