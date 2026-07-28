# Reasoning, Verification & Test-Time Compute

> Draft — seeded from reading notes, to expand. Related: [LLM Evals](../evals/llm_evals.md) (verifiers as judges).

## Inference-time scaling — 3 methods

Source: [Raschka on inference-time scaling](https://magazine.sebastianraschka.com/)

- Explain step by step (chain-of-thought prompting)
- Majority voting (self-consistency)
- Verification and sequential revision
  - In the revision setup, a logprob scorer can be used just to determine whether the revised answer is better

*[figure from source — re-add]*

## DeepSeekMath V2: Self-Verification and Self-Refinement

Source: [Raschka's DeepSeek technical deep-dive](https://magazine.sebastianraschka.com/p/technical-deepseek)

- Train a verifier with RL and labelled data
- Train a **meta-verifier** with RL and labelled data
  - Why do we need a meta-verifier? The verifier can receive full reward by predicting the correct scores while hallucinating non-existent issues, undermining its trustworthiness
- Retrain the verifier with meta-verifier scores
- Self-Verification and Self-Refinement
  - Ask the model to self-verify, and train this bit with the meta-verifier
- Training the verifier
  - Use majority voting to determine what is correct
  - When the verifier (as deemed by the meta-verifier) is unsure, send to humans to label

## Related reward-design notes (rule-based)

- DeepSeek: rule-based outcome reward, length penalty, and language-consistency reward (see [Post-Training / Reward Modeling](../post_training/notes.md))
