# RL

| File | Contents |
|---|---|
| [notes.md](notes.md) | **Main document.** Setup (MDP vs bandit view, rewards, on/off-policy), VPG derivation + EGLP, REINFORCE, RLOO, TRPO (trust region, natural gradient), PPO, GAE, GRPO; token- vs sequence-level importance sampling (GSPO), async RL & staleness, DAPO, TIS / CISPO / masking, Dr. GRPO |
| [kl_divergence.md](kl_divergence.md) | Schulman's k1/k2/k3 estimators; KL in reward vs KL as loss from the gradient's point of view (why k3-as-loss is a biased first-order surrogate); implementation hygiene; forward vs reverse KL in distillation / RLHF / SFT / TRPO |
| [rlhf_ppo.md](rlhf_ppo.md) | The N implementation details of RLHF with PPO; rollout/minibatch/microbatch |
| [reasoning.md](reasoning.md) | Inference-time scaling, verifiers & meta-verifiers, self-verification |
| `code.ipynb` | Q-learning / DQN / basic PPO code |

Classic RL foundations (MDPs, value functions, MC vs TD, DQN) live in [fundamentals/dl/18_rl](../../fundamentals/dl/18_rl/notes.md).

Write-up backlog: see [TODO.md](../TODO.md#rl).
