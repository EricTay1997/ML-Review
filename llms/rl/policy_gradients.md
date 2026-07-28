# Policy Gradients

> Draft — seeded from reading notes, to expand from [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/). See [RL fundamentals](../../fundamentals/dl/18_rl/notes.md) for MDPs/value functions.

## To write up (Spinning Up)

- Policy gradient theorem / deriving the simplest policy gradient
- REINFORCE
  - Reward-to-go trick; baselines for variance reduction
- Vanilla Policy Gradient (VPG)
  - Advantage estimation (GAE)
- TRPO
  - Trust region via KL constraint; natural gradient; samples come from the *old* policy, hence $`D_{KL}(\pi_{old} \| \pi_{new})`$ — see [KL Divergence](kl_divergence.md) for why the direction matters
- PPO
  - Clipped surrogate objective as a first-order approximation of the trust region
  - Clipping constrains the log-prob difference between old and new policies (relevant to the KL story — see [KL Divergence](kl_divergence.md))

## Code

- [Coding PPO from scratch with PyTorch (4-part series)](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8)
- `code.ipynb` in this folder has basic PPO code.
