# RLHF with PPO — Implementation Details

> Draft — seeded from reading notes, to expand. Primary source: [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo). For the conceptual RLHF pipeline see [Post-Training](../post_training/notes.md); for PPO itself see [Policy Gradients](policy_gradients.md).

## Reward model training

- 1 reward per token, generated autoregressively, then just extract the reward of the last token
- Reward head initialized with bias = 0 and variance = $`d_{model} + 1`$ *(verify against source — likely init scale $`1/\sqrt{d_{model}+1}`$)*
- To keep the scale of the reward model consistent across training, normalize it to mean 0 and variance 1
  - Normalization is applied before and after reward model training

## Policy training

- A rule granting additional reward when a period appears at a position between 16 and 24 *(verify against source — task-specific response-length shaping)*
- Discount factor $`\gamma = 1`$
- Shuffles batch indices at each epoch
- Per-token KL penalty
- Whiten rewards without shifting the mean, then calculate advantage with GAE ($`\lambda = 0.95`$, so 0.05 weight on TD), then whiten advantage
  - Note that in the "reward-to-go" framing, the last-token reward is copied to all past tokens, and the per-token KL penalty is propagated backward in a "triangular" fashion
  - This triangular shape is exactly what changes when KL is applied at the sequence level instead — see [GRPO](grpo.md)
- Adaptive KL: the penalty coefficient is adjusted to bring the measured KL within a target range. Hyperparameters include a clip scaled by batch_size / horizon

## Batch terminology (rollout / minibatch / microbatch)

- (Rollout) batch size: you have a rollout pool of say 10k examples
- Minibatch:
  - If you calculate gradients and do an SGD update using 256 samples, this is a minibatch of 256
  - PPO often does multiple epochs over the SAME rollout pool
- Microbatch:
  - 1 GPU pass, added together with gradient accumulation
  - Used when the minibatch doesn't fit in memory
  - Or for efficiency — think pipeline parallelism (see [Parallelism](../performance/parallelism.md))
