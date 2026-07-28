# GRPO

> Draft — objective relocated from the old post-training notes; variants seeded from reading notes, to expand. See [RLHF with PPO](rlhf_ppo.md) for the PPO-based pipeline this simplifies.

## Objective

- ```math
  \frac{1}{G} \sum_{i=1}^G\left(\min \left(\frac{\pi_\theta\left(o_i \mid q\right)}{\pi_{\theta_{o l d}}\left(o_i \mid q\right)} A_i, \mathrm{clip}\left(\frac{\pi_\theta\left(o_i \mid q\right)}{\pi_{\theta_{o l d}}\left(o_i \mid q\right)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right)-\beta \mathbb{D}_{K L}\left(\pi_\theta \| \pi_{r e f}\right)\right)
  ```
  - $`\mathbb{D}_{K L}\left(\pi_\theta \| \pi_{r e f}\right)=\frac{\pi_{r e f}\left(o_i \mid q\right)}{\pi_\theta\left(o_i \mid q\right)}-\log \frac{\pi_{r e f}\left(o_i \mid q\right)}{\pi_\theta\left(o_i \mid q\right)}-1`$
    - This is Schulman's k3 estimator — positive and unbiased; see [KL Divergence](kl_divergence.md)
  - Questions $`q \sim P(Q)`$, Outputs $`\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{o l d}}(O \mid q)`$

## From RLHF PPO to GRPO

- GRPO removes the need for a value network — the group-relative advantage (normalize rewards within the group of $`G`$ samples) replaces the learned baseline
- Similar to having the reward at the last token with discount factor $`\gamma = 1`$, the entire sequence is awarded the reward
  - Another way to think about it: the *sequence* gets the reward, and the probability of the sequence is just the sum of token log-probs
- KL can be applied as a **reward** or as a **loss**. As a reward, what's different is that we'd apply the full-sequence KL divergence at the last token too — as opposed to the triangular shape induced by the PPO RLHF implementation (per-token KL propagated backward through reward-to-go; see [RLHF with PPO](rlhf_ppo.md))
- Both are technically valid from a gradient perspective. The way to think about it: should the state (token) actually incur the negative reward (log-prob diff), or is it a regularizer? (I find the latter framing more natural.)

## Variants

- DAPO
  - Asymmetric clipping
  - Dynamic sampling
  - Token-level loss
  - Length-based reward shaping
  - No KL loss
- Dr. GRPO
  - Remove length and std normalizations
- OLMo 3
  - Truncated Importance Sampling
- DeepSeek V3.2
  - Domain-specific KL strengths
  - KL reweighing
  - Off-policy sequence masking
  - Keep routing for MoE models
  - Keep sampling mask for top-p / top-k
