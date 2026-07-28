# KL Divergence in Practice

> Draft — seeded from reading notes, to expand.

## Estimating KL (John Schulman's estimators)

Source: [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

- We want to estimate $`D_{KL}(q \| p) = \mathbb{E}_{x \sim q}[\log \frac{q(x)}{p(x)}]`$ from samples. Let $`r = \frac{p(x)}{q(x)}`$.
- k1: $`-\log r`$ — unbiased, but high variance (can be negative, while KL is always non-negative)
- k3: $`r - 1 - \log r`$ — guaranteed to be positive (lower variance) and unbiased
  - This is the estimator used in GRPO's KL term — see [GRPO](grpo.md)
- f-divergence view: we can use the f-divergence machinery to build estimators for $`D_{KL}(p \| q)`$ with the expectation still taken over $`q`$ (i.e. estimate the *reverse* direction from samples of the same distribution)

## Order of KL divergence — where each direction shows up

### Distillation

- $`D_{KL}(p_{teacher} \| p_{student})`$ (forward KL, mode-covering)
  - Means the student needs to assign probability everywhere the teacher does (exaggerates rare modes)
  - Failure mode: hallucinations + low-quality outputs
- $`D_{KL}(p_{student} \| p_{teacher})`$ (reverse KL, mode-seeking)
  - Means the student focuses on the strongest modes
  - Cleaner, but more deterministic generations
- Task dependence:
  - Machine translation — few correct answers (mode-seeking is fine)
  - Dialogue — many valid responses exist (want more variability → mode-covering has appeal)

### RLHF

- $`D_{KL}(\pi_{new} \| \pi_{ref})`$
  - Technically we sample from the old policy (but it's close to the new policy)
  - Idea is to "sharpen" around reference modes

### SFT / pre-training

- We can see SFT (and pre-training) as minimizing $`D_{KL}(p_{data} \| p_{model})`$
  - Mode covering
  - Then sharpen/concentrate in RL

### TRPO vs RLHF — why the directions differ

- TRPO does $`D_{KL}(\pi_{old} \| \pi_{new})`$ because samples come from the old policy
- RLHF/PPO uses $`D_{KL}(\pi_{new} \| \pi_{ref})`$ — this is a genuinely different object: the reference model is a fixed anchor, and samples come from the old policy $`\approx`$ new policy
  - Clipping in PPO somewhat constrains the log difference between old and new policy, and samples are taken from the old policy — so from a KL perspective, we're asking the new policy to cover modes of the old policy
