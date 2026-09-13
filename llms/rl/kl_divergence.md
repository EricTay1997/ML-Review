# KL Divergence in Practice

How KL shows up in LLM training: as a *quantity to estimate* from samples, as a *regularizer* toward a reference policy, and as a *trust region* between consecutive policies. Sources: [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) (Schulman) for the estimators, and [Rethinking KL Regularization in RLHF](https://arxiv.org/pdf/2510.01555) (Liu et al., 2025) for the gradient view. KL *definitions* live in [probability & info theory](../../fundamentals/classical/02_probability_and_info_theory/notes.md); the algorithms that use these terms are in [RL notes](notes.md).

## Order of KL divergence: where each direction shows up

### Distillation

- Off-policy (SFT): $`D_{KL}(p_{teacher} \,\|\, p_{student})`$ (forward KL, mode-covering)
  - Means the student needs to assign probability everywhere the teacher does (exaggerates rare modes)
  - Failure mode: hallucinations + low-quality outputs
- On-policy (RL, OPD, SDPO) $`D_{KL}(p_{student} \,\|\, p_{teacher})`$ (reverse KL, mode-seeking)
  - Means the student focuses on the strongest modes
  - Cleaner, but more deterministic generations
- One can think of the current training pipeline of SFT -> RL as learning various modes, followed by sharpening.

## Estimating KL (Schulman's estimators)

- We want $`D_{KL}(q \,\|\, p) = \mathbb{E}_{x \sim q}\!\left[\log \frac{q(x)}{p(x)}\right]`$ from samples $`x \sim q`$ only. Let $`\delta = \frac{p(x)}{q(x)}`$ (in RL: $`q = \pi_\theta`$, $`p = \pi_{\mathrm{ref}}`$, so $`\delta = \pi_{\mathrm{ref}}/\pi_\theta`$).
  - $`k_1 = -\log \delta`$: unbiased, high variance, can be negative even though KL is non-negative
  - $`k_2 = \frac12 (\log \delta)^2`$: biased, low variance, always positive. Its expectation is not the KL, but it is an $`f`$-divergence whose value approximates KL when $`p \approx q`$
  - $`k_3 = \delta - 1 - \log \delta`$: unbiased *as a value estimator* (since $`\mathbb{E}_q[\delta - 1] = 0`$ when $`\mathrm{supp}(p) \subseteq \mathrm{supp}(q)`$), always non-negative, and lower variance than $`k_1`$ when the distributions are close. The $`\delta - 1`$ term is a control variate: zero mean, negatively correlated with $`-\log \delta`$
  - The general recipe: any $`f`$-divergence $`\mathbb{E}_q[f(\delta)]`$ with $`f`$ convex and $`f(1) = 0`$ can be estimated by $`f(\delta) - f'(1)(\delta - 1)`$, which is non-negative by convexity. $`k_3`$ is this for $`f = -\log`$. The same machinery gives estimators for the *reverse* direction $`D_{KL}(p \,\|\, q)`$ from samples of $`q`$
- **Caveat on $`k_3`$.** "Unbiased and lower variance" holds when $`p`$ and $`q`$ are close and share support. When tails or supports differ substantially, $`\delta`$ is heavy-tailed and $`k_3`$ can have severe bias and infinite variance. Also, being a good *value* estimator says nothing about being a good *loss*, which what we discuss in the next section.
- **Where each is used.** PPO / REINFORCE put $`k_1`$ in the reward. GRPO puts $`k_3`$ in the loss. See [RL notes](notes.md#grpo) for the objectives.

## KL as a regularizer: reward vs loss, and why the gradient is what matters

Source: [Liu et al., 2025](https://arxiv.org/pdf/2510.01555). The paper's thesis: implementations pick a $`k_n`$ for its *value-estimation* properties (unbiased, low variance) and then differentiate through it, but what regularizes the policy is the *gradient*, and the two are unrelated.

- **The object.** The RLHF objective is $`\mathcal{J}(\theta) = \mathbb{E}_{x, y \sim \pi_\theta}[r(x, y)] - \beta\, D_{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big)`$: a *reverse* KL. The paper defines KL on the **full response**: $`\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})`$, i.e. the analysis below is in the bandit view (whole response = one action).
- **The true gradient.** Differentiate $`\mathcal{J}_{KL}(\theta) = \sum_y \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}`$ with the product rule, then rewrite as an expectation over the (detached) sampling distribution: <div align="center">
  $`\displaystyle \begin{aligned} \nabla_\theta \mathcal{J}_{KL} &= \sum_y \nabla_\theta \pi_\theta(y \mid x)\Big(\log \frac{\pi_\theta}{\pi_{\mathrm{ref}}} + 1\Big) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\Big(\log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + 1\Big)\nabla_\theta \log \pi_\theta(y \mid x)\right] \\ &= \mathbb{E}_{y \sim \pi_\theta}\!\left[\underbrace{\log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}}_{= -\log\delta = k_1,\ \text{detached}} \nabla_\theta \log \pi_\theta(y \mid x)\right] \end{aligned}`$ </div>
  The $`+1`$ vanishes by EGLP ($`\mathbb{E}[\nabla \log \pi_\theta] = 0`$, see [RL notes](notes.md#vanilla-policy-gradient-vpg)). **So the principled KL gradient is the score function weighted by the detached coefficient $`-\log \delta`$.** Every implementation can be judged by which coefficient it puts in front of $`\nabla_\theta \log \pi_\theta`$.
- **Two implementation styles**, evaluated by their induced coefficient:
  - *"$`k_n`$ in reward"*: $`k_n`$ is a detached scalar multiplying the score, $`\mathcal{J} = \mathbb{E}[\mathrm{sg}[k_n] \cdot \log \pi_\theta]`$. The coefficient is $`k_n`$ itself.
  - *"$`k_n`$ as loss"*: $`\mathcal{J} = \mathbb{E}_{y \sim \pi_\theta}[k_n]`$ with gradients flowing through $`k_n`$ but the sampling distribution detached. By the chain rule the coefficient is $`k_n' = \frac{\partial k_n}{\partial \log \pi_\theta}`$, evaluated at the current snapshot.
- **Scorecard**, with $`\delta = \pi_{\mathrm{ref}} / \pi_\theta`$:

  | Implementation | Coefficient on $`\nabla_\theta \log \pi_\theta`$ | Verdict |
  |---|---|---|
  | $`k_1`$ in reward (PPO, REINFORCE) | $`-\log \delta`$ | principled |
  | $`k_2`$ as loss | $`\frac{\partial}{\partial \log \pi_\theta}\frac12(\log \pi_\theta - \log \pi_{\mathrm{ref}})^2 = -\log \delta`$ | principled, identical to the row above |
  | $`k_1`$ as loss | $`\frac{\partial}{\partial \log \pi_\theta}(\log \pi_\theta - \log \pi_{\mathrm{ref}}) = 1`$ | zero-mean noise, **no regularization** |
  | $`k_3`$ as loss (GRPO) | $`\frac{\partial}{\partial \log \pi_\theta}(\delta - 1 - \log \delta) = -\delta + 1 = 1 - \delta`$ | first-order approximation of $`-\log \delta`$, biased |

  (For the $`k_3`$ row: $`\partial \delta / \partial \log \pi_\theta = -\delta`$ and $`\partial(-\log \delta)/\partial \log \pi_\theta = +1`$.)
- **$`k_1`$ in reward $`\Leftrightarrow`$ $`k_2`$ as loss.** Same coefficient, same expected gradient, both equal to the true reverse-KL gradient. This is the paper's headline equivalence and the justification for PPO's convention. The equivalence needs **on-policy samples**; with a stale rollout policy $`\pi_{\theta_k}`$ the "as loss" form must be importance-weighted by $`\pi_\theta / \pi_{\theta_k}`$ (and, in PPO, clipped) like any other head. "In reward" gets this for free because the KL is folded into the advantage before the PPO ratio and clip are applied. Most GRPO implementations omit the correction on the KL term.
- **$`k_1`$ as loss is a trap.** Its gradient $`\mathbb{E}[\nabla_\theta \log \pi_\theta]`$ has expectation exactly zero and is independent of $`\pi_{\mathrm{ref}}`$. It's the *inverse* of a REINFORCE baseline: the same zero-mean score term, added instead of subtracted, injecting variance. Empirically indistinguishable from no KL. **This is the counterexample to "unbiased value estimator implies good loss."**
- **$`k_3`$ as loss: the Taylor trap.** Around $`\delta = 1`$, $`-\log \delta = (1 - \delta) + \frac12(\delta - 1)^2 - \frac13(\delta - 1)^3 + \dots`$, so the $`k_3`$ coefficient $`1 - \delta`$ is only the first-order term of the principled $`-\log \delta`$. Three consequences:
  - *Biased*: for every $`\delta \ne 1`$, $`1 - \delta < -\log \delta`$ (strict lower bound from $`\log \delta \le \delta - 1`$), so the update direction differs from the true KL gradient.
  - *Pathologically asymmetric in the tails*:
    - Over-coverage, $`\delta \to 0`$ ($`\pi_\theta \gg \pi_{\mathrm{ref}}`$, the policy has sharpened onto something the reference found unlikely): $`-\log \delta \to +\infty`$, a strong sustained restoring force, whereas $`1 - \delta \to 1`$ saturates. A much weaker regularizer exactly where late-stage RLHF spends its time.
    - Under-coverage, $`\delta \to \infty`$ ($`\pi_\theta \ll \pi_{\mathrm{ref}}`$): $`-\log \delta`$ grows only logarithmically, but $`1 - \delta \to -\infty`$ linearly, inducing explosive updates on tokens the policy has suppressed.
  - *Statistically unstable*: under on-policy sampling $`\mathbb{E}[1 - \delta] = 0`$ and $`\mathrm{Var}[1 - \delta] = \mathbb{E}[(\delta - 1)^2] = \chi^2(\pi_{\mathrm{ref}} \,\|\, \pi_\theta)`$, the chi-squared divergence, which is notoriously heavy-tailed and infinite if the support condition fails. The stochastic gradient inherits this variance.
  - Empirically both $`k_2`$ and $`k_3`$ as loss do regularize, but $`k_2`$ keeps a tighter coupling to the reference with lower reward variance. $`k_3`$'s weaker constraint at high policy probability can be a *feature* if you want a mild late-stage leash, but it's not the KL you wrote down.
- **A bounded alternative.** MiniMax-01's regularizer is the full-vocabulary MSE $`\frac12 \sum_y (\pi_\theta - \pi_{\mathrm{ref}})^2`$, whose on-policy gradient has coefficient $`\pi_\theta(y \mid x) - \pi_{\mathrm{ref}}(y \mid x) \in [-1, 1]`$: bounded, symmetric in probability space, and fully compatible with IS/clipping since it's an "in reward" head.
- **Recommendations** (paper's, which I'd adopt): never $`k_1`$ as loss. Prefer $`k_1`$ in reward or $`k_2`$ as loss. Treat $`k_3`$ as loss as a biased approximation, not the default. Off-policy, apply IS and clipping to the KL head, either *combined* (fold $`-\beta k_n'`$ into the advantage before the PPO clip, then baseline/normalize the combined signal, not the KL alone) or *decoupled* (two clipped heads, reward and KL, possibly with asymmetric clipping on the reward head only).

## Implementation notes

From the same paper's appendix:

- The KL term needs to inherit importance sampling ratios. This is usually inherited with the reward form, but often missed in the loss form.
  - Concretely: DeepSeekMath's GRPO objective (eq. 3) carries the ratio on the advantage only; the $`k_3`$ term is differentiated on samples from $`\pi_{\mathrm{old}}`$ with no correction. Liu et al.'s Table 1 lists GRPO as "requires explicit IS/clipping, commonly omitted in practice." TRL's GRPOTrainer v0.14 through v0.19 reproduce exactly this.
  - [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (eq. 7) multiplies the $`k_3`$ estimator by $`\rho = \pi_\theta/\pi_{\mathrm{old}}`$ and states, without proof, that "the gradient of this KL estimator becomes unbiased"; it also notes the original "assigns disproportionately large, unbounded weights" when $`\pi_\theta \ll \pi_{\mathrm{ref}}`$, which is the under-coverage tail above. 
    - **Proof.** Let $`s = \nabla_\theta \log\pi_\theta(y)`$, with $`\pi_{\mathrm{old}}`$ and $`\pi_{\mathrm{ref}}`$ constant. 
    - By the log-derivative trick $`\nabla\rho = \rho\,s`$ and $`\nabla\delta = -\delta\,s`$, so $`\nabla k_3 = \nabla\delta - \nabla\log\delta = (1-\delta)\,s`$ (the biased "$`k_3`$ as loss" coefficient). 
    -Product rule: <div align="center">
      $`\displaystyle \nabla_\theta(\rho\,k_3) = k_3\,\nabla\rho + \rho\,\nabla k_3 = \rho\,\big[(\delta - 1 - \log\delta) + (1-\delta)\big]\,s = \rho\,(-\log\delta)\,s = \rho\,k_1\,\nabla_\theta \log\pi_\theta(y)`$ </div>
      which is the importance-weighted "$`k_1`$ in reward" gradient (Liu et al. eq. 69 with coefficient $`k_1`$).
- **Sequence-level ratio and clipping.** Even in token-level PPO with a sequence-level KL, compute the ratio as $`\rho = \exp\big(\sum_t \log \pi_\theta(y_t \mid \cdot) - \sum_t \log \pi_{\theta_k}(y_t \mid \cdot)\big)`$ and clip at the *sequence* level; per-token clipping is overly conservative for the KL head. (Compare the token-vs-sequence IS discussion in [RL notes](notes.md#sequence-level-vs-token-level-importance-sampling).)
- **Masking.** Sum log-probs only over the action tokens that contribute to the reward and KL: exclude prompt, padding and any masked tokens, so the ratio and the heads are aligned.
- **Numerical stability.** Ensure $`\pi_{\theta_k}(y \mid x) > 0`$ for every sample (guard the log), and consider capping the KL log-difference or the ratio to avoid overflow under extreme ratios. Clip the group std in GRPO-style normalization so near-constant reward groups don't produce huge advantages (or drop the std entirely, see [Dr. GRPO](notes.md#dr-grpo)).
- **Adaptive $`\beta`$.** If targeting a KL budget, update $`\beta`$ outside the gradient path and don't mix it into advantage normalization.