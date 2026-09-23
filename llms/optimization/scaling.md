# Scaling

How to take a tuned small run to a big one. 

Primary sources: 
- [Scaling Laws, Carefully (Weng)](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)
- [How To Scale](https://howtoscalenn.github.io/)
- [Complete(d)P (Mlodozeniec et al., 2025)](https://arxiv.org/abs/2512.22382)
- [μP & μTransfer](muP.md), [Norms](norms.md) and [Muon](muon.md)
- [fundamentals/dl/04](../../fundamentals/dl/04_optimization_and_regularization/notes.md)
- [Performance §Batch size](../performance/basics.md#batch-size)

## 0. The starting point

- I have a small proxy run whose hyperparameters are tuned:
  - Width $`n`$ ($`d_{model}`$)
  - Depth $`L`$
  - Batch size $`B`$ (tokens/step)
  - Tokens $`D`$
  - Derived:
    - Steps $`S = D/B`$
    - Non-embedding params $`N \approx 12 L n^2`$
    - Compute $`C \approx 6ND`$
  - **Hyperparameters (HPs)**, which have to be re-derived: 
    - Peak LR $`\eta`$ (and schedule)
    - Weight decay $`\lambda`$
    - Adam $`\beta_1, \beta_2, \epsilon`$
    - Init std
    - Forward multipliers.
  - Ratios, target / proxy (subscript 0 = proxy)
    - $`m_N = n/n_0`$
    - $`m_L = L/L_0`$
    - $`m_B = B/B_0`$
    - $`m_D = D/D_0`$
    - $`m_S = m_D/m_B`$
- Complete(d)P does the following process: **pick a limit, and parametrize so that proxy and target are two discretizations of the same limiting process.** Same process ⇒ same optimum.
  - Width → the infinite-width limit (μP). Depth → the infinite-depth limit (Depth-μP / CompleteP, [§2](#depth)). Batch size and training length → the SDE (stochastic differential equation) limit of the optimizer ([Malladi et al., 2022](https://arxiv.org/abs/2205.10287), [§3](#3-scaling-data-at-a-fixed-model)).
  - The following rules therefore only hold while its limit is a decent approximation: the proxy can't be too small (≳100M params, [§2](#beyond-the-table)) and the batch can't be too big (≲ the critical batch size $`B_{\mathrm{crit}}`$, [§3](#critical-batch-size)).
- Tensor types in the table: **embedding** (vocab → width), **hidden matrices** (width → width: every attention and MLP projection), **readout** (width → vocab, the unembedding). $`\alpha`$ is the depth exponent ([§2](#depth)).

| | width | depth | steps ($`m_S`$) |
|---|---|---|---|
| residual branch multiplier | — | $`m_L^{-\alpha}`$, $`\alpha \in [\tfrac12, 1]`$ | — |
| init variance, hidden matrices | $`\times m_N^{-1}`$ (fan-in) | — | — |
| init variance, readout | $`\times m_N^{-2}`$ (or zero) | — | — |
| LR, hidden matrices | $`\times m_N^{-1}`$ | $`\times m_L^{\alpha-1}`$ | $`\times m_S^{-1/2}`$ |
| LR, readout | $`\times m_N^{-1}`$ | — | $`\times m_S^{-1/2}`$ |
| LR, embedding | — | — | $`\times m_S^{-1/2}`$ |
| weight decay, hidden & readout | $`\times m_N`$ | — | $`\times m_S^{-1/2}`$ |
| Adam $`\epsilon`$, hidden & embedding | $`\times m_N^{-1}`$ | $`\times m_L^{-\alpha}`$ (hidden) | $`\times m_S^{1/2}`$ |
| Adam $`\epsilon`$, readout | — | — | $`\times m_S^{1/2}`$ |
| $`1-\beta_1`$, $`1-\beta_2`$ | — | — | $`\times m_S^{-1}`$ |

- Two things to call out:
  - The last column depends on $`B`$ and $`D`$ only through $`m_S`$. **The hyperparameters care about steps, not tokens** ([§3](#3-scaling-data-at-a-fixed-model)).
  - Every row keeps $`\eta\lambda`$ (PyTorch AdamW's decay per step) fixed in width. **The per-tensor LR moves with width; the per-tensor decay rate doesn't** ([§3](#weight-decay-as-an-ema-window) says why that product is the thing to hold).
- The table, and this document, are in μP. For contrast, [§2](#width-in-sp) works one example in SP (fan-in init, one global LR): doubling the width.
- Batch size belongs to the configuration: below $`B_{\mathrm{crit}}`$ it's a throughput knob, picked for wall-clock.

## 1. Scaling compute: N vs D

_Source: [Weng](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/) unless noted._

- $`C \approx 6ND`$: 
  - $`2ND`$ for forward 
  - $`4ND`$ for backward (grads w.r.t. activations and weights)
  - Drops attention's $`n_{ctx}`$ (context length) term, fine while $`n_{ctx} \lesssim 12 d_{model}`$.
- Chinchilla's parametric fit ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)), with $`E`$ the irreducible loss and $`a, b, \alpha, \beta`$ fitted constants: <div align="center">
  $`\displaystyle \hat L(N, D) = E + \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}}`$ </div>
    - Minimizing under $`6ND = C`$ gives $`N_{\mathrm{opt}} \propto C^{\beta/(\alpha+\beta)}`$, $`D_{\mathrm{opt}} \propto C^{\alpha/(\alpha+\beta)}`$. 
    - The fit has $`\alpha \approx \beta`$, so $`N`$ and $`D`$ grow at the same rate, $`\propto C^{0.5}`$.
- Let's reconcile [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) ($`N_{\mathrm{opt}} \propto C^{0.73}`$: 10× compute → 5.5× params, 1.8× tokens) with Chinchilla ($`C^{0.5}`$). Same principle, different fits:
  - Kaplan counted non-embedding params and fit mostly small models, where embeddings are a big share of the total. Rewritten in non-embedding units, Chinchilla's law has a *local* exponent ≈ 0.73 at Kaplan's scale that tends to 0.5 as $`C`$ grows ([Pearce & Song, 2024](https://arxiv.org/abs/2406.12907)). 
  - Kaplan also used one LR schedule for every run; Chinchilla found the cosine length has to match the token budget ([Hoffmann et al.](https://arxiv.org/abs/2203.15556)) ([§3](#3-scaling-data-at-a-fixed-model)).
- Epochs: once unique data runs out, you repeat it. Setup: $`U_D`$ unique tokens seen for $`1 + R_D`$ epochs, so $`D = U_D(1 + R_D)`$.
  - [Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264): **models repeats as diminishing returns.** The $`k`$-th repeat is worth $`e^{-k/r_D}`$ of a fresh epoch ($`r_D`$ a fitted decay constant, in epochs); summing gives the effective data that replaces $`D`$ in the Chinchilla fit: <div align="center">
    $`\displaystyle D' = U_D + U_D\, r_D \left(1 - e^{-R_D/r_D}\right)`$ </div>
    - Fitted $`r_D \approx 15`$: 4 epochs ≈ fresh data ($`D' \approx 0.93D`$); past ~16 epochs a repeat is worth less than $`1/e`$ of fresh data; infinite repeats cap at $`D' \approx 16\,U_D`$.
    - $`D'`$ only increases with $`R_D`$: this model can say repeats stop helping, never that they hurt.
  - [Lovelace et al. (2026)](https://arxiv.org/abs/2605.01640): **models repeats as full data plus an overfitting penalty.** <div align="center">
    $`\displaystyle \hat L = E + \frac{a}{N^{\alpha}} + \frac{b}{\left(U_D(1+R_D)\right)^{\beta}} + P\, R_D\, \frac{N}{U_D}`$ </div>
    - The penalty grows with repeats and with parameters per unique token $`N/U_D`$ (bigger models overfit repeats faster). Their 4-parameter fit makes it superlinear in both ($`\approx R_D^{1.7} N^{1.3}`$).
    - So loss is U-shaped in epochs. **Too many repeats** is where $`\partial \hat L / \partial R_D = 0`$: <div align="center">
      $`\displaystyle \text{epochs}^* = \left(\frac{\beta\, b\, U_D^{1-\beta}}{P\, N}\right)^{1/(1+\beta)} \propto \frac{U_D^{0.40}}{N^{0.70}}`$ </div>
    - From their fit: ~5–10 epochs when $`N \approx U_D`$, more for models small relative to their data. Doubling the model cuts tolerable epochs by $`2^{0.7} \approx 1.6\times`$.
    - At fixed compute the law recommends fewer (3–6 epochs in their runs), and past a compute threshold, **spend on a bigger model and cut epochs**. 
    - Strong weight decay ($`\lambda = 1.0`$ vs 0.1) cuts $`P`$ by ~70%.

## 2. Scaling width and depth

_The full width derivation is in [μP & μTransfer](muP.md). This section is the main insights._

- Under the standard parametrization (SP: fan-in init, one global LR) the optimal LR slides left as width grows ([Tensor Programs V](https://arxiv.org/abs/2203.03466), "TP5", Fig. 1; GPT-3's own table goes $`6\times10^{-4}`$ at 125M → $`0.6\times10^{-4}`$ at 175B). Mechanism: after one step, $`\Delta W x`$ is a *correlated* sum over the fan-in, so the same per-entry step moves a width-$`n`$ layer $`n\times`$ as much ([μP §Intuition](muP.md#intuition)).
  - Recall: $`n`$ independent centred terms sum to $`O(\sqrt n)`$ (CLT — what fan-in init controls); $`n`$ aligned terms sum to $`O(n)`$ (LLN — what the LR has to control).

### Width in μP

- The rules (for stability) under Adam:
  - **LR $`\propto 1/\text{fan-in}`$ for every matrix.** Adam's per-entry step is ≈ $`\eta`$ whatever the gradient size, so a layer's output $`h = Wx`$ moves by $`\Delta h = O(\text{fan-in} \cdot \eta)`$, a correlated sum over the fan-in.
    - Hidden matrices: fan-in is $`d_{model}`$ or $`d_{ff}`$, so the LR shrinks with width.
    - Readout: fan-in is $`d_{model}`$, same rule.
    - Embedding: fan-in is the vocab (a one-hot), which doesn't grow, so the LR stays constant.
  - **Init: fan-in variance, except the readout at $`1/\text{fan-in}^2`$** (or zero-init). Once $`W_{out}`$ aligns with the features, the logits are a correlated sum over its fan-in, so its entries have to be $`\Theta(1/\text{fan-in})`$. Side effect: logits start near zero.
  - **Attention: $`q^\top k / d_{head}`$** instead of $`/\sqrt{d_{head}}`$, if $`d_{head}`$ is what grows. Same CLT → LLN story ([μP §Deriving the rules](muP.md#deriving-the-rules)).
  - **Adam $`\epsilon \propto 1/n`$ on everything but the readout** ($`n = d_{model}`$, the readout's fan-in). $`\epsilon`$ has to shrink with the gradients, or it takes over the denominator and turns Adam into SGD for wide models. Every tensor upstream of the readout gets its gradient through a $`\Theta(1/n)`$ readout, so those gradients are $`\Theta(1/n)`$; the readout's own gradient is $`\Theta(1)`$.
  - **Weight decay (PyTorch AdamW, decay $`\eta\lambda`$ per step): $`\lambda \propto \text{fan-in}`$ on hidden and readout matrices**, so $`\eta\lambda`$ doesn't change with width.
- How much of this matters in practice: "μP-simple" — only LR $`\propto 1/\text{fan-in}`$ on matrices, SP init untouched — already transfers the LR ([Wortsman et al., 2023](https://arxiv.org/abs/2309.14322)), and any parametrization transfers given the right per-layer LR exponents ([Everett et al., 2024](https://arxiv.org/abs/2407.05872)). **Under Adam, the $`1/\text{fan-in}`$ LR rule is most of μP.** (Same conclusion as [μP §Why SP doesn't blow up](muP.md#why-sp-doesnt-blow-up-in-practice))

### Width in SP

- Since this document is written for μP, it helps to think through an illustrative example for SP (fan-in init, one global LR). If I double $`n`$; what do I change?
  - **Lower the LR by somewhere between $`\sqrt2`$ and $`2\times`$.** Theory's $`1/n`$ is the largest *stable* exponent as $`n \to \infty`$, not the optimum. Empirically, $`\eta_{\mathrm{opt}} \propto n^{-a}`$ with ([Haas et al., 2025](https://arxiv.org/abs/2505.22491)):
    - Adam, MLPs: $`a \approx 1`$ (halve)
    - AdamW Transformers with trainable LayerNorm gains: $`a \approx \tfrac12`$ ($`\approx 1`$ without the gains)
    - SGD + cross-entropy: $`a \approx \tfrac12`$ (divide by $`\sqrt2`$)
    - GPT-3's table: $`a \approx 0.8`$ in $`d_{model}`$ from 125M to 175B (depth, batch and length changed too)
  - Why there's no clean rule: one global LR serves tensors with different stability limits (under SGD: readout $`1/n`$, hidden $`n^{-1/2}`$, embedding and norm gains $`O(1)`$), and whichever binds sets $`a`$. Hotter than $`1/n`$, the logits grow as $`n^{1-a}`$, which cross-entropy tolerates ([μP §Why SP doesn't blow up](muP.md#why-sp-doesnt-blow-up-in-practice)). **SP transfers the neighbourhood of the optimum, not the point: re-sweep the LR at each width**, or fit $`\eta_{\mathrm{opt}}(n)`$ on a few small widths and extrapolate ([§3](#hp-scaling-laws)).
  - **Weight decay: raise $`\lambda`$ by the factor the LR came down**, so $`\eta\lambda`$ stays put. The EMA-window argument behind it ([§3](#weight-decay-as-an-ema-window)) doesn't depend on the parametrization.
  - **Watch the logits.** Growing logits are where large SP runs break: z-loss for the output logits, QK-norm or soft-capping for attention logits.
  - **The cheapest way out is μP-simple** ([above](#width-in-μp)): per-tensor LR $`\propto 1/\text{fan-in}`$, embedding LR constant. 

### Depth

- Depth-μP ([Tensor Programs VI](https://arxiv.org/abs/2310.02244)) and CompleteP ([Dey et al., 2025](https://arxiv.org/abs/2505.01618), which Complete(d)P builds on) put a multiplier on every residual branch, $`h^{\ell+1} = h^\ell + m_L^{-\alpha} F_\ell(h^\ell)`$ ($`h^\ell`$ the residual stream, $`F_\ell`$ block $`\ell`$'s attention or MLP) with $`\alpha \in [\tfrac12, 1]`$, and scale the hidden LR by $`m_L^{\alpha-1}`$ (and in-block $`\epsilon`$ by $`m_L^{-\alpha}`$, since the multiplier scales those gradients too).
- Worth restating: **whatever $`\alpha`$, multiplier × LR $`= m_L^{-1}`$.** Each block moves the residual stream by $`\Theta(1/L)`$ per step, and $`L`$ blocks add coherently to $`\Theta(1)`$ — the correlated-sum argument again, summed over depth instead of width. $`\alpha`$ only decides how that $`1/L`$ is split between forward multiplier and LR, which matters at init, where the blocks are independent (a random sum):
- See also [Transformer notes](../architecture/attention_transformers/notes.md#additional-details): GPT-2 scales residual-layer init by $`1/\sqrt{N}`$ ($`N`$ residual layers). That's the $`\alpha = \tfrac12`$ random-sum fix applied to *init only*; Adam's per-entry step ignores the init scale, so the trained sum isn't controlled. [Parametrization ≠ initialization](muP.md#abc-parametrization), now along depth.

### Beyond the table

- Muon sidesteps most of the width table: it replaces each matrix update by its orthogonalized version (all singular values 1), so updates have a fixed spectral norm. Scaled by $`\sqrt{\text{fan-out}/\text{fan-in}}`$ ([Norms §Muon normalization schemes](norms.md#muon-normalization-schemes)), that is the spectral form of μP (weights and updates both have spectral norm $`\propto \sqrt{\text{fan-out}/\text{fan-in}}`$), so the LR transfers across width by construction. 
- Per-module HPs transfer too (Complete(d)P §3): LR, weight decay, Adam $`\beta_1, \beta_2, \epsilon`$, and init scale.
- Norms flatten the LR landscape: e.g. QK-LayerNorm ([Wortsman et al., 2023](https://arxiv.org/abs/2309.14322), [Methods of improving LLM training stability](https://arxiv.org/abs/2410.16682)), keep attention and output logits bounded, which widens the range of near-optimal LRs. So a slightly-off transferred LR costs less; a complement to μP, not a replacement (QK-norm also caps attention sharpness, which may hurt long context).
- How small the proxy can be: transfer assumes proxy and target are close to the same infinite-width limit, so a proxy that's too narrow has a slightly different optimum.
  - It's a floor on the proxy's own size. Past the floor, TP5 transfers from width 256 to 8192, and from a 40M proxy to GPT-3 6.7B.
  - TP5's floor for Transformer LMs: width $`d_{model} \approx 256`$, depth ≈ 4, batch ≈ 32, sequence length ≈ 128, ≈ 5000 steps. Soft, not a cliff: in Complete(d)P an LR tuned at ≥136M is optimal at 483M; tuned at 58M it costs a little.
  - Test: tune at two small sizes. If the optimum agrees, the smaller one is a safe proxy.
- **What μP doesn't include**: transfer across batch size and training length.
  - TP5's "LR transfers across batch size" grew the batch at fixed *steps* (more data), which by §3 changes nothing. At fixed data a bigger batch means fewer steps, and the LR must move.
  - At fixed batch, more tokens lower the optimal LR even under μP ([Bjorck et al., 2025](https://arxiv.org/abs/2409.19913)).
  - Both are the steps column of the §0 table, derived in [§3](#3-scaling-data-at-a-fixed-model).

## 3. Scaling data at a fixed model

_Sources: Complete(d)P §2.2–2.3 for the rules, [How To Scale](https://howtoscalenn.github.io/) for the rest._

- The question: same model, more data\. How do $`\eta`$, $`\lambda`$, $`\beta`$, $`\epsilon`$ move?
- Empirically, at fixed batch, longer training wants a smaller peak LR, μP or not. 
    - Bjorck et al. fit $`\eta_{\mathrm{opt}} \propto D^{-p}`$ with $`p \approx 0.3`$–$`0.7`$ depending on model size; 
    - Complete(d)P finds $`\eta_{\mathrm{opt}} \propto \kappa^{-1/2}`$ for $`\kappa\times`$ the steps.
- The answer (derived below): the HPs depend on $`B`$ and $`D`$ only through the step count $`S = D/B`$: <div align="center">
  $`\displaystyle \eta \propto S^{-1/2}, \quad \lambda \propto S^{-1/2}, \quad \epsilon \propto S^{1/2}, \quad 1-\beta \propto S^{-1}`$ </div>
  - **The hyperparameters care about steps, not tokens.** Double $`D`$ and $`B`$ together: change nothing. Double $`D`$ at fixed $`B`$: $`\eta, \lambda \div \sqrt2`$, $`\epsilon \times \sqrt2`$, $`1-\beta \div 2`$ (e.g. $`\beta_2`$: 0.95 → 0.975).

### The SDE picture

- Model the optimizer as a stochastic differential equation ([Malladi et al., 2022](https://arxiv.org/abs/2205.10287)). Small batch, so noise dominates: the minibatch gradient is $`g + \sigma e`$ ($`g`$ true gradient, $`e`$ unit Gaussian noise, $`\sigma \propto 1/\sqrt B`$), and Adam's denominator $`\sqrt v`$ (root of its second-moment estimate) ≈ $`\sigma`$:
  1. One step: $`\theta \leftarrow \theta - \eta\, g/\sigma - \eta\, e`$ (drift $`\eta g/\sigma`$, noise std $`\eta`$)
  2. Over $`k`$ steps the drift adds up to $`k\eta\, g/\sigma`$ and the noise to a Gaussian with variance $`k\eta^2`$. 
    - Measure time as $`t = k\eta^2`$, so the noise is a standard random walk; the drift is then $`t \cdot g/(\eta\sigma)`$, i.e. $`g/(\eta\sigma)`$ per unit time
  3. So training integrates $`d\Theta = -\tfrac{g}{\eta\sigma}\, dt + dW`$ ($`\Theta`$ the weights in continuous time, $`W`$ Brownian motion) up to horizon $`T = S\eta^2`$
- **Two numbers define the run: the horizon $`T = S\eta^2`$ and the signal-to-noise coefficient $`1/(\eta\sigma)`$.** Two configurations with the same $`T`$ and the same SNR are the same run at different resolutions, so they share their optimal HPs.

### From the SDE to the rule

1. **Same data, bigger batch: exact.** $`B \to \kappa B`$ gives $`\sigma \to \sigma/\sqrt\kappa`$ and $`S \to S/\kappa`$. Set $`\eta \to \sqrt\kappa\,\eta`$: $`T`$ is unchanged, and so is $`1/(\eta\sigma)`$. Identical SDE, so the HPs transfer by construction.
    - An intuitive statement is that "my gradient is cleaner so I can take a larger step".
2. **More data via a bigger batch, same steps: the one empirical fact.** $`B \to \kappa B`$ with $`S`$ and every HP unchanged: $`T`$ is unchanged, $`\sigma \to \sigma/\sqrt\kappa`$, so SNR rises by $`\sqrt\kappa`$. **This is a different, better run** (the extra data shows up as less noise). Empirically, the optimal LR doesn't move (Complete(d)P Fig. 14).
3. **More data at the same batch: matching to move 2.** $`S \to \kappa S`$ with $`B`$ fixed. Set $`\eta \to \eta/\sqrt\kappa`$: $`T`$ is unchanged and SNR rises by $`\sqrt\kappa`$, similar to scenario 2.

- All three moves agree with $`\eta \propto S^{-1/2}`$, the rule at the top.
- The other HPs follow from the same matching:
  - $`\lambda \propto \eta`$ keeps the decay drift fixed (decay adds $`\eta\lambda\theta`$ per step, i.e. $`\lambda\theta/\eta`$ per unit time)
  - $`1-\beta \propto 1/S`$ keeps the momentum window (≈ $`1/(1-\beta)`$ steps) a fixed share of the run
  - $`\epsilon`$ only matters relative to the $`\sqrt v \approx \sigma`$ it's added to, so match $`\epsilon/\sigma`$: it's fixed in move 1 and rises $`\sqrt\kappa`$ in move 2 (so move 3 raises $`\epsilon`$ by $`\sqrt\kappa`$)
- Why SGD gets a *linear* rule instead ([McCandlish](#critical-batch-size)): its noise per step is $`\eta\sigma`$, so in move 1 holding $`T = S\eta^2\sigma^2`$ while $`S\sigma^2 \to S\sigma^2/\kappa^2`$ needs $`\eta \to \kappa\eta`$. **Adam divides out one factor of $`\sigma`$, which turns linear into square root.**
- Caveats: one setup (RedPajama, cosine schedule, ≤7B). The schedule has to be stretched to the new $`S`$ ([§1](#1-scaling-compute-n-vs-d)). 

### Weight decay as an EMA window

- Same answer from a different angle ([Wang & Aitchison, 2025](https://arxiv.org/abs/2405.13698)). PyTorch AdamW is $`\theta_t = (1 - \eta\lambda)\,\theta_{t-1} - \eta u_t`$ ($`u_t`$ Adam's normalized update): the weights are an EMA of past updates with a window of $`\tau = 1/(\eta\lambda)`$ steps.
- Empirically the best window is a roughly fixed **fraction of the run**, $`\tau/S \approx`$ const across model and dataset size. So $`\eta\lambda \propto 1/S`$.
- With independent decay ($`\theta \leftarrow (1-\lambda)\theta`$) the window is $`1/\lambda`$: no width factor, and $`\lambda \propto 1/S`$.

### Critical batch size

_Source: [McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)._

- Train near the critical batch size $`B_{\mathrm{crit}}`$ (if you have the compute), estimated by the gradient noise scale $`B_{\mathrm{simple}}`$.
  - Definitions
    - $`B_{\mathrm{crit}} := E_{\min} / S_{\min}`$, where $`S_{\min}`$ is the fewest steps to reach a target loss (noise-free, $`B \to \infty`$) and $`E_{\min}`$ the fewest examples ($`B \to 0`$). The batch that overpays both by the same factor (≈2×) — the knee of the time/compute tradeoff.
    - $`B_{\mathrm{noise}} := \mathrm{tr}(H\Sigma) / (G^\top H G)`$: the batch at which gradient noise equals gradient signal, measured in the curvature metric. $`G`$ true gradient, $`\Sigma`$ per-example gradient covariance, $`H`$ Hessian.
    - $`B_{\mathrm{simple}} := \mathrm{tr}(\Sigma) / \|G\|^2`$: same with $`H \to I`$. The one you can measure.
  - Empirically $`B_{\mathrm{crit}} \approx B_{\mathrm{noise}} \approx B_{\mathrm{simple}}`$, each within an $`O(1)`$ factor, MNIST through Dota.
  - SNR view: $`\mathbb{E}\|G_B\|^2 = \|G\|^2 (1 + B_{\mathrm{noise}} / B)`$, so at $`B = B_{\mathrm{noise}}`$ the minibatch gradient is half signal, half noise.
  - Importantly, $`B_{\mathrm{crit}}`$ is a function of the data, the optimizer and training progress. Not model size (per recent work, see below).
- Below $`B_{\mathrm{crit}}`$, doubling $`B`$ ≈ halves the steps needed. Above, steps stop shrinking (wasting compute).
- $`B_{\mathrm{crit}}`$ grows during training ($`\|G\|`$ falls faster than the noise), so ramp the batch size up. Larger for harder tasks: tens for MNIST, millions for Dota.
  - In practice: DeepSeek-V3 ramps 12.6M → 62.9M tokens over its first 469B tokens; MiniMax-01 goes 16M → 128M, doubling along a fitted $`B_{\mathrm{crit}}(L)`$.
- Measuring: compare per-worker (batch $`B_{\mathrm{small}}`$) and all-reduced (batch $`B_{\mathrm{big}}`$) gradient norms.
  - $`\|G\|^2 \approx \frac{B_{\mathrm{big}} \|G_{\mathrm{big}}\|^2 - B_{\mathrm{small}} \|G_{\mathrm{small}}\|^2}{B_{\mathrm{big}} - B_{\mathrm{small}}}`$, $`\mathrm{tr}(\Sigma) \approx \frac{\|G_{\mathrm{small}}\|^2 - \|G_{\mathrm{big}}\|^2}{1/B_{\mathrm{small}} - 1/B_{\mathrm{big}}}`$. Average over steps before taking the ratio.
- How $`B_{\mathrm{crit}}`$ scales:
  - $`B_{\mathrm{noise}} = \mathrm{tr}(\Sigma)/\|G\|^2`$ is measured at the current weights: early on the true gradient $`\|G\|`$ is large and a small batch already sees it; later $`\|G\|`$ shrinks faster than the per-example noise, so it takes a bigger batch to see the signal. **So $`B_{\mathrm{crit}}`$ depends on the domain, the optimizer, and how far training has got.**
  - The papers disagree on how to measure "how far":
    - Kaplan et al.: by the loss. $`B_{\mathrm{crit}}(L) \approx B_* / L^{1/\alpha_B}`$ ($`B_* \approx 2 \times 10^8`$ tokens, $`\alpha_B \approx 0.21`$): doubles for every ~13% drop in loss. This implies a bigger model, which reaches a lower loss on the same data, gets a bigger $`B_{\mathrm{crit}}`$.
    - Newer work: by tokens seen. $`B_{\mathrm{crit}}`$ grows with the run's token count $`D`$ and barely with model size ([Zhang et al., 2024](https://arxiv.org/abs/2410.21676); Power Lines: $`B_{\mathrm{crit}} \propto D^{\approx 0.5}`$; StepFun's $`B_{\mathrm{opt}} \propto D^{0.57}`$, [below](#hp-scaling-laws)).
    - Kaplan's evidence is two small models (3M, 85M); the newer work goes to ~1B. 
  - Consequences:
    - Within a run: ramp the batch as $`B_{\mathrm{crit}}`$ grows (DeepSeek-V3, MiniMax-01 above).
    - Across runs, for this section's question (same model, double the data): $`B_{\mathrm{crit}}`$ grows only ≈ $`\sqrt2`$. So we can grow the batch by up to ≈ $`\sqrt2`$ (for throughput) and put the remaining ≈ $`\sqrt2`$ into more steps (the $`S^{-1/2}`$ rules).
    - Across optimizers: Muon has higher $`B_{\mathrm{crit}}`$ than AdamW ([Essential AI, 2025](https://arxiv.org/abs/2505.02222)).
- Confusion worth clearing up — "optimal" vs "critical" batch size. If *every* HP is rescaled with it, choosing any the batch size below $`B_{\mathrm{crit}}`$ shouldn't change the final loss; it's a throughput knob ([tuning playbook](https://github.com/google-research/tuning_playbook)). The "$`B_{\mathrm{opt}}`$" of HP scaling laws exists because those sweeps retune only $`\eta`$ and freeze $`\beta`$, $`\epsilon`$, $`\lambda`$ (How To Scale's guess).

### HP scaling laws

- The empirical alternative: grid-search small runs, keep the near-optimal ones, fit $`\eta_{\mathrm{opt}}`$ and $`B_{\mathrm{opt}}`$ as power laws.
  - [DeepSeek LLM](https://arxiv.org/abs/2401.02954): $`\eta_{\mathrm{opt}} = 0.3118\, C^{-0.125}`$, $`B_{\mathrm{opt}} = 0.2920\, C^{0.3271}`$ (SP, multi-step schedule).
  - [StepFun](https://arxiv.org/abs/2503.04715): $`\eta_{\mathrm{opt}} = 1.79\, N^{-0.713} D^{0.307}`$, $`B_{\mathrm{opt}} = 0.58\, D^{0.571}`$ (cosine to a fixed $`10^{-5}`$ floor). Dense and MoE with the same total params land on the same HPs.
  - [AFM (Apple, 2024)](https://arxiv.org/abs/2407.21075): a hybrid. LR from a $`d_{model} = 768`$ proxy transferred with μP-simple (still a slight left-shift for much deeper/larger models); $`B`$ from a scaling law in $`(N, C)`$, with results flat over 0.5–2× the predicted batch; independent weight decay 3.16e-4 held fixed across all sizes and budgets.
  - Only valid for the setup they were fit on (schedule, FLOPs-per-token formula, width/depth ratio). How To Scale plugs DeepSeek's 7B and 67B configs into each law and gets peak LRs 2–3× apart, StepFun the outlier.
- Open question: StepFun's LR *rises* with $`D`$. Along its own $`B_{\mathrm{opt}} \propto D^{0.571}`$ the step rule ($`\eta \propto S^{-1/2} = (B/D)^{1/2}`$) predicts $`\eta \propto D^{-0.21}`$ — off by $`D^{0.52}`$, basically $`\sqrt D`$. Power Lines' $`\tau`$ is off from the step rule by the same $`\sqrt D`$.

### Summary: I double my data

| If | Then |
|---|---|
| $`2B`$ is still ≤ $`B_{\mathrm{crit}}`$ | Double $`B`$, change nothing else |
| batch stays fixed | $`\eta, \lambda \div \sqrt2`$; $`\epsilon \times \sqrt2`$; $`1-\beta \div 2`$; stretch the schedule to $`2S`$ |
| anything in between | Only $`S = D/B`$ matters: apply the $`S`$ rules with the new step count |
| the extra data is repeats | ≤ 4 epochs ≈ fresh; stop before epochs* ([§1](#1-scaling-compute-n-vs-d)); lean on weight decay |
