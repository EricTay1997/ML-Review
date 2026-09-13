# μP and μTransfer (Warning - fully LLM generated. I haven't gotten to clean this up yet)

How to scale initialization, forward multipliers and per-layer learning rates *together* as width grows, so that training dynamics stay comparable across widths — and the payoff, μTransfer: tune hyperparameters on a small model and copy them to the big one. Primary sources: Yang & Hu, [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/html/2011.14522v3) (Tensor Programs IV — the theory) and Yang et al., [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/html/2203.03466v2) (the recipe). See also [NTK](ntk.md) — the kernel / lazy limit that μP is designed to escape — and [Initialization](../../fundamentals/dl/03_initialization/notes.md) for the fan-in variance argument this builds on.

The one-line version: **when width changes, init and optimizer have to be rescaled together so that every layer keeps learning by a nonvanishing amount without activations or logits blowing up.** Everything below is working out what "together" means, tensor by tensor.

## A parametrization is more than an initialization

- Let's reconcile something first, because it's the crux: two networks can have the *same distribution over functions at init* and still train completely differently
- Take $`h = \widetilde W x`$ with $`\widetilde W_{ij} \sim \mathcal{N}(0, 1/n)`$ (ordinary fan-in init) vs $`h = \frac{1}{\sqrt n} W x`$ with $`W_{ij} \sim \mathcal{N}(0, 1)`$ (explicit multiplier). At init $`\widetilde W = W / \sqrt n`$ exactly, so identical functions
- Now do one SGD step with LR $`\eta`$ on each:
  - $`\nabla_W L = \frac{1}{\sqrt n} \nabla_{\widetilde W} L`$ (chain rule through $`\widetilde W = W / \sqrt n`$)
  - $`\Delta W = -\eta \nabla_W L`$ induces $`\Delta \widetilde W = \Delta W / \sqrt n = -\frac{\eta}{n} \nabla_{\widetilde W} L`$
  - i.e. **LR $`\eta`$ on $`W`$ ≡ LR $`\eta / n`$ on $`\widetilde W`$**. The multiplier silently divided the learning rate by $`n`$
- So a parametrization is the triple (init variance, forward multiplier, per-tensor LR), each as a function of width. Yang & Hu formalize this as an **abc-parametrization**: $`W = n^{-a} w`$ with $`w_{ij} \sim \mathcal{N}(0, n^{-2b})`$ and LR $`\eta n^{-c}`$, per tensor. SP, NTKP and μP are just different choices of $`(a, b, c)`$, and comparing them on init alone is meaningless
- (Under SGD there's a symmetry $`(a, b, c) \to (a + \theta, b - \theta, c - 2\theta)`$ that leaves the dynamics unchanged — this is why "μP" gets written with different-looking tables. Under Adam the equivalence breaks; see [Base width](#base-width-and-the-practical-table))

## Running example: a 2-hidden-layer MLP

- Forward pass: <div align="center">
  $`\displaystyle h^1 = W^1 x,\ x^1 = \phi(h^1); \quad h^2 = W^2 x^1,\ x^2 = \phi(h^2); \quad f(x) = W^3 x^2`$ </div>
- $`W^1 \in \mathbb{R}^{n \times d}`$, $`W^2 \in \mathbb{R}^{n \times n}`$, $`W^3 \in \mathbb{R}^{k \times n}`$, with $`d`$ (input dim) and $`k`$ (output dim) fixed and the hidden width $`n \to \infty`$
- The three matrices play structurally different roles, and this classification is what everything else hangs on:

| Matrix | Maps | μP name |
|---|---|---|
| $`W^1`$ | finite → infinite | input / vector-like |
| $`W^2`$ | infinite → infinite | hidden / matrix-like |
| $`W^3`$ | infinite → finite | output / vector-like |

- "Infinite" = the dimension scales with the width parameter of the model family. The rule is about *which dimensions grow*, not what the layer is called — e.g. biases and LayerNorm gains are vector-like, and a scalar like an attention temperature has no growing dimension and stays constant

## What should stay well-behaved as n → ∞

For a fixed number of steps, three things:

- **Stability**: hidden preactivations $`h^\ell_i = \Theta(1)`$ at init and $`O(1)`$ throughout training; logits $`O(1)`$ (they may → 0, they must not blow up)
- **Nontrivial learning**: $`f_t(x) - f_0(x) = O(1)`$ — the infinite-width network actually changes its function
- **Feature learning**: $`h^\ell_t(x) - h^\ell_0(x) = O(1)`$ coordinatewise. Equivalently, the feature kernel $`K^{\mathrm{feat}}_t(x, x') = \frac{1}{n} h^\ell_t(x)^\top h^\ell_t(x')`$ moves by $`O(1)`$. If hidden changes vanish, features freeze and training collapses to a kernel method — the [NTK](ntk.md) picture
- **Dynamical dichotomy** ([Yang & Hu](https://arxiv.org/html/2011.14522v3)): among stable, nontrivial abc-parametrizations, the infinite-width limit is *either* a kernel limit *or* a feature-learning limit — never both. SP and NTKP (scaled to be stable) land on the kernel side; μP is on the feature-learning side, and specifically it's the one where *every* layer's features move by the maximal stable amount
- Scope of the theorem: width → ∞ with depth and the number of steps fixed; tanh or a smooth approximation of ReLU for the formal classification. μTransfer's empirical results across depth / batch / sequence length / training duration go beyond what's proved

## The key intuition: random sums vs correlated sums

- At init, weights and activations are independent and centered. $`h_i = \sum_{j=1}^n W_{ij} x_j`$ with $`W_{ij} = O(n^{-1/2})`$ is $`n`$ random-sign terms → CLT → $`h_i = O(1)`$. This is exactly what fan-in init ([Initialization](../../fundamentals/dl/03_initialization/notes.md)) is designed to control
- A gradient update has a different structure: it's an outer product, $`\Delta W_{ij} \propto \delta_i x_j`$ (upstream gradient × downstream activation). On the *next* forward pass, with a similar $`x`$:
  - $`(\Delta W x)_i \propto \delta_i \sum_j x_j^2 = \delta_i \cdot O(n)`$
  - the update is *correlated* with the activation it multiplies, so the terms add coherently — no cancellation
- **Important intuition**: an independent centered sum of $`n`$ terms is $`O(\sqrt n)`$; a correlated sum is $`O(n)`$. Fan-in init controls the first kind. A parametrization that's consistent across width *during training* has to control the second kind too — and that's a statement about learning rates and multipliers, not about init
- (Same phenomenon in attention: $`q^\top k`$ is a random sum at init and a correlated sum after training — see [Transformers](#transformers))

## Standard parametrization (SP)

- Fan-in init, one global LR: $`W^1_{ij} \sim \mathcal{N}(0, 1/d)`$, $`W^2_{ij} \sim \mathcal{N}(0, 1/n)`$, $`W^3_{ij} \sim \mathcal{N}(0, 1/n)`$, and $`\eta_{W^1} = \eta_{W^2} = \eta_{W^3} = \eta`$
- Stable at init (every preactivation has $`O(1)`$ variance), but init stability ≠ training stability. Track the first SGD step on the readout of the running MLP:
  - $`\Delta W^3_j = -\eta\, \partial_f L\, x^2_j`$, so $`\Delta f = \Delta W^3 x^2 = -\eta\, \partial_f L\, \|x^2\|^2 = O(\eta n)`$ — a correlated sum
  - With $`\eta = O(1)`$ the logits move by $`O(n)`$ → unstable. To keep them $`O(1)`$ you need $`\eta_{\mathrm{SP}} = O(1/n)`$
  - But at $`\eta = O(1/n)`$ the hidden layers barely move: $`\delta^2_i = O(n^{-1/2})`$ (it backpropagates through $`W^3`$), so $`(\Delta W^2 x^1)_i = -\eta\, \delta^2_i\, \|x^1\|^2 = O(\eta \sqrt n) = O(n^{-1/2}) \to 0`$
- So the stable SP limit is a **kernel limit**: features freeze, and the network learns only through the lazy mechanism in the [NTK notes](ntk.md#the-infinite-width-limit). (Not the *same* kernel as NTKP — in the running MLP the input layer's effective LR is $`1/n`$ under SP but $`\eta / d = O(1)`$ under NTKP, so the input layer drops out of SP's kernel. Verify against source; the paper only states that the kernels differ)
- Important qualification: **finite-width SP networks obviously do learn features.** The claim is about the $`n \to \infty`$ limit and — the practically relevant version — about *consistency across width*: SP has no per-tensor balancing, so as you widen the model, the LR that was right becomes wrong. This is the empirical fact μTransfer fixes ([TP5, Fig. 1](https://arxiv.org/html/2203.03466v2): under SP the optimal LR drifts left as width grows)

## Neural tangent parametrization (NTKP)

- $`h^{\ell+1} = \frac{1}{\sqrt n} W^{\ell+1} x^\ell`$ with $`W^{\ell+1}_{ij} \sim \mathcal{N}(0, 1)`$ and an $`O(1)`$ LR on the raw $`W`$ — the setup in [NTK](ntk.md#the-infinite-width-limit)
- The effective matrix $`\widetilde W = W / \sqrt n`$ has the same $`1/n`$ init variance as SP's hidden matrix, but by [§1](#a-parametrization-is-more-than-an-initialization) its effective LR is $`\eta / n`$. NTKP is *by construction* the "SP with an $`O(1/n)`$ LR" regime, applied uniformly to every layer
- 1-hidden-layer version: $`f = \frac{1}{\sqrt n} \sum_i a_i \phi(w_i^\top x)`$. Every per-neuron derivative carries $`1/\sqrt n`$, so $`\Delta a_i, \Delta w_i = O(n^{-1/2})`$ → each feature $`\phi(w_i^\top x)`$ moves by → 0, while the $`n`$ coherent contributions still move the output by $`O(1)`$
  - **feature change per neuron → 0, total output change $`= O(1)`$**
- So the network is described by its linearization $`f_t \approx f_0 + \nabla_\theta f_0^\top (\theta_t - \theta_0)`$ and its tangent kernel stays put, $`K_t \approx K_0`$. Which is the point — NTKP is the right choice when you *want* a tractable kernel limit. It was never meant to keep representation learning at infinite width

## Maximal update parametrization (μP)

- μP asks a different question: **what is the largest update each tensor can take while the whole network stays stable?**
- "Maximal" does *not* mean every raw entry moves by $`O(1)`$. It means each tensor's effect on its layer's *preactivations*, $`\Delta W^\ell x^{\ell-1}`$, is as large as stability permits — $`O(1)`$
- The central mechanism, for a hidden matrix. Suppose $`\Delta W_{ij} = O(1/n)`$ — tiny next to an init entry of $`O(n^{-1/2})`$. But the update is correlated with $`x`$, so
  - $`(\Delta W x)_i = \sum_{j=1}^n \Delta W_{ij} x_j = n \cdot O(1/n) = O(1)`$
  - **many coordinated $`O(1/n)`$ weight updates → an $`O(1)`$ representation change.** Weight movement can be negligible entrywise, and even in Frobenius norm ($`\|\Delta W\|_F / \|W_0\|_F \to 0`$), while feature learning is $`O(1)`$ — so "the weights barely moved" is not evidence of lazy training
- Direct asymptotic form for the running MLP under SGD ($`\eta`$ is a width-independent master LR; $`n`$-independent constants dropped):

| Parameter | Init variance | SGD LR |
|---|---:|---:|
| Input $`W^1 : d \to n`$ | $`1/d`$ | $`\eta n`$ |
| Hidden $`W^2 : n \to n`$ | $`1/n`$ | $`\eta`$ |
| Output $`W^3 : n \to k`$ | $`1/n^2`$ | $`\eta / n`$ |

- Why each row is what it is — it's all the correlated-sum bookkeeping:
  - **Output LR $`\eta / n`$**: $`\Delta f = \Delta W^3 x^2`$ is a correlated sum over $`n`$ terms (the SP failure above), so divide the LR by $`n`$
  - **Output init $`1/n^2`$** (entries $`O(1/n)`$, not $`O(n^{-1/2})`$): once training correlates $`W^3`$ with $`x^2`$, $`f = \sum_j W^3_j x^2_j`$ is $`n`$ coherent terms, which need to be $`O(1/n)`$ each to keep $`f = O(1)`$. The init entries are put on that same scale so the random and learned parts of $`W^3`$ live on the same footing
  - **Hidden LR $`\eta`$**: because $`W^3 = O(1/n)`$, the backpropagated $`\delta^2 = O(1/n)`$, so the raw gradient $`\delta^2_i x^1_j`$ is *already* $`O(1/n)`$ per entry — precisely the scale the mechanism above wants. No correction needed
  - **Input LR $`\eta n`$**: same $`\delta^1 = O(1/n)`$, but $`h^1 = W^1 x`$ sums over only $`d`$ (finite) terms, so there's no width-sized sum to amplify the update. Multiply the LR by $`n`$ to get $`\Delta h^1 = O(1)`$
- Side effect worth knowing: the **initial output vanishes**. At init $`W^3`$ is *uncorrelated* with $`x^2`$, so $`f_0 = O(\sqrt n \cdot 1/n) = O(n^{-1/2}) \to 0`$, while $`f_t - f_0 = O(1)`$ after training. Not a defect — Yang & Hu show *any* stable, nontrivial feature-learning parametrization has $`f_0 \to 0`$ in the limit. (Practical corollary: zero-initializing the readout improves finite-width transfer, because it removes the mismatch between a small proxy's $`O(1)`$ random logits and the limit's zero ones)
- Another way to state all of this ([Yang, Simon & Bernstein, *A Spectral Condition for Feature Learning*](https://arxiv.org/abs/2310.17813) — beyond the two primary sources): μP ⇔ every weight matrix has $`\|W\|_2 = \Theta(\sqrt{n_{out} / n_{in}})`$ **and** $`\|\Delta W\|_2 = \Theta(\sqrt{n_{out} / n_{in}})`$. A rank-1 update with spectral norm 1 on an $`n \times n`$ matrix has entries $`O(1/n)`$ — that's the hidden row of the table in one line. This is also the bridge to spectral-norm-aware optimizers like Muon (to write up in [notes.md](notes.md))

## Base width and the practical table

- Literal factors like $`\eta n`$ are asymptotics. In practice, pick a **base width** $`n_0`$ (a model you'd have trained anyway, in SP), define $`m = n / n_0`$, and arrange μP so that at $`m = 1`$ it *is* the ordinary SP model — all the $`n_0`$ factors get absorbed into the master HPs
- For an MLP whose hidden dims all scale by $`m`$ ([TP5, Table 3](https://arxiv.org/html/2203.03466v2), with fan-in written in terms of $`m`$):

| Parameter type | Init at target width | SGD LR | Adam LR |
|---|---|---:|---:|
| Input weights | ordinary fan-in | $`\eta m`$ | $`\eta`$ |
| Hidden biases | usually zero | $`\eta m`$ | $`\eta`$ |
| Hidden matrices | ordinary fan-in | $`\eta`$ | $`\eta / m`$ |
| Output weights | ordinary fan-in variance $`\times\, 1/m`$ | $`\eta / m`$ | $`\eta / m`$ |

- At $`m = 1`$ every extra factor is 1. For the output layer, variance $`\times 1/m`$ means std $`\times 1/\sqrt m`$
- **Why Adam has different rules**: Adam normalizes each coordinate by its running gradient magnitude, so its per-entry step is ≈ $`\eta`$ regardless of how big the gradient was. That erases the gradient's own width scaling — which is exactly what SGD was relying on:
  - hidden: SGD got the needed $`1/n`$ per-entry update for free from $`\delta^2 = O(1/n)`$; Adam throws that away, so put $`1/m`$ back by hand
  - input: SGD needed $`\times m`$ to *compensate* for the small $`\delta^1`$; Adam already normalized it away, so no factor
  - output: the correlated sum over $`n`$ is in the forward pass, not in the gradient, so both optimizers need the $`1/m`$
- So "we use μP" is underspecified until you say which optimizer and which convention. Equivalent variants move factors between init, multipliers and LR — TP5's reference implementation (`MuReadout` in the `mup` package) uses an explicit $`1/m`$ output *multiplier* instead of the shrunken init — and base-shape implementations do the bookkeeping by comparing each tensor's target shape to its base shape

## SP vs NTKP vs μP

| | SP | NTKP | μP |
|---|---|---|---|
| Forward pass stable at init | yes | yes | yes |
| Width-aware optimization | no (one global LR) | via the $`1/\sqrt n`$ multiplier, uniformly | explicit, per tensor type |
| Fixed $`O(1)`$ master LR stable as $`n \to \infty`$ | no | yes, in raw $`W`$ coordinates | yes, after per-tensor scaling |
| Hidden features in the stable limit | freeze | freeze | move by $`O(1)`$ |
| Feature kernel during training | fixed in the limit | fixed | evolves |
| Infinite-width dynamics | kernel | NTK / kernel | feature learning |
| Initial logits | $`O(1)`$ random | $`O(1)`$ random | → 0 |
| What it's for | the finite-width default | a tractable kernel limit | feature learning + HP transfer |

- The distinction to keep straight: NTKP is a kernel parametrization *on purpose*; SP wasn't designed as one, but its stable limit ends up kernel-like; μP is a feature-learning parametrization on purpose. SP and NTKP don't share a kernel — they share the property that hidden representations freeze

## Transformers

- Same finite / infinite classification. If $`d_{model}`$ and $`d_{ffn}`$ scale while the vocabulary is fixed:
  - the token embedding ($`V \to d_{model}`$) is input-like; the unembedding ($`d_{model} \to V`$) is output-like
  - every internal projection ($`W^{q}, W^{k}, W^{v}, W^{o}`$, FFN up / down) maps growing → growing: hidden / matrix-like
  - biases and LayerNorm gains: vector-like; scalars (attention temperature, residual multipliers) stay constant
  - which rule applies depends on which dimensions *actually* change in your model family — e.g. you can widen by scaling $`d_{head}`$ at fixed $`n_{heads}`$, or $`n_{heads}`$ at fixed $`d_{head}`$, and the attention scaling below only bites in the first case
- **Attention scaling**: TP5 uses $`q^\top k / d_{head}`$ instead of the usual $`q^\top k / \sqrt{d_{head}}`$
  - Recall the [$`\sqrt{d_k}`$ motivation](../architecture/attention_transformers/notes.md#attention): with *independent* unit-variance entries, $`q^\top k`$ has variance $`d_k`$, so $`/\sqrt{d_k}`$ gives unit-variance logits. That's the init-time, random-sum argument
  - After training, $`q`$ and $`k`$ are correlated (that's what learning to attend *is*), and their inner product is a correlated sum scaling like $`d`$. The $`1/d`$ controls the learned quantity, at the cost of near-uniform attention at init — which μP accepts, same as it accepts vanishing initial logits
  - To agree with standard attention at a base head dim $`d_{head, 0}`$: $`\mathrm{AttnLogit} = \alpha_{attn}\, \frac{\sqrt{d_{head, 0}}}{d_{head}}\, q^\top k`$, with $`\alpha_{attn}`$ a width-independent tunable multiplier (at $`d_{head} = d_{head, 0}`$ this is the usual $`1/\sqrt{d_{head, 0}}`$)
  - **Conventional scaling controls variance at init; μP controls the whole training trajectory.** Same sentence as [§4](#the-key-intuition-random-sums-vs-correlated-sums), applied to attention
- TP5 also recommends zero-initializing the query projection along with the readout, for the same finite-width-transfer reason

## μTransfer

- Write the loss after $`T`$ steps at width $`n`$ with master HPs $`\lambda`$ as $`\mathcal{L}_n(\lambda)`$. If the parametrization is width-consistent, the whole HP landscape should converge, $`\mathcal{L}_n(\lambda) \to \mathcal{L}_\infty(\lambda)`$, and so should its argmin: $`\arg\min_\lambda \mathcal{L}_n(\lambda) \approx \arg\min_\lambda \mathcal{L}_N(\lambda)`$ for a small proxy width $`n`$ and a big target width $`N`$
- The recipe:
  1. Pick a base model and decide which dimensions will scale
  2. Put the model family in μP (the per-tensor rules above, in terms of $`m`$)
  3. Tune the master HPs on a small proxy
  4. Scale up
  5. Copy the master HPs unchanged
  6. Let the parametrization turn them into the target model's tensor-specific values
- "Zero-shot" = no HP search *on the target*. You still train the target once. μTransfer doesn't remove tuning, it moves it: **tune once on a cheap proxy instead of re-tuning at every width**
- Headline result ([TP5](https://arxiv.org/html/2203.03466v2)): HPs tuned on a 40M-param proxy, transferred to GPT-3 6.7B, beat the published 6.7B, with total tuning cost ≈ 7% of one pretraining run. The figure to remember is their Fig. 1 — loss-vs-LR curves for widths 128 → 8192: under SP the optimum slides left with width; under μP the curves line up and wider is uniformly better

### What transfers

| Category | Examples | Status |
|---|---|---|
| Optimization HPs | master LR, momentum, Adam betas, LR schedule | theory + experiments |
| Init scales & multipliers | per-layer init, output / attention multipliers | theory + experiments |
| Regularization | dropout, weight decay | *not* assumed to transfer |
| Across width | | the theoretical setting |
| Across depth, batch size, seq length, training steps | | empirical, with caveats |
| To a new task or architecture | different data, objective, substantially different design | not guaranteed |

- The proxy has to be big enough: in TP5's experiments, below a couple hundred in width (and similarly minimal depth / batch / seq length / steps) the proxy's dynamics are qualitatively different and transfer degrades
- Depth transfer is architecture-dependent — TP5 finds it works for pre-LN Transformers and not post-LN. Depth as a *theoretical* scaling axis is its own line of work ([Tensor Programs VI](https://arxiv.org/abs/2310.02244))

### Proved, observed, open

| Status | Claim |
|---|---|
| Theorem | Stable, nontrivial abc-limits are either kernel or feature-learning (dynamical dichotomy) |
| Theorem | Stable SP and NTKP limits freeze features; μP's limit learns features in every layer |
| Theorem | The Tensor Programs machinery computes the resulting infinite-width dynamics |
| Empirical | HP optima are far more stable across width under μP than under SP |
| Empirical | Transfer also works across depth, batch, seq length and training time in the studied settings |
| Empirical | With fixed μP HPs, wider is better on training loss throughout training |
| Open | *Why* the HP optimum stabilizes long before the trained function or the loss has converged in width |

- That last one is the interesting puzzle. For μTransfer to be useful the proxy must be (a) wide enough that its HP optimum matches the target's and (b) small enough to be much cheaper — and *materially worse* — than the target. Empirically both hold: the HP optimum is a coarse quantity that converges fast in width, the learned function is a fine one that converges slowly. TP5 explicitly leaves the explanation open

## Practical checks

- **Coordinate check** (the `mup` package ships one): for several widths, log the mean $`|\cdot|`$ of each layer's activations and of the logits over the first few steps, and plot against width. Correct μP → flat in width. Anything sloping up or down with width means some tensor, multiplier, attention factor or LR is mis-scaled. This is the single most useful debugging tool, because μP is easy to get *subtly* wrong — one forgotten $`1/m`$ and you have a network that trains fine at the proxy width and silently diverges, or freezes, at the target
- **Wider-is-better**: with fixed μP HPs, increasing width should improve training loss at every step. A good sanity check, not a theorem about test performance
- **Keep the comparison controlled**: μTransfer is a claim about scaled versions of the *same* family on the *same* task. Change the data, loss, optimizer, regularization or architecture and you may need to retune even if the widths are parametrized perfectly

## Summary

- **SP** says: fan-in init and, ordinarily, one LR. Fine at any fixed width; the optimal LR drifts as width changes, and the stable infinite-width limit is lazy
- **NTKP** says: make each feature move infinitesimally while the combined output moves by $`O(1)`$. The limit is a fixed kernel method — see [NTK](ntk.md)
- **μP** says: scale each tensor by how its input and output dimensions grow, so that its effect on learned representations is as large as stability permits
- The deepest point is that μP is not an initializer. It's a coordinated prescription for **init + forward multipliers + per-tensor optimizer scaling**, chosen so that the *entire training trajectory* — not just the first forward pass — has a stable feature-learning limit
- **μP makes training dynamics comparable across width; μTransfer exploits that to tune large models cheaply**
