# μP and μTransfer

- μP (Maximal Update Parametrization) picks each tensor's init, forward multiplier and learning rate (LR) as functions of width so that every weight's update moves its layer's preactivations by the largest amount that stays stable, $`\Theta(1)`$, as width → ∞. 
- The per-tensor scaling rules and μTransfer (tune hyperparameters, HPs, on a small proxy and copy them to the big model) all follow from that one requirement. 
- Primary source: Seunghyun Seo, [How To Scale](https://howtoscalenn.github.io/) (the model-size half). The theory is Yang & Hu, [Tensor Programs IV](https://arxiv.org/abs/2011.14522) and Yang et al., [Tensor Programs V](https://arxiv.org/abs/2203.03466) (TP4 / TP5). See also [NTK](ntk.md) (neural tangent kernel: the lazy regime μP is built to escape), [Norms](norms.md) and [Muon](muon.md) (the spectral view of the same question), [Initialization](../../fundamentals/dl/03_initialization/notes.md) (fan-in variance) and [Scaling](scaling.md) (the resulting rules for width, depth, batch and duration, side by side).

## Intuition

- Every layer computes $`Av`$, where $`v \in \mathbb{R}^n`$ is an activation vector with $`\Theta(1)`$ entries and $`A`$ is a weight, a weight update, or the readout. The question, per tensor: how big must $`A`$'s entries be for $`Av`$ to have $`\Theta(1)`$ entries? TP5 (App. J) answers it with one rule of thumb for a sum of $`n`$ roughly iid terms: <div align="center">
  $`\displaystyle \sum_{i=1}^n x_i \text{ has typical size } \begin{cases} \Theta(n) & \text{if } \mathbb{E}[x_i] \neq 0 \quad \text{(law of large numbers, LLN)} \\ \Theta(\sqrt n) & \text{if } \mathbb{E}[x_i] = 0 \quad \text{(central limit theorem, CLT)} \end{cases}`$ </div>
- So it comes down to whether the terms of $`Av`$ have a nonzero mean. Three cases (TP5, Table 14):
  - **Random init matrix, $`n \times n`$** (hidden weights at init): $`W_0`$ is zero-mean and independent of $`v`$ → CLT → entries $`\Theta(1/\sqrt n)`$, i.e. fan-in init
  - **Weight update, $`n \times n`$**: a batch-1 SGD step is an outer product, $`\Delta W = -\eta\, g x^\top`$ ($`g`$ the loss gradient w.r.t. the layer's output, $`x`$ its input). On the next forward pass: <div align="center">
    $`\displaystyle \Delta W x' = -\eta\, g\, (x^\top x')`$ </div>
    - $`x`$ and $`x'`$ are the same $`n`$ neurons on two inputs, so $`x^\top x' / n \to \mathbb{E}[x_i x'_i]`$, which is often nonzero → LLN → $`\Theta(n)`$. Entries must be $`\Theta(1/n)`$
  - **Readout, $`1 \times n`$**: training correlates the last features with $`W_{out}`$ (their gradient comes through $`W_{out}^\top`$), and a single row puts all of an aligned $`\|v\| = \Theta(\sqrt n)`$ into one output → LLN → entries $`\Theta(1/n)`$, *including at init* (mechanism in [Deriving the rules](#deriving-the-rules))
  - (Input weights, $`n \times d`$ with $`d`$ fixed, sum over no growing dimension: entries $`\Theta(1)`$)
- **Important intuition**: 
  - A full-rank random matrix is a CLT object ($`1/\sqrt n`$ entries); 
  - Matrices built from a few vectors that are aligned with their input (updates, the readout) are LLN objects ($`1/n`$ entries).
  - SP (standard parametrization: fan-in init everywhere and one global learning rate, the PyTorch default) only sizes the first. μP sizes all of them, tensor by tensor

## abc parametrization

- A parametrization is three things per tensor, each as a function of width: init std, forward multiplier, learning rate. TP4 calls this an **abc-parametrization**: 
  - $`W = n^{-a} w`$ 
  - $`w \sim \mathcal{N}(0, n^{-2b})`$
  - LR $`= \eta\, n^{-c}`$ with $`\eta`$ width-independent. 
- Same function at init ≠ same training. $`\widetilde W x`$ with $`\widetilde W_{ij} \sim \mathcal{N}(0, 1/n)`$ and $`\frac{1}{\sqrt n} W x`$ with $`W_{ij} \sim \mathcal{N}(0, 1)`$ agree at init, but the multiplier also shows up in the gradient: LR $`\eta`$ on $`W`$ ≡ LR $`\eta / n`$ on $`\widetilde W`$ (the NTK parametrization's trick, see [NTK](ntk.md#the-infinite-width-limit))
- **Symmetry**: the three knobs trade off. Scaling the multiplier by $`\theta`$ and compensating as below leaves the whole training trajectory unchanged:

| Optimizer | Multiplier | Init std | LR |
|---|---|---|---|
| SGD | $`\times\,\theta`$ | $`\div\,\theta`$ | $`\div\,\theta^2`$ |
| Adam | $`\times\,\theta`$ | $`\div\,\theta`$ | $`\div\,\theta`$ (and Adam's denominator constant $`\epsilon \times \theta`$, for exactness) |

- Consequences:
  - "μP" has several different-looking tables (TP5 has three) that are the same dynamics
  - A width factor can live in the forward pass (visible in a model config) or in init + LR (invisible). This matters when [reading a model's config](#what-do-popular-models-do)

## Desiderata, and what "maximal" means

For a fixed number of steps as $`n \to \infty`$, with $`h^\ell`$ the layer-$`\ell`$ preactivations and $`f`$ the logits (sizes are per coordinate, i.e. RMS over the $`n`$ entries). Every quantity scales as a power of $`n`$, and **Θ(1) (exponent 0) means width-independent**. A nonzero exponent breaks things: positive grows without bound (overflow, saturated nonlinearities, divergence), negative vanishes (frozen features, a net that doesn't train).

- **Stability**: $`h^\ell = \Theta(1)`$ throughout training; $`f = O(1)`$ (it may start at 0)
- **Non-triviality**: $`\Delta f = \Theta(1)`$, i.e. the function actually changes
- **Feature learning**: $`\Delta h^\ell = \Theta(1)`$, i.e. the representation actually changes. If it vanishes in every layer, training is a kernel method on frozen features ([NTK](ntk.md#what-this-says-about-feature-learning))
- **Dynamical dichotomy** (TP4): every stable, non-trivial abc-parametrization has a limit that is *either* a kernel limit *or* a feature-learning limit, never both
- **Why "maximal"**: each tensor's LR $`\eta_\ell = \eta\, n^{-c}`$ sets how much the tensor's *own* update moves its layer's preactivations, $`\Delta W^\ell x^{\ell-1}`$ ($`x^{\ell-1}`$ the layer's input). Each tensor has one edge between blowing up and freezing. **"Maximal update" = sitting on it**: $`\Delta W^\ell x^{\ell-1} = \Theta(1)`$, the largest change stability allows.
- **Maximal means maximal effect on the features, not maximal weight movement**
- **Why maximal leads to transfer**
  1. **Transfer**: the optimal HPs converge as $`n \to \infty`$ (TP5, Def. A.3). In practice: copy the proxy's HPs to a wider net and they're still ≈ optimal
  2. **Necessary conditions** (TP5, App. J.3): copying the HPs must reproduce the same training at the new width. So (a) the limit at fixed HPs must be stable; and (b) nothing the proxy relies on may fade as $`n`$ grows. The assumption behind (b): the loss depends on the learnt features (true wherever feature learning matters, e.g. LLM pretraining), so how much they move must be width-independent too.
  3. **Maximal buys transfer**: maximal update is exactly "exponent 0 for every tensor", and μP is the unique stable abc-parametrization that achieves it (TP4, Thm. 5.6). Proxy and target are then the same feature-learning process, differing only by finite-width noise
  4. **Anything non-maximal doesn't**: every other abc-parametrization is unstable, trivial (never trains in the limit), or by the dichotomy kernel or non-maximal feature learning.

## Deriving the rules

Width $`n`$, base width $`n_0`$, ratio $`m = n / n_0`$ (for hidden and output weights, $`m`$ is that tensor's fan-in ratio).

**The three tensor types.** μP sorts tensors by which of their dimensions grow with width.

| Type | Examples | Maps |
|---|---|---|
| Input (vector-like) | token embedding, biases, norm gains | fixed → $`n`$ |
| Hidden (matrix-like) | $`W_q, W_k, W_v, W_o`$, MLP up / gate / down | $`n \to n`$ |
| Output (vector-like) | unembedding / LM head | $`n`$ → fixed |

**The rules** ([TP5](https://arxiv.org/abs/2203.03466), Table 3), relative to the base width, with SP in parentheses.

| Tensor | Init variance | Adam LR | SGD LR |
|---|---|---:|---:|
| Input | $`\times\, 1`$ (SP: same) | $`\eta`$ (SP: $`\eta`$) | $`\eta\, m`$ (SP: $`\eta`$) |
| Hidden | $`\times\, 1/m`$ (SP: same) | $`\eta / m`$ (SP: $`\eta`$) | $`\eta`$ (SP: $`\eta`$) |
| Output | $`\times\, 1/m^2`$ (SP: $`\times\, 1/m`$) | $`\eta / m`$ (SP: $`\eta`$) | $`\eta / m`$ (SP: $`\eta`$) |
| Attention logits | scale $`1/d_{head}`$ (SP: $`1/\sqrt{d_{head}}`$) | | |

- **Hidden** ($`\sigma`$ = entry size; under Adam each entry moves ≈ $`\eta`$ per step)
  - Init: random sum, $`W_0 x \approx \sqrt n\, \sigma \Rightarrow \sigma = 1/\sqrt n`$. Fan-in, same as SP
  - Update: aligned sum, $`\Delta W x \approx n\, \eta \Rightarrow \eta \propto 1/m`$. SP's single LR gets hotter as you widen
- **Input (embedding)**
  - Init: a lookup reads one row, $`h = E_{tok}`$ (the token's embedding): no sum over $`n`$, no width factor. Same as SP
  - Update: $`\Delta h = \Delta E_{tok} \approx \eta \Rightarrow`$ constant LR. SP's LR, retuned $`\propto 1/m`$, starves it by $`m`$
- **Output (readout)**
  - Init: features move along $`W_{out}`$ (their gradient comes through $`W_{out}^\top`$), so $`W_{out} x`$ is an aligned sum, $`n\, \sigma \Rightarrow \sigma = 1/n`$ (SP: $`1/\sqrt n`$)
    - So $`f_0 \approx \sqrt n \cdot \frac1n \to 0`$. Fine (TP4: every feature-learning limit has $`f_0 \to 0`$); zero-init in practice
  - Update: aligned sum, $`\Delta W_{out}\, x \approx n\, \eta \Rightarrow \eta \propto 1/m`$
  - Equivalently: logits $`\times\, 1/m`$ with base init and LR (`MuReadout`; `logits_scaling` / `dim_model_base` in Hugging Face (HF) configs)
- **Attention logits** ($`q^\top k`$ sums over the head dimension $`d_{head}`$)
  - Init: random sum, $`q^\top k \approx \sqrt{d_{head}} \Rightarrow`$ SP's $`1/\sqrt{d_{head}}`$ ([recall](../architecture/attention_transformers/notes.md#attention)). μP also zero-inits $`W_q`$, so logits start at 0 at every width
  - Update: once attention has learned something, $`q`$ and $`k`$ are aligned, $`q^\top k \approx d_{head} \Rightarrow`$ μP's $`1/d_{head}`$
  - Only matters if $`d_{head}`$ grows with width; most LLM families fix it (128 or 256) and add heads instead
- **SGD**: SGD doesn't normalize, so the gradient's own size matters. In μP every gradient that flows back through the readout picks up a $`1/n`$ (the readout's entries are $`1/n`$). 
  - For hidden weights, that $`1/n`$ is exactly the discount they need, so LR $`\eta`$. 
  - For the embedding it's unwanted, since there's nothing to amplify, so LR $`\times\, m`$. 
  - The readout's own gradient has no such factor, so $`1/m`$ as under Adam
- **Summary**: the readout gets too much update, so discount it; hidden layers need $`1/n`$ to counter the aligned sum; the embedding gets too little, so boost it
- The table implies what happens when width changes, not what per-layer differences look like with the base proxy ($m=1$).

## SP, SP-stable and NTK: what goes wrong

- SP-stable = SP with its global LR shrunk as $`1/n`$, the largest scaling whose $`n \to \infty`$ limit is stable. 

| | LR | Hidden $`\Delta h`$ | Embedding $`\Delta h`$ | Logits $`\Delta f`$ | $`n \to \infty`$ |
|---|---|---|---|---|---|
| SP | global $`\Theta(1)`$ | $`\Theta(\sqrt n)`$ | $`\Theta(n^{-1/2})`$ | $`\Theta(n)`$ | blows up |
| SP-stable | global $`\Theta(1/n)`$ | $`\Theta(n^{-1/2})`$ | $`\Theta(n^{-3/2})`$ | $`\Theta(1)`$ | kernel |
| NTK | $`\Theta(1)`$ on raw $`W`$, $`1/\sqrt n`$ multipliers | $`\Theta(n^{-1/2})`$ | $`\Theta(n^{-1/2})`$ | $`\Theta(1)`$ | kernel (the NTK) |
| μP | per tensor, table above | $`\Theta(1)`$ | $`\Theta(1)`$ | $`\Theta(1)`$ | feature learning |

- **SP**: no stable limit. The only fix is to shrink the LR as width grows, which is why SP's optimal LR slides left as models widen (TP5 Fig. 1; GPT-3 went from $`6 \times 10^{-4}`$ at 125M to $`0.6 \times 10^{-4}`$ at 175B)
- **SP-stable**: stable, but the LR is pinned by the most-amplified tensor (the readout), so everything else is throttled and features freeze. The embedding is hit hardest
- **NTK**: stable *by design* with an $`O(1)`$ LR, and frozen *by design* too: each neuron moves $`O(n^{-1/2})`$ while $`n`$ coherent contributions move the output by $`O(1)`$ ([NTK](ntk.md#the-infinite-width-limit)). How To Scale's example: an NTK-parametrized BERT can reach a decent pretraining loss, but its hidden features are near-random, so fine-tuning a fresh head on them works badly. Wider can be *worse*

### Why SP doesn't blow up in practice

Let's reconcile: the theory says SP either diverges or goes lazy, and many LLMs are trained in SP without doing either.

- **At one width, we fit.** We tune the LR for the model we're training. By definition, it is stable.
- **Across widths, we refit.** SP never carries its LR to a wider model; it retunes, and the tuned LR falls with width (GPT-3: $`6 \times 10^{-4}`$ at 125M → $`0.6 \times 10^{-4}`$ at 175B). The left shift *is* the fix, applied by hand. The price is a new sweep at every scale (or a fitted HP scaling law standing in for one); μTransfer's point is to do that sweep once
- **At a finite width, "lazy" means throttled, not frozen.** Lazy is an infinite-width property (TP4 defines the kernel regime in the $`n \to \infty`$ limit). At a finite width, one LR set by the most-amplified tensor leaves the others below their own edge by a factor that grows with $`n`$ (≈ $`\sqrt n`$ for SGD's hidden layers when the readout sets the LR). They still move, just less than they could. That costs loss at a fixed width (SP models "underperform the same-width networks in [μP] even after tuning learning rate", TP5 Fig. 1).
- **In practice, the fitted LR escapes most of the throttling.** It falls slower than theory's $`1/n`$ (≈ $`n^{-1/2}`$ for SGD; [Haas et al., 2025](https://arxiv.org/abs/2505.22491)), past the readout's edge, so the hidden layers sit on theirs (SGD row: $`\Delta h = \eta \sqrt n = \Theta(1)`$) and learn features. Only the embedding stays throttled ($`\Delta h \propto \eta\, n^{-1/2} = n^{-1}`$). The logits pay: they grow with width, but under softmax cross-entropy $`\partial \mathcal{L} / \partial f = p - y`$ ($`p`$ the softmax probabilities, $`y`$ the one-hot target) is bounded, so training stays stable.
  - Theory gives the *maximal stable* exponent, not the optimum. Practice sits at the edge of stability, so whichever tensor's stability limit binds sets the exponent
  - Under Adam the optimum is ≈ $`1/n`$ for MLPs (the hidden layers bind). AdamW Transformers with trainable LayerNorm gains land at ≈ $`n^{-1/2}`$, dropping toward $`n^{-1}`$ if the gains are removed: the gains behave input-like and tolerate a hotter LR (Haas et al.)
- **Where refitting isn't enough: the logits.** Output-logit divergence and attention-logit growth still show up at scale ([Wortsman et al., 2023](https://arxiv.org/abs/2309.14322)), patched with z-loss (a penalty on the softmax normalizer), QK-norm (normalizing queries and keys) and logit soft-capping. These are hand fixes for the two places μP changes the scaling for (the readout init, and $`1/d_{head}`$)
- **Under Adam, the refit already does most of μP's job**: an LR tuned $`\propto 1/m`$ matches μP on the hidden layers and the output LR. What's left is the embedding LR (under-trained by $`m`$), the readout init and the attention scale. That's why "μP (simple)", which keeps fan-in init and takes only the $`1/n`$ LR, already transfers LR reasonably well ([Wortsman et al.](https://arxiv.org/abs/2309.14322))

## Exponents vs Constants

- μP classifies by shape, not function. $`W_q, W_k, W_v, W_o`$ and the MLP's up / gate / down are all $`n \to n`$, so **they get the same rule**.
- The only attention-specific rules are about the logit $`q^\top k`$, not the weights: $`1/d_{head}`$ and zero-init $`W_q`$. (One wrinkle: QK-norm gains are shared across heads, so if you widen by adding heads they see a growing sum and need their own rule, see [Complete(d)P](#completedp))
- So μP says nothing about QKV vs MLP *constants*, and empirically they differ. [Wang et al., 2025](https://arxiv.org/abs/2502.19002) find a **sharpness disparity** across block types that appears within the first ~2% of training and persists. Ordering blocks by sharpness (average Hessian diagonal per parameter), with Emb = embedding, QK = query/key projections, FFN = MLP, VO = value/output projections, Norm = norm gains: <div align="center">
  $`\displaystyle \text{Emb} \ll \text{QK} < \text{FFN} < \text{VO} \ll \text{Norm}`$ </div>
  - Raising each block's LR roughly in inverse proportion (Emb 10×, QK 8×, FFN 6×, VO 4×, all relative to Norm at the base LR) gives ≈2× faster pretraining. Note that the embedding gets the biggest boost, the same thing μP's table says
- Should you use different LRs per layer? 
  - This is an optional extra gain, and a hard search. With a μP-style parametrization we can do that search once on a small proxy and transfer it ([Complete(d)P](#completedp)); without one you redo it at every scale (Wang et al. retune the global LR per scale)

## μTransfer

- The recipe:
  1. Put the model family in μP relative to a base width (at the base width it's your ordinary SP model)
  2. Tune the master HPs (global $`\eta`$, init scale, multipliers, Adam betas, schedule) on a small proxy
  3. Widen, and copy them unchanged
- Headline ([TP5](https://arxiv.org/abs/2203.03466)): HPs tuned on a 40M proxy beat the published GPT-3 6.7B, for ~7% of one pretraining run's compute. Under μP, wider is uniformly better at fixed HPs
- Debug with a **coordinate check**: log the mean $`\lvert h \rvert`$ of every layer (and the logits) over the first few steps, at several widths. μP gives flat lines in width; any slope means some tensor is mis-scaled
- **Which parametrizations transfer?**
  - In theory, only μP ([§maximal](#desiderata-and-what-maximal-means)), plus everything that is μP in disguise: TP5's equivalent tables, [u-μP](https://arxiv.org/abs/2407.17465) (unit-scaled μP), and spectral μP with Muon ([below](#muon-and-the-spectral-view)). Not SP with a global LR (no stable limit), and not SP-stable or NTK (kernel limits)
  - In practice it's looser. [Everett et al., 2024](https://arxiv.org/abs/2407.05872) get transfer from SP, NTK and the mean-field parametrization too, once each tensor gets the right LR exponent, and SP with per-layer LRs sometimes beats μP (is that still "SP", though?). [Kosson et al., 2025](https://arxiv.org/abs/2510.19093) find μP's alignment assumptions only hold early in LLM training. After that, independent weight decay is what keeps update sizes width-invariant, so μP acts mostly like an implicit warmup
  - Takeaway: **in practice, transfer comes from two ingredients: the right width exponent on every tensor's LR, and independent weight decay** (decay the weights by λ directly, $`\theta \leftarrow (1-\lambda)\theta`$, instead of PyTorch AdamW's $`\eta\lambda`$, which shrinks whenever μP shrinks $`\eta`$)
- **Not a silver bullet**, because μP is only about width:
  - Fixing HPs to account for changes in depth, batch size, training horizon, etc. is what [Complete(d)P](#completedp) does; the rules are in [Scaling §2–3](scaling.md#2-scaling-width-and-depth)

## Muon and the spectral view

- μP restated in operator norms ([Yang, Simon & Bernstein, 2023](https://arxiv.org/abs/2310.17813)): for every layer, <div align="center">
  $`\displaystyle \|W_\ell\|_* = \Theta\!\left(\sqrt{n_\ell / n_{\ell-1}}\right) \quad \text{and} \quad \|\Delta W_\ell\|_* = \Theta\!\left(\sqrt{n_\ell / n_{\ell-1}}\right)`$ </div>
  - Sanity check against the table: an aligned rank-1 update of spectral norm 1 on an $`n \times n`$ matrix has $`\Theta(1/n)`$ entries, i.e. the hidden row
- Muon's update $`UV^\top`$ has every singular value equal to 1, so scaling it by $`\sqrt{\text{fan-out} / \text{fan-in}}`$ satisfies the update half *by construction*, with no per-layer $`1/m`$ ([Norms: Muon normalization schemes](norms.md#muon-normalization-schemes)). Per How To Scale, spectral-μP init + Muon transfers LR across width
- Muon only covers the hidden matrices. The embedding and LM head stay on AdamW with μP's rules ([Qwen3.8-Next](https://arxiv.org/abs/2608.30320) does exactly this split)

## What do popular models do?

- It's not typically easy to infer from an open-source config. A config only shows the forward pass (multipliers on embeddings, logits and residual branches, the attention scale), not per-tensor LRs or the real init (`initializer_range` is just what HF uses to init fresh modules). By the symmetry, μP can be written with no multipliers at all, so **multipliers in a config are evidence of μP, and their absence is evidence of nothing**

| Model | What the report says | Config fingerprint | Camp |
|---|---|---|---|
| [Cerebras-GPT](https://arxiv.org/abs/2304.03208) (2023) | Trains "+µP" variants alongside SP and reports better loss predictability | | μP |
| [MiniCPM](https://arxiv.org/abs/2404.06395) (2024) | "Tensor Program" width *and* depth scaling; *not* the attention scaling | `scale_emb: 12`; `dim_model_base: 256` (pre-logit hidden ÷ 2304/256 = 9 = $`m`$); `scale_depth: 1.4` (branches × 1.4/√L, L = layers) | μP + depth |
| [Granite 3.0](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) (IBM, 2024) | μP for "hyperparameter transfer after a hyperparameter search on smaller models", plus the [Power scheduler](https://arxiv.org/abs/2408.13359) for batch size and token count | `embedding_multiplier: 12`; `logits_scaling: 8`; `attention_multiplier` = 1/64 = $`1/d_{head}`$; `residual_multiplier: 0.22` | μP |
| [Command A](https://arxiv.org/abs/2504.00698) (Cohere, 2025) | "µP and µTransfer to tune hyper-parameters on smaller models and zero-shot transfer them"; sweeps per depth, since μP assumes fixed depth | | μP |
| [Apple AFM](https://arxiv.org/abs/2407.21075) (2024) | "a simplified version of µParam", like μP (simple), plus decoupled weight decay. It "stabilizes the optimal learning rate as model size increases", with "a slight left-shift" for much larger or deeper models | | μP (simple) |
| [DeepSeek LLM](https://arxiv.org/abs/2401.02954) → [V3](https://arxiv.org/abs/2412.19437) | "Scaling laws of hyperparameters": optimal LR and batch size fitted against compute. Init std 0.006 for every parameter, V1 through V3 | no multipliers | SP + HP scaling laws |
| [Qwen2.5](https://arxiv.org/abs/2412.15115) / [Qwen3](https://arxiv.org/abs/2505.09388) / [Qwen3.8](https://arxiv.org/abs/2608.30320) | Scaling laws for the optimal LR and batch size. Refit for Qwen3.8-Next after moving hidden matrices to Muon; the optimal LR still decays with size, just more slowly | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json): no multipliers, `head_dim**-0.5` (fixed `head_dim: 256`), QK-norm, `initializer_range: 0.02` | SP + HP scaling laws |

## Complete(d)P

[Mlodozeniec et al. (Apple), *Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration*](https://arxiv.org/abs/2512.22382), Dec 2025. Its parametrization, Complete(d)P, extends CompleteP (width + depth) to batch size and training duration. Two contributions:

- **One parametrization for four axes**
  - Width: μP plus fixes. AdamW $`\epsilon`$ is scaled to match the gradient's size. Weight decay is $`\times\, m`$ on hidden and output weights, so $`\eta\lambda`$ is width-invariant. QK-norm gains are treated as shared across heads. The LM-head multiplier is folded into init + LR (the symmetry again), so memory-efficient losses like Cut Cross-Entropy still work
  - Depth: CompleteP's residual branches $`\times\, m_L^{-\alpha}`$ ($`m_L`$ = depth ratio, $`\alpha \in [\frac12, 1]`$ a depth exponent), and both ends transfer
  - Batch and duration: treat AdamW as discretizing a stochastic differential equation (SDE), and hold the SDE fixed. Everything then depends only on the step count $`S = D / B`$ ($`D`$ training tokens, $`B`$ tokens per batch) (derivation in [Scaling §3](scaling.md#3-scaling-data-at-a-fixed-model)): <div align="center">
    $`\displaystyle \eta \propto S^{-1/2}, \quad \lambda \propto S^{-1/2}, \quad 1 - \beta_{1,2} \propto S^{-1}, \quad \epsilon \propto S^{1/2}`$ </div>
    - E.g. batch $`\times\,\kappa`$ at fixed data → $`\eta, \lambda \times \sqrt\kappa`$ (the weight-decay half is new). Tokens $`\times\,\kappa`$ at fixed batch → $`\eta \div \sqrt\kappa`$, which matches the observed decay of the optimal LR with horizon
- **Per-module HPs transfer too.** They tune LR, weight decay, $`\beta_1`$, $`\beta_2`$, $`\epsilon`$ and init scale per module type (13 types), plus per-depth multipliers (79 HPs in total), on a 50M-param / 1.6B-token proxy, then transfer: 2.3× speed-up at the proxy scale, 1.32× at 7.2B (~600× the FLOPs)
  - Most of the gain comes from different module types getting different LRs; per-depth multipliers add a bit more
