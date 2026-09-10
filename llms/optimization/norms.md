# Norms and Steepest Descent

Choosing an optimizer *is* choosing a norm to measure the weight update in (or equivalently, a norm to measure the output feature update in given another choice of input feature norm). Primary source: Bernstein & Newhouse, [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325). See also [Optimization](notes.md) (Muon), [μP](muP.md) — the width-scaling side of the same question — and [fundamentals/dl/04](../../fundamentals/dl/04_optimization_and_regularization/notes.md) for the classical second-order material this reframes.

## Induced operator norms

A weight matrix is a *map* from input features to output features, so the natural way to measure its size is by how much it can stretch that map — which means picking a norm on each side.

- **Definition - induced operator norm.** Given $`M \in \mathbb{R}^{d_\mathrm{out}\times d_\mathrm{in}}`$ and two normed vector spaces $`(\mathbb{R}^{d_\mathrm{in}}, \lVert\cdot\rVert_\alpha)`$ and $`(\mathbb{R}^{d_\mathrm{out}}, \lVert\cdot\rVert_\beta)`$, the "$`\alpha`$ to $`\beta`$" induced operator norm is

  ```math
  \lVert M\rVert_{\alpha\to\beta} \;=\; \max_{x \in \mathbb{R}^{d_\mathrm{in}}} \frac{\lVert Mx\rVert_\beta}{\lVert x\rVert_\alpha}
  ```

  - Read it as **worst-case amplification**: how much can $`M`$ blow up an input, measuring the input in $`\alpha`$ and the output in $`\beta`$?
  - **The key move**: varying $`\alpha`$ and $`\beta`$ induces a large family of matrix norms, and therefore (via steepest descent below) a correspondingly large family of optimizers. So *choosing an optimizer = making a claim about the geometry of the features.* 
- Practically, we typically this definition in the following context:
  - $x$ := input feature vector
  - $M = \Delta W$ := Single layer matrix weight update
  - $Mx$ := update in output feature vector
- Some open questions we'll cover later:
  - Does specifying a choice of norms for the update have implications on the final weights / features?
  - How should we choose our matrix norm, or equivalently, our input and output norms?

Here's a summary of various optimizers and what they mean:

| Norm on $`\Delta w`$ (flat vector) | Norm on $`\Delta W`$ (matrix) | Induced $`\alpha\to\beta`$ (in → out) | Unit ball for $`\Delta w`$ in coordinates | Unit ball for $`\Delta w`$ in singular values | Direction of $`\Delta w`$ | Optimizer |
|---|---|---|---|---|---|---|
| $`\ell_2`$ | Frobenius ($`S_2`$) | — | $l_2$ ball | $l_2$ ball | $`\mathbf{g} = U\Sigma V^{\top}`$ | normalized GD |
| $`\sqrt{\Delta w^{\top}H\,\Delta w}`$ | — | — | ellipsoid aligned to $`H`$ | — | $`H^{-1}\mathbf{g}`$ | [Newton](../../fundamentals/dl/04_optimization_and_regularization/notes.md#second-order-methods) |
| $`\sqrt{\Delta w^{\top}\mathrm{diag}(H)\,\Delta w}`$ | — | — | axis-aligned ellipsoid | — | $`\mathrm{diag}(H)^{-1}\mathbf{g}`$ | [diagonal preconditioning](../../fundamentals/dl/04_optimization_and_regularization/notes.md#second-order-methods) (Jacobi) |
| $`\ell_\infty`$ | max entry $`\max_{ij}\lvert\Delta W_{ij}\rvert`$ | $`\ell_1\to\ell_\infty`$ | **cube** | — | $\mathrm{sign}(\mathbf{g})  \;\approx\; \mathrm{diag}(H)^{-1/2}\,\mathbf{g}$ | signSGD, Adam as $`\beta_1,\beta_2,\epsilon\to0`$ |
| — | spectral ($`S_\infty`$) | $`\ell_2\to\ell_2`$ | — | **cube** | $`UV^{\top}`$ | [Muon](notes.md) w/o momentum, Shampoo w/o accumulation |
| — | Schatten-$`p`$, $`2<p<\infty`$ | — | — | between $l_2$ ball and cube | $`U\Sigma^{1/(p-1)}V^{\top}`$ | *approximated by* Shampoo **with** accumulation, SOAP |

- Adam as a "half-diagonal preconditioner":
  - Jacobi: $`P = \mathrm{diag}(H)`$ — *curvature*. 
  - Adam: $`P = \mathrm{diag}(\sqrt{v})`$ — *gradient magnitude*, not curvature. The two do connect, but only at half power: the Fisher identity gives $`\mathbb{E}[g_i^2]\approx H_{ii}`$, so $`\sqrt{v_i}\approx\sqrt{H_{ii}}`$ and

  ```math
  \mathrm{sign}(\mathbf{g}) \;\approx\; \mathrm{diag}(H)^{-1/2}\,\mathbf{g}
  ```
  - Note that this is without momentum
- **First order or second order?** 
  - Newton sits in the table as steepest descent under $`\sqrt{\Delta w^{\top}H\,\Delta w}`$ — a first-order step in a ball.
  - Or as a second-order step: jump to the minimum of the local quadratic $`\mathbf{g}^{\top}\Delta w + \tfrac12\Delta w^{\top}H\Delta w`$. **These are the same step.** The $`H`$-norm is the norm in which that quadratic is a *sphere* — substitute $`z = H^{1/2}\Delta w`$ and it becomes $`(H^{-1/2}\mathbf{g})^{\top}z + \tfrac12\lVert z\rVert_2^2`$, a round bowl. Plain GD in $`z`$ is Newton in $`\Delta w`$; the ellipsoid $`\Delta w^{\top}H\Delta w \le r^2`$ is just that bowl's level set. "Furthest downhill inside the ball" and "bottom of the quadratic" pick the same direction.
  - I therefore think that the question isn't whether we approximate the loss as first or second order, but rather what norm is "best", which awkwardly I don't have an answer for (although we do touch on this in [§Why an RMS→RMS Operator May Make Sense](#why-an-lrms--lrms-operator-may-make-sense), below)

## Sign Descent (Adam)

- The setup: **don't approximate $`H`$ at all.** Take a purely *linear* model of the loss, and stop it running away by penalizing the step in some norm $`\lVert\cdot\rVert`$:

  ```math
  \Delta w^* = \arg\min_{\Delta w}\left[\, \mathbf{g}^{\top}\Delta w + \frac{\lambda}{2}\lVert\Delta w\rVert^2 \,\right]
  ```

  - $`\lambda`$ is a scalar "sharpness", **chosen a priori**
- Proposition 1: the solution splits into a step size and a direction, where the step size is the **dual norm** $`\lVert\mathbf{g}\rVert^{\dagger} = \max_{\lVert t\rVert=1}\mathbf{g}^{\top}t`$:

  ```math
  \Delta w^* = -\frac{\lVert\mathbf{g}\rVert^{\dagger}}{\lambda} \cdot \arg\max_{\lVert t\rVert=1}\mathbf{g}^{\top}t
  ```

  - First factor = how far to step, second = which direction. Recall the dual of $`\ell_p`$ is $`\ell_q`$ with $`1/p+1/q=1`$, so $`\ell_2\to\ell_2`$ and $`\ell_\infty\to\ell_1`$.
- Now pick $`\lVert\cdot\rVert = \lVert\cdot\rVert_\infty`$. The dual norm is $`\lVert\mathbf{g}\rVert_1`$, and the maximizer over the unit cube is the sign vector. **Sign descent just falls out:**

  ```math
  \Delta w^* = -\frac{\lVert\mathbf{g}\rVert_1}{\lambda}\,\mathrm{sign}(\mathbf{g})
  ```

  - Note the implied learning rate — running sign descent at constant $`\eta`$ is assuming $`\lVert\mathbf{g}\rVert_1/\lambda`$ is roughly constant over training.
- Note that we've been dealing with a flattened weights vector $`w`$. This can be extended to multiple weight matrices over layers: For $`w`$ flattened from layers $`W_1,\dots,W_L`$:

  ```math
  \lVert w\rVert_\infty \;=\; \max_l \max_r \lVert \mathrm{row}_r(W_l)\rVert_\infty \;=\; \max_l \lVert W_l\rVert_{\ell_1\to\ell_\infty}
  ```
- The choice of sign descent then implies: 
  - An implicit choice of $l_\infty$ over the weight update
  - An implicit choice of norms ($l_\infty$ and $l_1$) over the output feature update and input features respectively

## Muon and Shampoo

### Muon without Momentum = Shampoo without Accumulation = Spectral Norm Steepest Descent

- **Muon** ([Jordan](https://kellerjordan.github.io/posts/muon/)) — momentum, orthogonalize, step:

  ```math
  m_t = \mu\,m_{t-1} + G_t, \qquad O_t = \mathrm{NewtonSchulz}(m_t) \approx UV^{\top}, \qquad W_t = W_{t-1} - \eta\,O_t
  ```

- **Shampoo** ([Gupta et al. 2018](https://arxiv.org/abs/1802.09568)) — accumulate two one-sided covariances, precondition by their inverse fourth roots:

  ```math
  L_t = L_{t-1} + G_tG_t^{\top}, \qquad R_t = R_{t-1} + G_t^{\top}G_t, \qquad W_t = W_{t-1} - \eta\,L_t^{-1/4}G_tR_t^{-1/4}
  ```

- **The identity.** Drop the accumulation, so $`L = GG^{\top}`$ and $`R = G^{\top}G`$ from the current gradient alone. With $`G = U\Sigma V^{\top}`$:

  ```math
  L^{-1/4} = U\Sigma^{-1/2}U^{\top}, \qquad R^{-1/4} = V\Sigma^{-1/2}V^{\top}
  ```

  ```math
  L^{-1/4}GR^{-1/4} = U\Sigma^{-1/2}U^{\top}\cdot U\Sigma V^{\top}\cdot V\Sigma^{-1/2}V^{\top} = U\Sigma^{0}V^{\top} = UV^{\top}
  ```

  - Drop the momentum from Muon ($`\mu = 0`$) and $`O_t = \mathrm{NS}(G)\approx UV^{\top}`$ as well. **Same matrix.**
 
  - Note that momentum and accumulation do **different jobs**. Muon's $`\mu`$ smooths *what gets orthogonalized* and leaves the direction pinned on the $`p=\infty`$ corner. Shampoo's accumulation changes *the preconditioner itself* and slides you off that corner → [§Shampoo with Accumulation](#shampoo-with-accumulation).

<a id="why-an-lrms--lrms-operator-may-make-sense"></a>
### Why an $l_{RMS} \rightarrow l_{RMS}$ Operator May Make Sense

- **The architecture already picked this norm.** RMSNorm makes input features unit-RMS by
  construction.
- **Per update**, $`\ell_{RMS}\to\ell_{RMS}`$ promises that for *every* input direction:

  ```math
  \lVert \Delta W x\rVert_{RMS} \;\le\; \eta\,\lVert x\rVert_{RMS}
  ```
  - This bounds the RMS of the output feature change, for a given input feature norm that synergizes with the default LLM normalization scheme.
  - It is plausible that over successive updates, the cumulative weights are "well-behaved", which we define as:

  ```math
  \sigma_i(W) \;=\; \Theta\!\left(\sqrt{\tfrac{\text{fan-out}}{\text{fan-in}}}\right) \quad \text{for all } i
  ```

    - This implies $`\lVert Wx\rVert_{RMS} \asymp \lVert x\rVert_{RMS}`$ for *every* $`x`$:
      - $`\lVert W_l\rVert_{RMS\to RMS}=\Theta(1)`$ → $`\sigma_{\max}`$ at the right scale, no output feature **explodes**
      - $`\kappa(W_l)=O(1)`$ → $`\sigma_{\min}`$ not far below it, no output feature **collapses**
  - Are final and/or intermediate $W$ matrices well-behaved? ([§Open Question](#open-questions))
    - A Muon update does $`UV^{\top}`$, which has every $`\sigma = 1`$, hence $`\kappa = 1`$ exactly; scaled, a Muon step *is* a well-behaved matrix of size $`\eta`$. 
    - $`W`$ becomes a running sum of perfectly-conditioned, correctly-scaled matrices. That is at least suggestive that intermediate and final $`W`$ **could** inherit the shape — in a way they have no reason to under Adam, whose steps are near-low-rank.
    - **Against**: a sum of flat-spectrum matrices needn't be flat. $`I`$ and $`e_1e_1^{\top}`$ are both flat on their supports, but $`I + e_1e_1^{\top}`$ has $`\sigma = (2,1,\dots,1)`$.
  - Consequence if it does hold: unit-RMS features stay unit-RMS at every depth, and a step of size $`\eta`$ moves each layer's output by $`O(\eta)`$ RMS — **independent of width**. [μP](muP.md) reconstructs this by hand with per-layer LRs; here it falls out of the norm.

### Muon Normalization Schemes

Write $`m = `$ fan-out, $`n = `$ fan-in, and $`Q = UV^{\top} = \mathrm{NewtonSchulz}(\cdot)`$. Every scheme is $`\Delta W = -\eta\,\alpha(m,n)\,Q`$ and differs **only in $`\alpha`$**: $`\alpha`$ normalizes $`Q`$ to unit "size" under each scheme's notion of size, and $`\eta`$ then sets the step. So $`\eta`$ appears nowhere below — each bullet is just *choose $`\alpha`$ so that $`\alpha Q`$ has unit stretch*.

| Scheme | $`\alpha`$ | What it controls |
|---|---|---|
| [Bernstein](https://jeremybernste.in/writing/deriving-muon) — RMS→RMS | $`\sqrt{m/n}`$ | **worst-case** RMS→RMS operator norm $`=1`$ |
| [Jordan](https://kellerjordan.github.io/posts/muon/) — clipped shape | $`\sqrt{\max(1,\,m/n)}`$ | **typical** feature change, *assuming isotropic activations* |
| [Moonshot](https://arxiv.org/abs/2502.16982) / Megatron | $`\sqrt{\max(m,n)}`$, $`\times\,0.2`$ | the update's **entries** have unit RMS |

- **Bernstein — worst case, distribution-free.** $`\lVert Q\rVert_{RMS\to RMS} = \sqrt{n/m}\,\sigma_{\max}(Q) = \sqrt{n/m}`$, so $`\alpha = \sqrt{m/n}`$ makes $`\lVert\alpha Q\rVert_{RMS\to RMS} = 1`$ **exactly, for any input** — no distributional assumption anywhere.
- **Jordan — average case, under isotropy.** Same question as Bernstein — *choose $`\alpha`$ so $`\alpha Q`$ has unit RMS stretch* — but **averaged over isotropic inputs** instead of maximized over all inputs. Take $`\mathbb{E}[xx^{\top}] = I_n`$, so $`\mathbb{E}\lVert x\rVert_{RMS}^2 = 1`$. Since $`\mathrm{tr}(QQ^{\top}) = \mathrm{rank}(Q) = \min(m,n)`$:

  ```math
  \mathbb{E}\lVert \alpha Qx\rVert_{RMS}^2 \;=\; \alpha^2\,\frac{\mathrm{tr}(QQ^{\top})}{m} \;=\; \alpha^2\,\frac{\min(m,n)}{m}
  ```

  Setting this to 1:

  ```math
  \alpha \;=\; \sqrt{\frac{m}{\min(m,n)}} \;=\; \sqrt{\max(1,\,m/n)}
  ```

  - The clip is just $`\min(m,n)`$ switching which dimension it picks:
  - **Expansion** ($`m>n`$): $`\min(m,n) = n`$, so $`\alpha = \sqrt{m/n}`$ — *same as Bernstein*.
  - **Contraction** ($`m\le n`$): $`\min(m,n) = m`$, so $`\alpha = 1`$. Bernstein's $`\sqrt{m/n} < 1`$ would make the *typical* stretch less than 1. Hence the clip.
  - **Why they disagree only on contraction.** A contraction $`Q`$ is a projector onto $`m`$ of the $`n`$ input directions ($`Q^{\top}Q`$ is a rank-$`m`$ projection). Its RMS stretch on an *individual* input is

  ```math
  \frac{\lVert Qx\rVert_{RMS,\,\text{out}}}{\lVert x\rVert_{RMS,\,\text{in}}} = \sqrt{\tfrac{n}{m}}\;\frac{\lVert Qx\rVert_2}{\lVert x\rVert_2} \;\in\; \left[\,0,\;\sqrt{n/m}\,\right]
  ```

  - $`0`$ if $`x\perp`$ the row space, $`\sqrt{n/m}`$ if $`x`$ lies entirely inside it. **Bernstein guards against the input that lives entirely in those $`m`$ directions; Jordan assumes a random input has only $`m/n`$ of its energy there.** Same $`Q`$, two different inputs in mind — which is why both make sense.
  - On an expansion every input direction is stretched by exactly $`\sigma=1`$, so worst case $`=`$ typical case and the two agree. 
- **Moonshot / Megatron default — unit-RMS entries.** $`\lVert Q\rVert_F^2 = \mathrm{rank}(Q) = \min(m,n)`$, so the RMS of $`Q`$'s entries is

  ```math
  \mathrm{RMS}(Q) = \frac{\lVert Q\rVert_F}{\sqrt{mn}} = \sqrt{\frac{\min(m,n)}{mn}} = \frac{1}{\sqrt{\max(m,n)}} \quad\Longrightarrow\quad \mathrm{RMS}\!\left(\sqrt{\max(m,n)}\,Q\right) = 1
  ```
  - Moonshot's stated justification for 0.2 is empirical - "From empirical observations, AdamW's update RMS is usually around 0.2 to 0.4".
  - Here's a derivation that explains the range. Write Adam's first-moment buffer as $`\bar g`$ (conventionally $`m`$ — renamed here since $`m`$ is fan-out above). For i.i.d. zero-mean gradients of variance $`\sigma^2`$:
    - $`\bar g_t = (1-\beta_1)\sum_{k\ge0}\beta_1^k\,g_{t-k}`$ is a weighted average with weights $`w_k = (1-\beta_1)\beta_1^k`$ summing to 1. Independent zero-mean terms → variances add: $`\mathrm{Var}(\bar g) = \sigma^2\sum_k w_k^2 = \sigma^2\frac{(1-\beta_1)^2}{1-\beta_1^2} = \sigma^2\frac{1-\beta_1}{1+\beta_1}`$.
    - $`v`$ averages $`g^2`$, each with mean $`\sigma^2`$, so $`\mathbb{E}[v]=\sigma^2`$ and $`\sqrt v\approx\sigma`$ (tight: $`\beta_2\approx0.999`$ averages ~2000 samples).
    - Per-coordinate update RMS is therefore

  ```math
  \mathrm{RMS}\!\left(\bar g/\sqrt v\right) \;\approx\; \frac{\sqrt{\mathrm{Var}(\bar g)}}{\sigma} \;=\; \sqrt{\tfrac{1-\beta_1}{1+\beta_1}} \;\overset{\beta_1=0.9}{=}\; 0.229
  ```
  - But I personally do think that matching weight update RMS is quite unprincipled.

### GD (p=2) vs Orthogonalized Update (p=∞)

- **Proof that $`\ell_2`$ in coordinate space $`=`$ $`\ell_2`$ in singular-value space.** One line, via the trace:

  ```math
  \sum_{ij} M_{ij}^2 \;=\; \mathrm{tr}(M^{\top}M) \;=\; \mathrm{tr}(V\Sigma^2V^{\top}) \;=\; \sum_k \sigma_k^2
  ```

  - So Frobenius is **simultaneously** $`\ell_2`$ on the $`mn`$ entries and $`\ell_2`$ on the $`r`$ singular values — the unique norm sitting in *both* unit-ball columns of the table. At $`p\ne2`$ the two worlds come apart entirely.
- **Note: GD does not enforce a bound.** Frobenius is not an induced operator norm, so there is no $`(\alpha,\beta)`$ for which bounding $`\lVert\Delta W\rVert_F`$ bounds $`\lVert\Delta Wx\rVert_\beta/\lVert x\rVert_\alpha`$. 
  - The only feature statement available is average-case: for $`x`$ uniform on the unit sphere, $`\mathbb{E}\lVert Mx\rVert_2^2 = \lVert M\rVert_F^2/n`$. **Spectral bounds the worst input; Frobenius bounds the typical one.**
  - Note that Frobenius *does* bound spectral, just loosely:

  ```math
  \sigma_{\max} \;\le\; \lVert M\rVert_F \;\le\; \sqrt{r}\,\sigma_{\max}
  ```
  - The difference now is that a Frobenius budget of $`\eta`$ permits the **entire** update to land in one direction — and per Jordan Keller's observation, deep-net updates *are* empirically near-low-rank.

### Shampoo with Accumulation

- We noted that Shampoo without Accumulation is spectral descent in the spectral norm ($p = \infty$). We also noted that vanilla GD corresponds to $p = 2$. Here, we provide some intuition for how shampoo with accumulation picks a $p \in [2, \infty]$. 
- Consider the Shampoo equations: 

  ```math
  L_t = L_{t-1} + G_tG_t^{\top}, \qquad R_t = R_{t-1} + G_t^{\top}G_t, \qquad W_t = W_{t-1} - \eta\,L_t^{-1/4}G_tR_t^{-1/4}
  ```

- Without accumulation, we have:

  ```math
  L^{-1/4} = U\Sigma^{-1/2}U^{\top}, \qquad R^{-1/4} = V\Sigma^{-1/2}V^{\top}
  ```

  ```math
  L^{-1/4}GR^{-1/4} = U\Sigma^{-1/2}U^{\top}\cdot U\Sigma V^{\top}\cdot V\Sigma^{-1/2}V^{\top} = U\Sigma^{0}V^{\top} = UV^{\top}
  ```

- With accumulation, we rewrite: 

  ```math
  L^{-1/4} = U\Sigma_L^{-1/2}U^{\top}, \qquad R^{-1/4} = V\Sigma_R^{-1/2}V^{\top}
  ```

  ```math
  L^{-1/4}GR^{-1/4} = U\Sigma_L^{-1/2}U^{\top}\cdot U\Sigma V^{\top}\cdot V\Sigma_R^{-1/2}V^{\top} = U(\Sigma_L^{-1/2}\Sigma\Sigma_R^{-1/2})V^{\top}
  ```

- Redefine $`C := \Sigma_L^{-1/2}\Sigma\,\Sigma_R^{-1/2}`$, such that the update is just $`UCV^{\top}`$. Both effects now live in one $`m\times n`$ matrix:
  - $`\mathrm{diag}(C)`$ → **rescaling**. This is the effective $`p`$.
  - off-diagonal $`C`$ → **rotation** of the singular directions. Hence why "effective $`p`$" is approximate: if $`C = PDQ^{\top}`$ then the product's SVD is $`(UP)\,D\,(VQ)^{\top}`$, so the left basis is $`UP`$, not $`U`$.
- **What sets the effective $`p`$.** On the diagonal, $`C_{ii} = \sigma_i/(\sigma_{L,i}\sigma_{R,i})^{1/2}`$. If $`\sigma_{L,i}\propto\sigma_i^{\theta}`$, then $`C_{ii}\propto\sigma_i^{1-\theta}`$. Matching against the Schatten-$`p`$ exponent $`\sigma^{1/(p-1)}`$ then gives

  ```math
  p \;=\; 1 + \frac{1}{1-\theta} \qquad\qquad \theta=1 \Rightarrow p=\infty, \qquad \theta=0 \Rightarrow p=2
  ```

  - **Effective $p$ is determined by how strongly $`L`$'s spectrum tracks the current gradient's.** Stated more plainly, singular vectors of $G$ that are strongly/weakly represented in the history will be dampened similarly to the $p = \infty/2$ case.  
- **What the off-diagonals do.** $L$'s spectrum contains the history of $G_t$'s left singular vectors. As a result, the current update is **tilted away from historically-active directions and toward quiet ones — even when the current $`G_t`$ doesn't point there.** 
- Above we showed that shampoo with accumulation will have an effective $p < \infty$. What does Muon with momentum do? Note that even with momentum, Muon's updates are always $p=\infty$. One possible view on the function of momentum is that it helps to "cancel out" the noise of a singular update, which would otherwise be "amplified" to singular value 1.

## Open Questions

- Empirically, [2](https://arxiv.org/pdf/2605.06654) finds that a network pretrained with Muon has **denser activations** (input *and* output, metric $`\lVert x\rVert_1/(\sqrt d\lVert x\rVert_2)`$, Fig. 4) and **higher weight stable rank** (GPT-2: 114.7 vs 65.8, Table 2) than one pretrained with AdamW; [1](https://arxiv.org/pdf/2605.10468) independently confirms the stable-rank part.
  - **What's missing is a mechanism connecting steps to levels.** A per-step bound controls the *rate of change*, not the *level*: the triangle inequality gives only $`\lVert W_T\rVert \le \lVert W_0\rVert + T\eta`$.
  - But with weight decay $`\lambda`$ and a normalized update ($`\lVert O_t\rVert = 1`$):

  ```math
  W \leftarrow (1-\lambda\eta)W - \eta\,O_t \;\Longrightarrow\; \lVert W_{t+1}\rVert \le (1-\lambda\eta)\lVert W_t\rVert + \eta \;\Longrightarrow\; \limsup_t \lVert W_t\rVert \le \tfrac{1}{\lambda}
  ```

    - This may be why [Moonshot](https://arxiv.org/abs/2502.16982) finds weight decay *essential* for Muon at scale.
    - But note that this does not admit a lower bound to the smallest singular value.
- But if we assume this, then this may explain why using the same optimizer as pretraining leads to the best learning-forgetting tradeoff ([2](https://arxiv.org/pdf/2605.06654)). 
  - The paper assumes that forgetting ≈ $`\tfrac12\mathbb{E}\lVert\Delta Wx\rVert_2^2`$ where x denotes the input activation of pretraining data at $\theta_0$ and the expectation is taken over $x$.
  - To handwave the argument in the paper, suppose we pretrained with Adam. Then our $x$ will have sparse activations. Continuing to train with Adam will lead to weight updates "thinking" in the $l_1$ input norm space, but training with Muon will lead to weight updates "thinking" in the $l_2$ input norm space. This ultimately reduces forgetting for a given learning target.
    - More concretely, **why does $`1\to\infty`$ approximate $`1\to2`$ better than $`2\to2`$ does?** Neither Adam nor Muon *is* $`\mathcal{A}_{1,2}`$ — both are the wrong optimizer for the metric. The question is why Adam's wrongness is cheaper.
      - **Structure vs ruler.** Every $`1\to\beta`$ norm is *max over columns* of (the $`\beta`$-norm of that column). So Adam's $`1\to\infty`$ and the metric's $`1\to2`$ agree on **what** is constrained — the worst column — and differ only in **how a column is measured**, $`\ell_\infty`$ vs $`\ell_2`$ inside it. $`2\to2`$ constrains singular values, which mix columns: a different structure entirely.