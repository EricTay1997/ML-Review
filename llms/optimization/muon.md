# Muon

Muon (**M**oment**U**m **O**rthogonalized by **N**ewton–Schulz) is an optimization algorithm. Primary source: Keller Jordan, [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (Dec 2024). See also:
- [Norms §Muon and Shampoo](norms.md#muon-and-shampoo): *why* orthogonalize — steepest descent under the spectral norm, the Shampoo-without-accumulation identity, what momentum does, the per-matrix normalization schemes, and what Muon-trained weights look like
- [μP §Muon and the spectral view](muP.md#muon-and-the-spectral-view) and [Scaling §Beyond the table](scaling.md#beyond-the-table): why Muon's learning rate transfers across width
- [Scaling §Critical batch size](scaling.md#critical-batch-size): Muon tolerates larger batches than AdamW

## Definition

- For a weight matrix $`W`$ with gradient $`G_t`$, momentum coefficient $`\mu`$ and learning rate $`\eta`$: 
  - Accumulate $`m_t = \mu\,m_{t-1} + G_t`$, 
  - Then step $`W_t = W_{t-1} - \eta\,\mathrm{NewtonSchulz5}(m_t)`$. 
  - Per-matrix scale factors go on top ([Norms §Muon normalization schemes](norms.md#muon-normalization-schemes)).
- $`\mathrm{NewtonSchulz5}`$ ([below](#newton-schulz)) approximates <div align="center">
  $`\displaystyle \mathrm{Ortho}(G) = \arg\min_O \left\{ \lVert O - G\rVert_F \;:\; O^\top O = I \text{ or } OO^\top = I \right\} = UV^\top`$ </div>
  - i.e. the **nearest semi-orthogonal matrix** in Frobenius norm. With the singular value decomposition (SVD) $`G = USV^\top`$: keep the singular vectors, set every singular value to 1. (Equivalently, the orthogonal factor of the polar decomposition $`G = (UV^\top)(VSV^\top)`$.)
- Nesterov-style momentum (orthogonalize $`G_t + \mu\,m_t`$ instead of $`m_t`$) works a bit better in every case tested, so it's the default. Purely empirical.
- Why orthogonalize? "divine benevolence" (haha), but there are other views:
  - Theory: [Norms §Muon and Shampoo](norms.md#muon-and-shampoo).
  - Empirics: by inspection, SGD-momentum and Adam updates for transformer weight matrices have **very high condition number** — nearly low-rank, with every neuron's update dominated by a few directions. The speculation: orthogonalizing scales up "rare directions" that are small in the update but still important for learning.

## Which parameters get Muon

- **Muon: the 2D hidden-layer matrices.** Conv filters too, by flattening the last three dimensions (out-channels × in-channels·kernel height·kernel width).
- **AdamW (or another standard optimizer) for everything else:**
  - Scalars and vectors (norm gains, biases): nothing to orthogonalize.
  - **The embedding and the LM head, even though they're 2D** — needed for the best transformer results. The μP side of the same split is in [μP §Muon and the spectral view](muP.md#muon-and-the-spectral-view).
    - Embedding: follows from modular-norm theory ([Large et al., 2024](https://arxiv.org/abs/2405.14813)). My read: its input is one-hot, so $`\Delta W x`$ is a single column of $`\Delta W`$, and the natural norm is the max column norm ($`\ell_1\to\ell_2`$), whose steepest-descent step normalizes each column on its own ([Norms](norms.md#induced-operator-norms), second table) instead of orthogonalizing the whole matrix.
    - LM head: purely empirical. The post says it does *not* follow from the theory.
- **Q, K and V as three matrices, not one fused QKV matrix** (Vlado Boza's experiment).

## Newton-Schulz

- Newton–Schulz (NS) pushes every singular value toward 1 **without ever forming an SVD** — 5 steps of a tuned quintic, all matmuls, so it is cheap and GPU-friendly:

  ```python
  def newtonschulz5(G, steps=5, eps=1e-7):
      assert G.ndim == 2
      a, b, c = (3.4445, -4.7750, 2.0315)
      X = G.bfloat16()
      X /= (X.norm() + eps)
      if G.size(0) > G.size(1):
          X = X.T
      for _ in range(steps):
          A = X @ X.T
          B = b * A + c * A @ A
          X = a * X + B @ X
      if G.size(0) > G.size(1):
          X = X.T
      return X
  ```

  - `X.norm()` is Frobenius. The transpose makes `X` wide, so the Gram matrix `A` has the short side's size ([§Cost](#cost)). Everything runs in bfloat16.

## Cost

- Memory: same as SGD-momentum — one buffer per matrix, vs AdamW's two.
- FLOPs, for a matrix whose short side is $`m`$ and long side is $`n`$ (the post's convention, not the fan-out/fan-in $`m, n`$ of [Norms](norms.md#muon-normalization-schemes)). After the transpose `X` is $`m\times n`$, and one NS step costs:
  1. `A = X @ X.T`: $`2nm^2`$
  2. `A @ A`: $`2m^3`$
  3. `B @ X`: $`2nm^2`$
  - → $`4nm^2 + 2m^3`$ per step, at most $`6nm^2`$ (square). Building `B` from `A` first is the trick: the naive $`aX + b(AX) + c\,A(AX)`$ does three $`n`$-sized matmuls, $`6nm^2`$ whatever the shape (improvement found by Jeremy Bernstein, Jiacheng You and Franz Cesista).
  - $`T`$ steps (typically 5) → at most $`6Tnm^2`$ on top of SGD.
- A forward + backward pass through the same linear layer costs $`6nmB`$, with $`B`$ the tokens through it per step. So <div align="center">
  $`\displaystyle \text{overhead} \;\le\; \frac{6Tnm^2}{6nmB} \;=\; \frac{Tm}{B}`$ </div>
  - NanoGPT speedrun (a GPT-2-small-sized race, [§Evidence](#evidence)): $`m = 768`$, $`B = 524{,}288`$ → $`5\cdot768/524288 \approx 0.7\%`$
  - [Llama 3](https://arxiv.org/abs/2407.21783) 405B: $`m = 16384`$, $`B \approx 16`$M → $`\approx 0.5\%`$
  - **Under 1% at both ends, because NS runs once per step while the layer's matmuls run once per token** — the overhead falls as the batch grows.
- That's FLOPs, not wallclock. Muon is still slower per step than AdamW, and NS needs each full matrix, which sharded training (see [Parallelism](../performance/parallelism.md)) splits across GPUs.
