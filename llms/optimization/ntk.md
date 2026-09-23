# Neural Tangent Kernels

What gradient descent does to a network's *output* rather than its weights, and the infinite-width limit in which that becomes a plain kernel method. Primary source: Jacot, Gabriel & Hongler, [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/pdf/1806.07572). See also [μP](muP.md) (the parametrization built to *escape* this regime), and for the kernel background [Gaussian Process](../../fundamentals/classical/13_gaussian_process/notes.md) and the [kernel trick](../../fundamentals/classical/08_svms/notes.md#kernel-trick).

(1) For *any* differentiable network, parameter-space GD is *exactly* kernel GD in function space, with a kernel that moves over time.

(2) At infinite width, in the right parametrization, that kernel stops moving. This has interesting implications. 

## Setup

- Terminology:
  - $`f_\theta(x) \in \mathbb{R}`$ : scalar-output network with parameters $`\theta \in \mathbb{R}^P`$ (vector outputs work the same way — see the remark at the end of [Parameter GD is kernel GD](#parameter-gd-is-kernel-gd))
  - $`X = (x_1, \ldots, x_N)`$, $`y \in \mathbb{R}^N`$ : training inputs and targets
  - $`f_\theta(X) \in \mathbb{R}^N`$ : the vector of training predictions $`(f_\theta(x_1), \ldots, f_\theta(x_N))^\top`$. We'll be doing linear algebra in this **sample space** a lot
  - $`L(\theta) = L(f_\theta(X))`$ : the loss, a function of the predictions only
  - Gradient flow: $`\dot\theta_t = -\nabla_\theta L(\theta_t)`$, i.e. GD with the learning rate → 0. Continuous time keeps the algebra clean; the discrete version is in [Squared loss](#squared-loss-closed-form-eigenvectors-convergence)
- Only two assumptions for the first half: $`f_\theta`$ is differentiable in $`\theta`$, and we train by gradient flow. No width, no architecture, no init distribution yet

## Tangent features and the kernel

- Tangent feature of an input: $`\phi_t(x) = \nabla_\theta f_{\theta_t}(x) \in \mathbb{R}^P`$ — one entry per parameter, "how much would nudging $`\theta_p`$ move the prediction at $`x`$"
- The (empirical) NTK is just the Gram kernel of these features: <div align="center">
  $`\displaystyle K_t(x, x') = \phi_t(x)^\top \phi_t(x') = \sum_{p=1}^P \frac{\partial f_{\theta_t}(x)}{\partial \theta_p} \frac{\partial f_{\theta_t}(x')}{\partial \theta_p}`$ </div>
- Intuition: $`x`$ and $`x'`$ are "similar" under $`K_t`$ if parameter changes move their predictions in the same direction. It's a similarity in *what-training-does-to-you* space.
- Being a Gram matrix, it's PSD, and since the features are *gradients*, it's literally the kernel of the linear model $`\theta \mapsto \phi_t(x)^\top \theta`$ — which is where the linearization story below comes from

## Parameter GD is kernel GD

- Let $`J_t = \partial f_{\theta_t}(X) / \partial\theta \in \mathbb{R}^{N \times P}`$ be the Jacobian of the training predictions; row $`i`$ is $`\phi_t(x_i)^\top`$
- Chain rule twice:
  1. How the predictions move: $`\dot f_t(X) = J_t \dot\theta_t`$
  2. What the gradient is: $`\nabla_\theta L = J_t^\top \nabla_f L(f_t)`$, where $`\nabla_f L \in \mathbb{R}^N`$ is the per-sample loss derivative (the residual, for squared loss)
  3. Plug gradient flow in: $`\dot f_t(X) = -J_t J_t^\top \nabla_f L(f_t) = -K_t \nabla_f L(f_t)`$, with $`K_t = J_t J_t^\top \in \mathbb{R}^{N \times N}`$ and $`(K_t)_{ij} = K_t(x_i, x_j)`$
- **Important intuition**: GD in parameter space is GD in function space *preconditioned by $`K_t`$*. Plain function-space GD would be $`\dot f = -\nabla_f L`$, i.e. move each prediction independently toward its target. The network can't do that — it can only move along directions its parameters allow, and $`K_t`$ encodes which coordinated patterns of predictions are cheap or expensive to move
- This is **exact for any differentiable network at any width**. But note that $`J_t`$ and hence $`K_t`$ changes over time.

## The infinite-width limit

Source: [Jacot et al.](https://arxiv.org/pdf/1806.07572), Theorems 1 and 2

- The paper's setup, the "NTK parametrization":
  - Fully connected, depth $`L`$ fixed, hidden widths $`n_1, \ldots, n_{L-1} \to \infty`$ (sequentially)
  - $`\tilde\alpha^{(\ell+1)} = \frac{1}{\sqrt{n_\ell}} W^{(\ell)} \alpha^{(\ell)} + \beta b^{(\ell)}`$, with $`W^{(\ell)}_{ij}, b^{(\ell)}_j \sim \mathcal{N}(0,1)`$. Note the explicit $`1/\sqrt{n_\ell}`$ *forward multiplier*, rather than a $`1/n_\ell`$ init variance — same function at init, different training dynamics. That is a significant difference vs [μP §abc parametrization](muP.md#abc-parametrization)
  - Lipschitz, twice-differentiable nonlinearity $`\sigma`$; finite dataset; gradient flow on a finite horizon $`[0, T]`$
- Two results:
  1. **At init**, $`K_0 \to \Theta_\infty`$, a *deterministic* kernel (Theorem 1). Intuition: each entry of $`K_0`$ is an average over $`n`$ random hidden units → law of large numbers
  2. **During training**, $`K_t \to \Theta_\infty`$ for all $`t \in [0, T]`$ (Theorem 2). The kernel doesn't move
- So $`K_t \approx K_0 \approx \Theta_\infty`$, and the exact identity above becomes a *linear ODE with a constant coefficient*:
  - $`\dot f_t(X) = -\Theta_\infty(X, X)\, \nabla_f L(f_t)`$
  - A nonlinear model in parameter space → a linear dynamical system in function space
- Why the kernel doesn't move — the 1-hidden-layer picture, $`f = \frac{1}{\sqrt n}\sum_i a_i \sigma(w_i^\top x)`$:
  - Terminology
    - $`f \in \mathbb{R}, W \in \mathbb{R}^{n \times d}`$ with rows $`w_i`$, $`a \in \mathbb{R}^n`$. 
    - Init $`a_i \sim \mathcal{N}(0, 1)`$, $`w_{ij} \sim \mathcal{N}(0, 1)`$ — unit variance, *not* fan-in. 
    - Same $`O(1)`$ learning rate $`\eta`$ on $`a`$ and $`W`$
    - The **output** is $`f(x) \in \mathbb{R}`$. The **features** are the $`n`$ hidden-layer outputs $`\sigma(w_i^\top x)`$. 
  - Why every weight parameter moves by $`O(n^{-1/2})`$ — write the update out, with residuals $`r_j = f(x_j) - y_j`$ and mean squared loss:
    - $`\partial f / \partial a_i = \frac{1}{\sqrt n}\, \sigma(w_i^\top x)`$ — the $`1/\sqrt n`$ is the forward multiplier; $`\sigma(w_i^\top x) = O(1)`$ in $n$ because $`w_i`$ has $`O(1)`$ entries and $`d`$ is fixed
    - $`\partial f / \partial w_i = \frac{1}{\sqrt n}\, a_i\, \sigma'(w_i^\top x)\, x`$ — same $`1/\sqrt n`$, times $`a_i = O(1)`$
    - So with $`\eta = O(1)`$: $`\Delta a_i, \Delta w_i = O(n^{-1/2})`$
    - Feature change: $`\sigma(w_i^\top x + \Delta w_i^\top x) - \sigma(w_i^\top x) \approx \sigma'(w_i^\top x)\, \Delta w_i^\top x`$, and $`\Delta w_i^\top x`$ is $`d`$ terms each $`O(n^{-1/2})`$. So $`O(n^{-1/2}) \to 0`$: as $`n \to \infty`$, features stay frozen.
    - Frozen features ⇒ frozen $`\phi_t`$ ⇒ frozen $`K_t`$
    - Contrast SP, $`f = \sum_i a_i \sigma(w_i^\top x)`$ with $`a_i \sim \mathcal{N}(0, 1/n)`$ — the $`1/\sqrt n`$ is now *inside the value* of $`a_i`$. Same $`f_0 = O(1)`$ by the same CLT, but:
      - $`\partial f/\partial a_i = \sigma(w_i^\top x), \Delta a_i = O(\eta)`$, i.e. $`\sqrt n`$ times larger than $`a_i`$'s own init size
      - The output change from that step: <div align="center">
        $`\displaystyle \Delta f(x) = \sum_i \sigma(w_i^\top x)\, \Delta a_i = -\eta \cdot \frac1N \sum_j r_j \underbrace{\sum_{i=1}^n \sigma(w_i^\top x)\, \sigma(w_i^\top x_j)}_{n \text{ terms, nonzero mean}}`$ </div>
      - So $`\Delta f = O(\eta n)`$. Stability forces $`\eta = O(1/n)`$, and at that LR $`\Delta w_i = O(\eta\, a_i) = O(n^{-3/2})`$: features even more frozen than under NTKP ([μP §SP](muP.md#sp-sp-stable-and-ntk-what-goes-wrong))
      - **Init variance $`1/n`$ and a forward multiplier $`1/\sqrt n`$ give the same function at init but different gradients, because the multiplier is in the formula and the variance isn't.** That is the NTK parametrization's trick: move the $`1/\sqrt n`$ into the forward pass so it shows up in every gradient
  - Why the output still moves by $`O(1)`$: neuron $`i`$'s contribution to $`f`$ changes by $`\frac{1}{\sqrt n} \cdot O(n^{-1/2}) = O(1/n)`$, there are $`n`$ of them, and they're **coherent** (each change reduces the same residuals, so they add rather than cancel): $`n \cdot O(1/n) = O(1)`$

## Linearization around init

- First-order Taylor in $`\theta`$: $`f_\theta(x) \approx f_0(x) + \phi_0(x)^\top (\theta - \theta_0)`$
- This is a *linear* model in $`\theta`$ with fixed features $`\phi_0`$, and its kernel is exactly $`K_0`$. In the infinite-width limit the real network has the same dynamics as this linearized one.
- So training in this regime = fitting a linear model on the tangent features present at init.
- Confusion worth clearing up: this is **not** "only the last layer trains". Every layer moves — none of them moves enough to change its own features

## Squared loss: closed form, eigenvectors, convergence

- $`L = \frac12 \|f(X) - y\|^2`$, so $`\nabla_f L = f - y`$ and $`\dot f_t = -K (f_t - y)`$ with $`K = \Theta_\infty(X, X)`$ constant
- Constant-coefficient linear ODE → closed form: <div align="center">
  $`\displaystyle f_t(X) = y + e^{-Kt}\bigl(f_0(X) - y\bigr)`$ </div>
- Diagonalize $`K = U \Lambda U^\top`$ and project the residuals onto each eigenvector:
  - $`u_j^\top (f_t - y) = e^{-\lambda_j t}\, u_j^\top (f_0 - y)`$
  - large $`\lambda_j`$ → that residual component decays fast; small $`\lambda_j`$ → slowly (time scale $`1/\lambda_j`$); $`\lambda_j = 0`$ → never (the null space of $`K`$ is unreachable)

### What "learning an eigenvector" means

The eigenvectors live in $`\mathbb{R}^N`$ — **sample space**, one coordinate per training example. So "learning the $`u_j`$ component" means correcting a particular *pattern* of predictions across the training set, not fitting one example.

- The network is learning the target vector $`y = (y_1, \ldots, y_N)^\top`$, but not each $`y_i`$ independently: any parameter change moves many predictions at once, so what it can cheaply learn are coordinated patterns of labels and predictions
- One possible eigenvector (pattern to learn) might look like:
  - $`q_1 = \frac12 (1, 1, -1, -1)^\top`$ — raise predictions on samples 1 and 2, lower them on 3 and 4.
- So the network learns roughly in this order:
  1. the overall average / bias
  2. differences between broad groups
  3. fine distinctions between very similar examples

## Early stopping as a spectral filter

- After training for time $`t`$, the fraction of the initial residual along $`u_j`$ that has been fitted is $`1 - e^{-\lambda_j t}`$: ≈ 1 for $`\lambda_j \gg 1/t`$, ≈ 0 for $`\lambda_j \ll 1/t`$
- So stopping at $`t`$ is a soft low-pass filter over the kernel's spectrum: keep the patterns the kernel considers important, suppress the fine-grained ones (noise). 

## Off the training set: kernel regression

- Let $`k(x, X) = [K(x, x_1), \ldots, K(x, x_N)] \in \mathbb{R}^{1 \times N}`$. The same ODE holds for $`f_t(x)`$ at any $`x`$ (with $`k(x, X)`$ in place of $`K`$), and integrating to $`t \to \infty`$ with $`K`$ invertible gives <div align="center">
  $`\displaystyle f_\infty(x) = f_0(x) + k(x, X)\, K^{-1} \bigl(y - f_0(X)\bigr)`$ </div>
- With $`f_0 \equiv 0`$ this is $`f_\infty(x) = k(x, X) K^{-1} y`$ — ridgeless kernel regression, i.e. the GP posterior mean with $`\Theta_\infty`$ as the prior covariance (the same formula as in [Gaussian Process](../../fundamentals/classical/13_gaussian_process/notes.md), with zero noise). 
- The paper goes a step further: at infinite width $`f_0`$ is itself Gaussian, so the trained $`f_\infty`$ is a Gaussian process whose mean is the kernel-regression solution and whose variance vanishes on the training points
- So the NTK governs both halves of learning: *how fast* the training data are fitted (the spectrum) and *how* those fitted values extend to unseen inputs (the $`k(x, X) K^{-1}`$ interpolation).

## What this says about feature learning

- "Feature learning" in the ordinary sense = training changes the internal representation so the features suit the task. In the NTK regime, features are frozen and training instead re-weights features that were already there at init. This is **lazy training**.
- At finite width $`K_t \neq K_0`$, and note that the kernel drifting during training is a *symptom* of features adapting.
- Takeaway: **the NTK limit describes a regime where the network learns a function but doesn't learn new features.** The NTK parametrization is a *choice* (a specific $`1/\sqrt n`$ multiplier + $`O(1)`$ LR), and [μP](muP.md) is the alternative choice under which the feature kernel *does* move by $`O(1)`$ at infinite width.