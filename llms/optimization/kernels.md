# Kernels

The minimum kernel background for [NTK](ntk.md): what a kernel is, why the two definitions in [SVMs §Kernel Trick](../../fundamentals/classical/08_svms/notes.md#kernel-trick) and [Gaussian Process](../../fundamentals/classical/13_gaussian_process/notes.md) are the same object, and why every kernel method's prediction has the same shape. The function-space side — Hilbert spaces, the reproducing property, the RKHS norm — is in [RKHS](rkhs.md). References: Hofmann, Schölkopf & Smola, [Kernel Methods in Machine Learning](https://arxiv.org/abs/math/0701907); Rasmussen & Williams, [GPML](http://gaussianprocess.org/gpml/) ch. 6 for the kernel-regression ↔ GP identity.

## Notation

- $`\mathcal{X}`$ : the input space 
- $`\phi : \mathcal{X} \to \mathbb{R}^P`$ : the **feature map**. 
- $`x' \in \mathcal{X}`$ : a *second* input. 
- $`k(x, x') \in \mathbb{R}`$ : the **kernel**. 
- $`K \in \mathbb{R}^{N \times N}`$, $`K_{ij} = k(x_i, x_j)`$ : the **Gram matrix** — every pairwise kernel value on the training set
- $`\alpha \in \mathbb{R}^N`$ : one learned coefficient per *training point*. This is what a kernel method actually fits

## The Kernel Trick

- A linear model $`g(x) = x^\top w`$ is easy to fit (convex) but can only draw straight lines
- Fix: transform first, then be linear in *that* — $`g(x) = \phi(x)^\top w`$. Nonlinear in $`x`$, still linear in $`w`$. E.g. for $`x = (x_1, x_2)`$:
  - $`\phi(x) = (x_1^2,\ \sqrt2\, x_1 x_2,\ x_2^2)`$, so $`\phi : \mathbb{R}^2 \to \mathbb{R}^3`$, and "linear in $`\phi`$" now covers every ellipse and hyperbola
- Problem: the richer $`\phi`$, the bigger $`P`$. Degree-3 polynomial features on a 100-dim input → $`P = 171{,}700`$. The Gaussian/RBF feature map has $`P = \infty`$ — you can't write $`\phi(x)`$ down at all
- The trick: We define a kernel $`k(x, x') = \phi(x)^\top \phi(x')`$ for *some* feature map $`\phi`$, that we can evaluate *without building* $`\phi`$. 
- $`k(x, x')`$ is "how similar are $`x`$ and $`x'`$, measured in feature space"
- In doing so, we often can express predictions as a function of the kernel, and convert computational cost into a function of $N$, despite the very expressive $\phi$. 

## Which functions are kernels? (reconciling the SVM and GP notes)

- Let's reconcile something. The [SVM note](../../fundamentals/classical/08_svms/notes.md#kernel-trick) defines a kernel as an *inner product after a transformation*; the [GP note](../../fundamentals/classical/13_gaussian_process/notes.md) treats it as a *covariance / similarity function* you pick by hand. 
- Easy direction: if $`k = \phi^\top \phi`$, stack the feature vectors as rows of $`\Phi`$ so $`K = \Phi \Phi^\top`$, and $`u^\top K u = \|\Phi^\top u\|^2 \geq 0`$ automatically
- **Mercer / Moore–Aronszajn** give the converse: any symmetric PSD function is *already* $`\phi(x)^\top \phi(x')`$ for some $`\phi`$, possibly infinite-dimensional, whether or not you can name it
- Takeaway: **design either end and get the other free.** 

## Representer theorem: why every prediction is a weighted sum of similarities

- Fit $`g(x) = \phi(x)^\top w`$ by minimising any loss that depends on $`w`$ only through the training predictions, plus a penalty $`\|w\|^2`$
- Split $`w = w_\parallel + w_\perp`$, with $`w_\perp`$ orthogonal to every $`\phi(x_i)`$. Then $`\phi(x_i)^\top w = \phi(x_i)^\top w_\parallel`$ for all $`i`$ — the perpendicular part changes no training prediction, so it can't lower the loss, but it does add $`\|w_\perp\|^2`$ to the penalty. So the optimum has $`w_\perp = 0`$: <div align="center">
  $`\displaystyle w^\star = \sum_i \alpha_i\, \phi(x_i) \qquad\Rightarrow\qquad g(x) = \phi(x)^\top w^\star = \sum_i \alpha_i\, \phi(x)^\top \phi(x_i) = \sum_i \alpha_i\, k(x, x_i)`$ </div>
- **Every kernel method predicts by a weighted sum of similarities to the training points**: The SVM's $`\hat\lambda_0 + \sum_i \alpha_i y_i k(x_i, x)`$, the GP posterior mean, and the NTK's $`k(x, X) K^{-1} y`$.
- Note that we fit $`N`$ coefficients instead of $`P`$ weights — cost scales with *dataset size*, not feature dimension - GP's $`O(n^3)`$.
- Cool property: Penalty-free version. GD from $`w_0 = 0`$ only ever steps along $`\phi(x_i)`$ directions, so it stays in the span with no regulariser (NTK version).
- The kernel is where your assumptions live — made precise as the RKHS norm in [RKHS §The RKHS norm](rkhs.md#the-rkhs-norm), and worked through in [§Kernel ridge regression](#kernel-ridge-regression) below: the same function has a different norm under a different $`k`$, so choosing $`k`$ *is* choosing which functions are cheap

## Kernel ridge regression

- Minimise $`\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}}^2`$ over $`f = \phi^\top w`$
  - In weight space, that is equivalent to $`\|\Phi w - y\|^2 + \lambda \|w\|^2`$. 
  - By the representer theorem $`f = \sum_i \alpha_i k(\cdot, x_i)`$, so $`f(x_i) = (K\alpha)_i`$ and $`\|f\|^2 = \alpha^\top K \alpha`$
  - The objective becomes $`\|K\alpha - y\|^2 + \lambda\, \alpha^\top K \alpha`$
  - And zero gradient gives $`\alpha = (K + \lambda I)^{-1} y`$: <div align="center">
  $`\displaystyle f(x) = k(x, X)\,(K + \lambda I)^{-1}\, y`$ </div>

### What the ridge is actually penalising

- $`\lambda \|w\|^2 = \lambda \|f\|_{\mathcal{H}}^2`$, and $`\|f\|_{\mathcal{H}}`$ is priced *by the kernel* — the same function has a different norm under a different $`k`$ ([RKHS §The RKHS norm](rkhs.md#the-rkhs-norm)). That is the precise sense in which **the kernel is where your assumptions live**
- Why $`\|f\| = \|w\|`$ at all. Read $`\phi(x) \in \mathbb{R}^P`$ not as "a point in feature space" but as **$`P`$ basis functions evaluated at $`x`$**, so $`f_w = \sum_p w_p\, \phi_p`$ and $`w`$ is the coefficient vector of $`f`$ in that basis. Picking $`\phi`$ is picking a basis for (a subspace of) $`\mathcal{H}`$ *and declaring it orthonormal* — same trick as rescaling a basis vector of $`\mathbb{R}^3`$: the vector doesn't change, but its coordinates in that basis do. That declaration is what *induces* the kernel, $`k(x,x') := \phi(x)^\top \phi(x')`$, so ridge on those coordinates matching ridge on $`\mathcal{H}`$ is close to definitional: $`w = \Phi^\top \alpha`$, so $`\|w\|^2 = \alpha^\top \Phi \Phi^\top \alpha = \alpha^\top K \alpha = \|f\|_{\mathcal{H}}^2`$.
- What a wide RBF does. Two training points with $`k = k(x_1, x_2)`$ (so $`K`$ has 1 on the diagonal, $`k`$ off it); the interpolant has $`\|f\|^2 = y^\top K^{-1} y`$:
  - $`y = (1, -1)`$: $`\|f\|^2 = 2/(1 - k) \to \infty`$ as $`k \to 1`$
  - $`y = (1, 1)`$: $`\|f\|^2 = 2/(1 + k) \to 1`$ as $`k \to 1`$
  - Widening $`\sigma`$ pushes $`k \to 1`$ (points are considered more similar). Disagreeing across the two points becomes infinitely expensive; agreeing gets *cheaper*. To disagree is to get more wiggly, which is expensive. Ridge then refuses to pay for the wiggle, and the fit comes out smooth.
- Why narrow → wiggly: $`k(x_i, x_j) \approx 0`$ for $`i \neq j`$, so $`K \approx I`$, $`\alpha \approx y/(1+\lambda)`$, and $`f(x) \approx \frac{1}{1+\lambda} \sum_i y_i\, k(x, x_i)`$ — one bump of height $`\approx y_i`$ at each training point, decaying to 0 between them.
- To generalize, **the spectrum of $`K`$ is the price**. For the interpolant, with $`K = U \Lambda U^\top`$: <div align="center">
  $`\displaystyle \|f\|^2 = y^\top K^{-1} y = \sum_j \frac{(u_j^\top y)^2}{\lambda_j}`$ </div>
  - Wide kernel: $`K \approx \mathbf{1}\mathbf{1}^\top`$, one large eigenvalue (the constant direction) and the rest tiny → one cheap pattern, everything else expensive. Narrow kernel: $`K \approx I`$, flat spectrum → every pattern costs the same
  - This is [NTK §What "learning an eigenvector" means](ntk.md#what-learning-an-eigenvector-means) read as a bill instead of a clock: the directions GD learns last (small $`\lambda_j`$, slow $`e^{-\lambda_j t}`$) are exactly the ones the norm charges most for.

- Takeaway: **choosing $`k`$ is choosing which functions are cheap.** 

## Where the NTK fits

| Generic kernel idea | NTK instance |
|---|---|
| feature map $`\phi : \mathcal{X} \to \mathbb{R}^P`$ | $`\phi_t(x) = \nabla_\theta f_\theta(x)`$ — the gradient, one entry per parameter, so $`P`$ = parameter count |
| kernel $`k = \phi^\top \phi`$ | $`K_t(x, x') = \sum_p \partial f(x)/\partial\theta_p \cdot \partial f(x')/\partial\theta_p`$ |
| the linear model $`w \mapsto \phi(x)^\top w`$ | $`\theta \mapsto \phi_t(x)^\top \theta`$ — the network's own parameters play $`w`$ |
| representer theorem | $`f_\infty(x) = f_0(x) + k(x, X) K^{-1}(y - f_0(X))`$ — ridgeless kernel ridge on the residual |

- Two things are more unusual about NTK:
  - Normally *we choose* $`k`$ to encode a belief about which inputs behave alike. The NTK is the kernel *implied* by an architecture + init
  - Normally $`\phi`$ is fixed by construction (GP does have a few knobs that move). In a network, $`\phi_t`$ drifts more freely as training moves $`\theta`$, and only stops in the infinite-width limit.
