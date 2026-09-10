# RKHS

The function-space side of [Kernels](kernels.md): what a Hilbert space is, what makes one "reproducing kernel", and why the RKHS norm is the thing every kernel method — and gradient descent in the [NTK](ntk.md) regime — is actually regularising. References: Hofmann, Schölkopf & Smola, [Kernel Methods in Machine Learning](https://arxiv.org/abs/math/0701907) §2; Rasmussen & Williams, [GPML](http://gaussianprocess.org/gpml/) §6.1–6.2 for the RKHS ↔ GP view and the spectral norm; the original is [Aronszajn (1950)](https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/).

**An RKHS is a norm on functions that controls their values, packaged with a kernel that lets you compute that norm.** 

## TLDR

- **A Hilbert space** is a vector space equipped with an inner product, where it is complete with respect to the norm that inner product defines. This gives us length, angle, projection, and limits that don't leak out. Note that in finite dimensions, completeness is automatic: $`\mathbb{R}^P`$ with *any* inner product is a Hilbert space. This definition matters for infinite dimensions.
- **An RKHS** is a Hilbert space *whose elements are functions* $`f : \mathcal{X} \to \mathbb{R}`$, such that evaluation at each point, $`\delta_x : f \mapsto f(x)`$, is continuous: $`|f(x)| \leq C_x \|f\|_{\mathcal{H}}`$. Small norm ⇒ small at every point. 
  - RKHS-ness is *not* a property of the abstract Hilbert space. It's a property of how the elements are identified with functions. **RKHS = (Hilbert space, realisation of its elements as functions on $`\mathcal{X}`$), with point evaluation continuous**
  - The continuity is *in $`f`$, with $`x`$ held fixed*. For that fixed $x$, the change in $f(x)$ is bounded by (a constant times) the change in $f$.
- **Why an RKHS and not just a Hilbert space of functions?** 
  - In a generic Hilbert space of functions like $`L^2`$, $`\|f\|`$ can be tiny while $`f(x) = 1`$. Regularising $`\|f\|`$ then says nothing about predictions
  - In an RKHS, we get two things:
    - $`|f(x)| \leq \sqrt{k(x, x)}\, \|f\|_{\mathcal{H}}`$: small norm ⇒ small predictions everywhere, so penalising the norm is meaningful. 
    - Every inner product, norm and evaluation is computable from the kernel alone — $`\|\sum_i \alpha_i g_{x_i}\|^2 = \alpha^\top K \alpha`$, $`f(x) = \langle f, g_x \rangle`$ — so you never represent the (often infinite-dimensional) space
- **RKHSs and kernels are in bijection.** Given an RKHS, its reproducing kernel is unique (Riesz representers are unique). Given a PSD kernel, the RKHS having it as its reproducing kernel is unique (Moore–Aronszajn)
  ```math
  \{\text{RKHSs of functions on } \mathcal{X}\} \;\longleftrightarrow\; \{\text{PSD kernels on } \mathcal{X}\}
  ```
  - What is *not* unique is the **feature map**: many $`\phi`$ give the same $`k`$.
  - An RKHS is the functions *together with* their inner product. Same functions, inner product scaled by $`c`$ → a different RKHS, with kernel $`k / c`$. Change the norm, change the kernel
- **A kernel admits three equivalent definitions** — as a function, as a matrix property, and as a reproducing kernel
  1. Function view: $`k(x, x') = \langle \phi(x), \phi(x') \rangle`$ for *some* $`\phi`$ into *some* inner-product space
  2. Matrix view: $`k`$ symmetric, and every Gram matrix $`[k(x_i, x_j)]_{ij}`$ is PSD
  3. Reproducing-kernel view: $`k`$ is the reproducing kernel of some RKHS — a Hilbert space of functions in which $`f(x) = \langle f, g_x \rangle`$ for every $`f`$, where $`g_x(x') = k(x', x)`$
  - These are equivalent. 1 ⇒ 2 is one line: $`u^\top K u = \|\sum_i u_i \phi(x_i)\|^2 \geq 0`$. 3 ⇒ 1 is Riesz and 2 ⇒ 3 is Moore–Aronszajn — both proved at the bottom of this note

## Notation

- $`\mathcal{H}`$ : a set of functions $`f : \mathcal{X} \to \mathbb{R}`$ treated as a vector space. Each function is one point in $`\mathcal{H}`$
- $`\langle f, g \rangle_{\mathcal{H}}`$ : an inner product *between functions*.
- $`\|f\|_{\mathcal{H}} = \sqrt{\langle f, f \rangle_{\mathcal{H}}}`$ : the length of a function.
- $`g_x \in \mathcal{H}`$ : $`g_x(x') = k(x', x)`$. One function per $`x`$. Papers write it $`k(\cdot, x)`$ or $`k_x`$; this note uses $`g_x`$. When you start from the space (Riesz) $`g_x`$ is constructed first and $`k`$ is read off from it; when you start from $`k`$ (Moore–Aronszajn) it's defined from $`k`$. Same object either way
- $`\delta_x : f \mapsto f(x)`$ : evaluation at $`x`$. A **functional** — a function whose input is a function and output is a number. Linear: $`\delta_x(f + cg) = f(x) + c\, g(x)`$. 
- Bounded = continuous, for a linear functional: $`|L(f)| \leq C \|f\|_{\mathcal{H}}`$ for some constant $`C`$
- Cauchy sequence: terms get arbitrarily close *to each other*. 
- Complete: every Cauchy sequence has a limit *inside the space*
- $`L^2`$ : functions with $`\int f^2 < \infty`$, inner product $`\int f g`$. A Hilbert space of functions, and *not* an RKHS

## Mechanics

### Hilbert space = vector space + inner product + complete

- **Vector space** $`V`$ over $`\mathbb{R}`$: a set with addition $`u + v`$ and scaling $`c\, v`$ that obey the $`\mathbb{R}^P`$ rules
  - Addition: associative, commutative, a zero vector, every $`v`$ has a $`-v`$
  - Scaling: $`c(u + v) = cu + cv`$, $`(c + d)v = cv + dv`$, $`(cd)v = c(dv)`$, $`1v = v`$
  - Functions qualify, pointwise: $`(f + g)(x) = f(x) + g(x)`$, $`(cf)(x) = c\, f(x)`$
  - $`\mathbb{R}^P`$ is the case where a vector is a function $`\{1, \ldots, P\} \to \mathbb{R}`$, $`v_i = v(i)`$. A function on $`\mathbb{R}`$ is a vector with a continuum of coordinates
- **Inner product** $`\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}`$
  - Symmetric: $`\langle u, v \rangle = \langle v, u \rangle`$
  - Linear in each slot: $`\langle au + bw, v \rangle = a\langle u, v \rangle + b\langle w, v \rangle`$
  - Positive definite: $`\langle v, v \rangle > 0`$ for $`v \neq 0`$
  - Defines the norm $`\|v\| = \sqrt{\langle v, v \rangle}`$, hence distance $`\|u - v\|`$, angle, orthogonality, projection
  - It's a *choice*: $`\sum_i v_i w_i`$ on $`\mathbb{R}^P`$, or $`\int f g`$ on functions ($`L^2`$). Same set, different inner product, different geometry
- **Complete**: every Cauchy sequence converges to a limit that is *in* $`V`$
  - Cauchy: $`\|v_m - v_n\| \to 0`$, measured by the norm above — terms get close to *each other*
  - $`\mathbb{Q}`$ isn't complete: $`1, 1.4, 1.41, \ldots \to \sqrt 2 \notin \mathbb{Q}`$. Polynomials under $`L^2`$ aren't: Taylor sums of $`e^x`$ are Cauchy, $`e^x`$ isn't a polynomial
  - Finite-dimensional inner product spaces always are complete
  - *Completing* an incomplete space = adjoin every such limit
- **A Hilbert space is a set with all three**
  - Completeness gives us projections onto closed subspaces (the $`w = w_\parallel + w_\perp`$ split in [Kernels §Representer theorem](kernels.md)), Pythagoras, and convergent infinite expansions

### RKHS: the definition

- An **RKHS** is a Hilbert space $`\mathcal{H}`$ whose elements are functions $`f : \mathcal{X} \to \mathbb{R}`$, such that for every $`x \in \mathcal{X}`$ the evaluation functional $`\delta_x : f \mapsto f(x)`$ is continuous. I.e. there is a constant $`C_x`$ with $`|f(x)| \leq C_x \|f\|_{\mathcal{H}}`$ for all $`f`$
  - Continuity is in $`f`$, with $`x`$ fixed.
  - Note that no kernel appears in this definition. It is derived from it (proved at the bottom).
  - **The name.** Evaluating $`f`$ at $`x`$ turns out to equal an inner product with one fixed element $`g_x`$ of the space: $`f(x) = \langle f, g_x \rangle_{\mathcal{H}}`$. The **kernel** is the table of those elements' values, $`k(x', x) = g_x(x')`$, and it **reproduces** because $`\langle f, g_x \rangle`$ hands back $`f(x)`$. So: a Hilbert space of functions that comes with a kernel whose sections reproduce evaluation

### Why $`L^2`$ isn't an RKHS

- In $`L^2`$, two functions differing at a single point have distance 0 — a point contributes nothing to an integral.
- This is undesirable because if small $`\|f\|`$ says nothing about $`f(x)`$, then regularising $`\|f\|`$ says nothing about $`f(x)`$. 

## The RKHS norm

Three facts, proved in the last section of this note, are used freely here:

- **Reproducing property**: $`f(x) = \langle f, g_x \rangle_{\mathcal{H}}`$ for every $`f \in \mathcal{H}`$ and every $`x \in \mathcal{X}`$
- **Norm from the kernel**: for $`f = \sum_i \alpha_i g_{x_i}`$, $`\|f\|_{\mathcal{H}}^2 = \alpha^\top K \alpha`$, and the training values are $`f(x_i) = (K\alpha)_i`$
- **Representer theorem**: the minimiser of any $`\text{loss}(f(x_1), \ldots, f(x_N)) + \lambda \|f\|_{\mathcal{H}}^2`$ has the form $`\sum_{i=1}^N \alpha_i g_{x_i}`$ over the training points — anything orthogonal to the training sections is zero at the training points, so it changes no loss and only adds norm

### Why an RKHS

- Every Hilbert space has a norm, $`L^2`$ included. The difference is what the norm *controls*
  - In $`L^2`$, $`\|f\|`$ can be tiny while $`f(x) = 1`$. The norm says nothing about predictions
  - In an RKHS, $`|f(x)| = |\langle f, g_x \rangle| \leq \|g_x\| \, \|f\|_{\mathcal{H}} = \sqrt{k(x, x)}\, \|f\|_{\mathcal{H}}`$ by Cauchy–Schwarz. The norm bounds every value
- So an RKHS gives a norm on functions with two properties at once:
  1. It means something for predictions: small norm ⇒ small values everywhere
  2. It is computable from kernel evaluations alone: $`\alpha^\top K \alpha`$

### Example: Kernel ridge regression 

- Minimise $`\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}}^2`$
- Representer theorem: $`f = \sum_j \alpha_j g_{x_j}`$. Then $`f(x_i) = (K\alpha)_i`$ and $`\|f\|^2 = \alpha^\top K \alpha`$, so the objective is $`\|K\alpha - y\|^2 + \lambda\, \alpha^\top K \alpha`$
- Zero gradient gives $`\alpha = (K + \lambda I)^{-1} y`$. The infinite-dimensional problem became an $`N \times N`$ solve
- The abstract RKHS norm *is* the ridge penalty. GP regression is the same formula with $`\lambda = \sigma^2`$ ([Kernels](kernels.md)); the SVM is hinge loss with the same $`\|f\|_{\mathcal{H}}`$. All three penalise the same quantity
- In the explicit-feature view, $`\mathcal{H} = \{ f_w : f_w(x') = w^\top \phi_e(x') \}`$ and $`\|f\|_{\mathcal{H}} = \min\{\|w\| : f_w = f\}`$. So $`\lambda \|w\|^2`$ and $`\lambda \|f\|_{\mathcal{H}}^2`$ are the same penalty

### What "small norm" means: cheap according to $`k`$

- The norm is relative to $`k`$. The same function has different norms under different kernels.
- Disagreeing about similar points is expensive. Suppose $`f(x_1) = +1`$ and $`f(x_2) = -1`$. The no-penalty $`f`$ has $`\alpha = K^{-1} y`$ and
  ```math
  \|f\|_{\mathcal{H}}^2 = y^\top K^{-1} y = \frac{2}{1 - k(x_1, x_2)} \qquad \text{(RBF, so } k(x,x) = 1\text{)}
  ```
  - As $`k(x_1, x_2) \to 1`$ (points move closer, or $`\sigma`$ widens), the cost → ∞. Ridge shrinks this function hard; GD learns it last

## Proofs: the three definitions are equivalent

The two sections below prove 3 ⇒ 1 and 2 ⇒ 3; 1 ⇒ 2 is the one-liner in the TLDR. Skip on first read. Everything above uses only their three outputs: the reproducing property, $`\|\sum_i \alpha_i g_{x_i}\|^2 = \alpha^\top K \alpha`$, and orthogonal-to-the-training-sections ⇔ zero-at-the-training-points.

### RKHS → kernel (Riesz): definition 3 ⇒ definition 1

- **Goal.** Start from an RKHS $`\mathcal{H}`$ (the definition above). Produce a two-argument function $`k`$ with $`f(x) = \langle f, g_x \rangle`$ for every $`f`$ — its **reproducing kernel** — and show that $`k`$ has the definition-1 form. That proves a reproducing kernel (definition 3) is a kernel (definition 1)
- Let $`\mathcal{H}`$ be such an RKHS. Two facts combine:
  - **Riesz representation theorem**, true in *any* Hilbert space, RK or not: every continuous linear functional $`L`$ is an inner product with one fixed vector, $`L(f) = \langle f, v_L \rangle`$ for a unique $`v_L \in \mathcal{H}`$
  - **The RKHS definition**: $`\delta_x`$ is linear (evaluation is linear) and continuous (by definition)
- Apply Riesz to $`L = \delta_x`$. Since $`\delta_x(f) = f(x)`$ by definition, for every $`x`$ there is a unique $g_x \in \mathcal{H}$, with
  ```math
  f(x) = \langle f, g_x \rangle_{\mathcal{H}} \qquad \text{for every } f \in \mathcal{H}
  ```
  - This is the **reproducing property**: evaluation is an inner product
  - Two quantifiers, in this order. Riesz runs once per $`x`$ and returns one function $`g_x`$; that function then works for every $`f`$
  - $`g_x`$ is a function $`\mathcal{X} \to \mathbb{R}`$, and a different one for each $`x`$
- Define a function of two arguments by $`k(x', x) := g_x(x')`$. This $`k`$ is the **reproducing kernel** of $`\mathcal{H}`$ — definition 3. It is also a kernel by definition 1:
  - Apply the reproducing property to $`f = g_{x'}`$. Left side: $`f(x) = g_{x'}(x) = k(x, x')`$, by definition of $`k`$. Right side: $`\langle g_{x'}, g_x \rangle_{\mathcal{H}}`$
  - So $`k(x, x') = \langle g_{x'}, g_x \rangle_{\mathcal{H}} = \langle g_x, g_{x'} \rangle_{\mathcal{H}}`$, the second equality by symmetry of the inner product. Symmetry $`k(x, x') = k(x', x)`$ follows
  - Define the map $`\phi : \mathcal{X} \to \mathcal{H}`$ by $`\phi(x) := g_x`$. Two different functions are now in play, and the notation must keep them apart:
    - $`\phi : \mathcal{X} \to \mathcal{H}`$ takes a point $`x`$ and returns the function $`g_x`$
    - $`\phi(x) : \mathcal{X} \to \mathbb{R}`$ takes a point $`x'`$ and returns the number $`k(x', x)`$
  - Then $`k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}`$, which is definition 1 with $`\mathcal{H}`$ itself as the feature space
  - $`\phi`$ sends each point to its section — its similarity profile, "how similar is every point of $`\mathcal{X}`$ to $`x`$". It is called the **canonical feature map** because it is built from $`k`$ alone, with no choice of coordinates; every other feature map for $`k`$ is this one seen through an isometry
- $`g_x`$ is a function, not a number — the second line of the two types above. As a vector in $`\mathcal{H}`$ it has coordinates:
  - In [Kernels](kernels.md), the feature vector was an explicit $`\phi_e(x) \in \mathbb{R}^P`$ with coordinates indexed by $`1 \ldots P`$. Here $`g_x`$ is a vector in $`\mathcal{H}`$ with coordinates indexed by $`x' \in \mathcal{X}`$ — its coordinate at $`x'`$ is its value there
  - These are the same vector in two bases. For the quadratic kernel, $`\phi_e(x) = (x_1^2, \sqrt2\, x_1 x_2, x_2^2) \in \mathbb{R}^3`$
    - Every $`w \in \mathbb{R}^3`$ defines a function $`f_w(x') = w^\top \phi_e(x')`$. The map $`w \mapsto f_w`$ is an isometry from $`\mathbb{R}^3`$ onto $`\mathcal{H}`$ (the homogeneous quadratics), with $`\|f_w\|_{\mathcal{H}} = \|w\|`$
    - Under it, $`\phi_e(x) \mapsto g_x`$, because $`f_{\phi_e(x)}(x') = \phi_e(x)^\top \phi_e(x') = k(x', x) = g_x(x')`$
    - So $`g_x(x') = \phi_e(x)^\top \phi_e(x')`$: evaluating the function is a dot product with the feature vector of $`x`$, once $`x'`$ is lifted into the feature space too. It is not $`\phi_e(x)^\top x'`$ — $`x'`$ lives in $`\mathcal{X}`$, not in the feature space
- The constant in the continuity bound is explicit. By Cauchy–Schwarz, $`|f(x)| = |\langle f, g_x \rangle| \leq \|f\|_{\mathcal{H}} \|g_x\|_{\mathcal{H}}`$, and $`\|g_x\|^2 = \langle g_x, g_x \rangle = k(x, x)`$, so $`C_x = \sqrt{k(x, x)}`$
  - Hence $`|f(x) - h(x)| \leq \sqrt{k(x, x)}\, \|f - h\|_{\mathcal{H}}`$: functions close in $`\mathcal{H}`$ make close predictions everywhere. This is what $`L^2`$ could not give

### Kernel → RKHS (Moore–Aronszajn): definition 2 ⇒ definition 3

- **Goal.** Definition 3 says $`k`$ is the reproducing kernel of some RKHS. So, starting from a symmetric PSD $`k`$ (definition 2), build a Hilbert space $`\mathcal{H}`$ of functions $`\mathcal{X} \to \mathbb{R}`$ and check two things against the RKHS definition above: evaluation is continuous on it (so it is an RKHS), and $`\langle f, g_x \rangle = f(x)`$ with $`g_x(x') = k(x', x)`$ (so $`k`$ is its reproducing kernel)
- **Building blocks.** For each $`x \in \mathcal{X}`$, the section $`g_x : \mathcal{X} \to \mathbb{R}`$, $`g_x(x') = k(x', x)`$. There is one for every point of the input space — for $`\mathcal{X} = \mathbb{R}^d`$, a continuum of them. No data is involved
- **Step 1: the functions.** $`\mathcal{H}_0 = \mathrm{span}\{g_x : x \in \mathcal{X}\}`$: every function of the form $`f = \sum_{i=1}^m \alpha_i g_{x_i}`$ for some finite $`m`$, some points $`x_1, \ldots, x_m \in \mathcal{X}`$, and some $`\alpha_i \in \mathbb{R}`$
- **Step 2: the inner product.** For $`f = \sum_i \alpha_i g_{x_i}`$ and $`h = \sum_j \beta_j g_{y_j}`$, define
  ```math
  \langle f, h \rangle := \sum_{i,j} \alpha_i \beta_j\, k(x_i, y_j) = \alpha^\top K \beta
  ```
  - Symmetric because $`k`$ is. Bilinear by construction. $`\langle f, f \rangle = \alpha^\top K \alpha \geq 0`$ because $`K`$ is PSD — this is where definition 2 is used

- **Step 3: the reproducing property on $`\mathcal{H}_0`$.** $`\langle f, g_x \rangle = f(x)`$
  - Evaluate $`f`$ at $`x`$ pointwise: $`f(x) = \sum_i \alpha_i g_{x_i}(x) = \sum_i \alpha_i k(x, x_i)`$, by the definition of $`g`$
  - Compute $`\langle f, g_x \rangle`$ by the step-2 rule with $`h = g_x`$ (one term, coefficient 1): $`\sum_i \alpha_i k(x_i, x)`$
  - The two agree because $`k`$ is symmetric, $`k(x, x_i) = k(x_i, x)`$
- **Evaluation is continuous on $`\mathcal{H}_0`$.** By Cauchy–Schwarz, $`|f(x)| = |\langle f, g_x \rangle| \leq \|f\| \, \|g_x\| = \sqrt{k(x, x)}\, \|f\|`$. So $`\mathcal{H}_0`$ meets the RKHS definition except for completeness. (Also positive definite: $`\langle f, f \rangle = 0`$ forces $`f(x) = 0`$ for every $`x`$)
- **Step 4: complete.** Adjoin the limits of Cauchy sequences to get $`\mathcal{H}`$
  - The limits are still functions on $`\mathcal{X}`$: if $`f_n`$ is Cauchy in norm, the bound above makes $`f_n(x)`$ Cauchy in $`\mathbb{R}`$ for each $`x`$, so the pointwise limit exists. The reproducing property and the continuity bound pass to the limit
  - $`\mathcal{H}`$ is a Hilbert space of functions with continuous evaluation — an RKHS — and its reproducing kernel is $`k`$. Definition 3 holds
- **Uniqueness.** Any RKHS with reproducing kernel $`k`$ contains every $`g_x`$ and agrees with the step-2 inner product on their span, so it contains $`\mathcal{H}`$; and nothing else, since a function orthogonal to every $`g_x`$ is zero at every $`x`$. This is the bijection in the TLDR
