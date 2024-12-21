# Linear Algebra and Calculus

## Linear Algebra
- Geometric intuition
  - I really like thinking of linear algebra in geometric terms. I think [Zhang, Lipton, Li and Smola](http://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) does an amazing job of explaining this.
- Interpretation of matrix multiplication: Consider $\mathbf{Ax} = \mathbf{b}$ for $\mathbf{A} \in \mathbb{R}^{n \times m}, \mathbf{x} \in \mathbb{R}^{m \times 1}$
  - Column interpretation: $\mathbf{Ax} = \sum_i^m x_i\mathbf{A}_{:,i}$, a linear combination of the columns of $\mathbf{A}$ 
  - Row interpretation: $b_j = \mathbf{A}_{j,:}\mathbf{x}$, how similar are the rows of $\mathbf{A}$ to $\mathbf{x}$?
- Definitions
  - For an inverse to exist, the columns of $\mathbf{A}$ need to span $\mathbb{R}^n$, and they have to be linearly independent, i.e. the matrix is nonsingular.
    - Note that if columns are not independent, then a combination of standard basis vectors would map to the same line as another standard basis vector that's not included in that set.
    - Invertibility: If you have "collapsed" the space, how are you going to recreate it? 
  - $L^p$ norm: $\|\mathbf{x}\|_p=\left(\sum_i\left|x_i\right|^p\right)^{\frac{1}{p}}$
  - $L^{\infty}$ = $\|\mathbf{x}\|_{\infty}=\max _i\left|x_i\right|$.
  - Frobenius norm: $\|\mathbf{A}\|_F=\sqrt{\sum_{i, j} A_{i, j}^2}$
  - An orthogonal matrix is one whose rows and columns are orthonormal to each other: $\mathbf{A}^{\top} \mathbf{A}=\mathbf{A} \mathbf{A}^{\top}=\mathbf{I}$
- Eigenvectors and eigenvalues
  - The (right) eigenvector of a square matrix $\mathbf{A}$ is a nonzero vector $\mathbf{v}$ s.t. $\mathbf{Av}$ = $\lambda\mathbf{v}$. $\lambda$ is known as the eigenvalue associated with that eigenvector.
    - Think of $\mathbf{A}$ scaling space in the direction of $\mathbf{v}_i$ by $\lambda_i$.
  - Eigendecomposition
    - Suppose that $\mathbf{A}$ has $n$ linearly independent eigenvectors with eigenvalues $\lambda_i$
    - We concatenante these vectors and values together to form $\mathbf{V}$ and $\mathbf{\lambda}$.
    - Then the eigendecomposition of $\mathbf{A}$ is given by $\mathbf{V}diag(\mathbf{\lambda})\mathbf{V}^{-1}$ (since $\mathbf{AV=V\Lambda}$).
    - Spectral Theorem: Every real symmetric matrix can be decomposed into $\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\top}$, where $\mathbf{Q}$ is an orthogonal matrix comprising of only real-valued eigenvectors and our eigenvalues are also real-valued.
    - The matrix is singular $\iff$ any eigenvalue is 0 (this implies that columns are linearly dependent).
    - If all eigenvalues are positive, the matrix is positive definite. ($\forall$ $\mathbf{x}, \mathbf{xA^{\top}x}>0$. Proof: We can express $\mathbf{x}$ as a sum of eigenvectors and $\mathbf{vA^{\top}v}=\lambda>0$).
- Other Decompositions
  - SVD: $\mathbf{A} = \mathbf{UDV^{\top}}$, which exists for all matrices. Importantly, $\mathbf{U}$ and $\mathbf{V}$ are both orthogonal. 
    - The Moore-Penrose pseudoinverse effectively takes the inverse of this decomposition to help solve for $\mathbf{Ax} = \mathbf{b}$. 
  - LU: For square matrices, $\mathbf{A} = \mathbf{LU}$. This isn't guaranteed to exist, but if we permit a permutation matrix, then this is: $\mathbf{PA} = \mathbf{LU}$.
  - Choleseky: Every Hermitian, positive-definite matrix has a unique Cholesky decomposition $\mathbf{A} = \mathbf{LL}^*$, where $\mathbf{L}^*$ is the conjugate transpose of $\mathbf{L}$. 
    - If $\mathbf{A}$ is real, we can write $\mathbf{A} = \mathbf{LL}^{\top}$, where $\mathbf{L}$ has positive diagonal entries.
  - Why are these decompositions useful? When there is no inverse, or multiple pseudo-inverses. It is also computationally faster! (To add more details)
- Trace
  - Tr($\mathbf{A}$)$=\sum_i A_{ii}$
  - When the product is a square matrix, $\operatorname{Tr}\left(\prod_{i=1}^n \mathbf{F}^{(i)}\right)=\operatorname{Tr}\left(\mathbf{F}^{(n)} \prod_{i=1}^{n-1} \mathbf{F}^{(i)}\right)$
  - The trace of a square matrix is equal to the sum of its eigenvalues. 
- Determinant
  - The determinant of a square matrix is the product of all eigenvalues and can be thought of how much a matrix expands or contracts space. 
  - In the event that matrix is not full rank, then we're "collapsing" space and the determinant is 0. 

## Calculus

### Multivariable Calculus
- Let $y = f(x_1,x_2,\dots,x_n)$, i.e. $f: \mathbb{R}^n \rightarrow \mathbb{R}$.
- Then $\frac{\partial y}{\partial x_i}=\frac{\partial f}{\partial x_i}=\partial_{x_i} f=\partial_i f=f_{x_i}=f_i=D_i f=D_{x_i} f=\lim _{h \rightarrow 0} \frac{f\left(x_1, \ldots, 
x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n\right)-f\left(x_1, \ldots, x_i, \ldots, x_n\right)}{h},$
- $\nabla_{\mathbf{x}} f(\mathbf{x})=\left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots \partial_{x_n} f(\mathbf{x})\right]^{\top}$
- The **Jacobian**: Extending this, $\nabla_{\mathbf{x}}\mathbf{y} \in \mathbb{R}^{n \times m}$ if $\mathbf{y} \in \mathbb{R}^{m}$, and the (i,j) entry of this matrix encapsulates $\frac{\partial y_j}{\partial x_i}$.
- For all $\mathbf{A} \in \mathbb{R}^{m \times n}$ we have $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x}=\mathbf{A}^{\top}$ and $\nabla_{\mathbf{x}} \mathbf{x}^{\top} \mathbf{A}^{\top}=\mathbf{A}^{\top}$.
  - This is saying that the jth entry of $\mathbf{A} \mathbf{x}$ varies by $A_{j,i}$ wrt $x_i$, which makes sense because this entry is the dot product of the jth row of $\mathbf{A}$ and $\mathbf{x}$.
- For square matrices,  we have that $\nabla_{\mathbf{x}} \mathbf{x}^{\top} \mathbf{A} \mathbf{x}=\left(\mathbf{A}+\mathbf{A}^{\top}\right) \mathbf{x}$ and in particular, $\nabla_{\mathbf{x}}\|\mathbf{x}\|^2=\nabla_{\mathbf{x}} \mathbf{x}^{\top} \mathbf{x}=2 \mathbf{x}$.
  - It will help to note that $ \mathbf{x}^{\top} \mathbf{A} \mathbf{x}= \sum_i\sum_jx_ix_jA_{ij}$
- Similarly, for any matrix $\mathbf{X}$, we have $\nabla_{\mathbf{X}}\|\mathbf{X}\|_F^2=2 \mathbf{X}$.

- Chain Rule: $\frac{\partial y}{\partial x_i}=\frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i}+\frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i}+\cdots+\frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}$ and so 
$\nabla_{\mathbf{x}} y=\nabla_{\mathbf{x}}\mathbf{u} \nabla_{\mathbf{u}} y$ 
- Hessian: The Hessian $\mathbf{H}$ of $f(\mathbf{x})$ is such that $H_{ij} = \frac{\partial^2}{\partial x_i \partial x_j}f(\mathbf{x})$
  - It specifies the curvature in the direction of the basis vectors. 
  - It's bounded by the minimum and maximum eigenvalues.
- Taylor expansion: $f\left(x_k+\Delta x\right) \approx f\left(x_k\right)+\nabla f\left(x_k\right)^{\mathrm{T}} \Delta x+\frac{1}{2} \Delta x^{\mathrm{T}} H \Delta x$
- Hessian and the nature of a stationary point
  - $f(\mathbf{x})$ is a local minimum $\iff$ $f$ is twice continuously differentiable with $\nabla^2f$ positive semi-definite in the neighborhood of $\mathbf{x}$ and that $\nabla f(\mathbf{x}) = \mathbf{0}$.
  - We can reason this geometrically. 
  - Importantly, if there's one positive and one negative eigenvalue, we have a saddle point. 
  - It is also possible for a point to be inconclusive (with all eigenvalues sharing the same sign and at least one zero).
- Convexity:
  - A set $\mathcal{X} \subseteq \mathbb{R}^d$ is convex if $t\mathbf{x} + (1-t)\mathbf{y} \in \mathcal{X}$ for all $\mathbf{x, y} \in \mathcal{X}$ and all $t \in [0,1]$.
  - $f$ is convex if $f(t\mathbf{x} + (1-t)\mathbf{y}) \leq tf(\mathbf{x}) + (1-t)f(\mathbf{y})$ for all $\mathbf{x, y} \in Domain(f)$ and all $t \in [0,1]$.
    - E.g., Norms are convex
  - Let $\mathcal{X}$ be a convex set. If f is convex, then any local minimum of $f$ in $\mathcal{X}$ is also a global minimum.
  - $f$ is convex $\iff$ $\nabla^2f(\mathbf{x}) \geq 0$ for all $\mathbf{x} \in Domain(f)$
  - If $f_i$ are convex, and $\alpha_i \geq0$, then $\sum_i \alpha_if_i$ is convex.
  - If $f$ is convex, then $g(\mathbf{x}) = f(\mathbf{Ax + b)})$ is convex.
  - If $f$ and $g$ is convex, then $h(\mathbf{x}) = \max\{f(\mathbf{x}), g(\mathbf{x})\}$ is convex.
- Integrals:
  - Fubini: $\iint_{X \times Y} f(x, y) \mathrm{d}(x, y)=\int_X\left(\int_Y f(x, y) \mathrm{d} y\right) \mathrm{d} x=\int_Y\left(\int_X f(x, y) \mathrm{d} x\right) \mathrm{d} y$ if $f$ is continuous over $X \times Y$. 
  - Change of variables: $\int_{\phi(U)} f(\mathbf{x}) d \mathbf{x}=\int_U f(\phi(\mathbf{x}))|\operatorname{det}(D \phi(\mathbf{x}))| d \mathbf{x}$, where $D \pmb{\phi}=\left[\begin{array}{ccc}\frac{\partial \phi_1}{\partial x_1} & \cdots & \frac{\partial \phi_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial \phi_n}{\partial x_1} & \cdots & \frac{\partial \phi_n}{\partial x_n}\end{array}\right]$

### Lagrangian And KKT Conditions
- Simple case: Suppose we want to find $\hat{\mathbf{x}}$ that minimizes $f(\mathbf{x})$ subject to $g(\mathbf{x}) = c$
  - Minimizing $f(\mathbf{x})$ is equivalent to finding the level set of $f(\mathbf{x})$ that is tangent to the level set $g(\mathbf{x}) = c$.
  - Equivalently, $\nabla f(\mathbf{x}) = \lambda\nabla g(\mathbf{x})$ (eqn 1).
  - To know what $\lambda$ is, we plug in $g(\mathbf{x}) = c$ (eqn 2) and we're done!
    - A pretty convoluted way of saying this is to define the Lagrangian $\mathcal{L}(\mathbf{x},\lambda)=f(\mathbf{x}) -\lambda(g(\mathbf{x}) -c)$
    - Then we have that $\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x},\lambda) = 0$ (eqn 1) and $\nabla_{\lambda}\mathcal{L}(\mathbf{x},\lambda) = 0$ (eqn 2)
  - Note that sometimes we get the constraint in the form $g(\mathbf{x}) \leq c$. A common trick is that when the constraint is binding, we set $g(\mathbf{x}) = c$.
- Generalized Lagrangian
  - Suppose now we want to find $\hat{\mathbf{x}}$ that minimizes $f(\mathbf{x})$ s.t. $g^{(i)}(\mathbf{x}) = 0$ and $h^{(i)}(\mathbf{x}) \leq 0$ $\forall$ $i,j$
  - We now define the generalized Lagrangian $L(\mathbf{x}, \pmb{\lambda}, \pmb{\alpha})=f(\mathbf{x})+\sum_i \lambda_i g^{(i)}(\mathbf{x})+\sum_j \alpha_j h^{(j)}(\mathbf{x})$
    - Then $\hat{\mathbf{x}} = \min _{\mathbf{x}} \max _{\pmb{\lambda}} \max _{\pmb{\alpha}, \pmb{\alpha} \geq 0} L(\mathbf{x}, \pmb{\lambda}, \pmb{\alpha})$ (If any constraint is violated, you can choose $\pmb\lambda, \pmb\alpha$ that makes the lagrangian go to $\infty$)
    - If a feasible solution exists, this is equivalent to finding $(\hat{\pmb{\lambda}}, \hat{\pmb{\alpha}}) = \max _{\pmb{\lambda}} \max _{\pmb{\alpha}, \pmb{\alpha} \geq 0}\min _{\mathbf{x}} L(\mathbf{x}, \pmb{\lambda}, \pmb{\alpha})$
  - The KKT conditions are necessary (but not always sufficient) conditions for a point to be optimal.
    - Primal feasibility: $g^{(i)}(\mathbf{x^*}) = 0$ and $h^{(i)}(\mathbf{x^*}) \leq 0$ $\forall$ $i,j$
    - Dual feasibility: $\pmb{\alpha^*} \geq 0$
    - Lagrangian stationarity: $\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^*, \pmb{\alpha}^*, \pmb{\beta}^*\right)=\mathbf{0}$
    - Complementary slackness: $\pmb{\alpha}$ $\odot$ $\mathbf{h(x^*)} = \mathbf{0}$
      - This last condition essentially says that when $h^{(i)}(\mathbf{x^*}) < 0$ (the constraint is not binding), $\alpha_i = 0$ (See [Support Vectors](../08_svms/notes.md)) 