# Linear Algebra and Calculus

## Linear Algebra
- Interpretation of matrix multiplication: Consider $\mathbf{Ax} = \mathbf{b}$ for $\mathbf{A} \in \mathbb{R}^{n \times m}, \mathbf{x} \in \mathbb{R}^{m \times 1}$
  - Column interpretation: $\mathbf{Ax} = \sum_i^m x_i\mathbf{A}_{:,i}$, a linear combination of the columns of $\mathbf{A}$ 
  - Row interpretation: $b_j = \mathbf{A}_{j,:}\mathbf{x}$, how similar are the rows of $\mathbf{A}$ to $\mathbf{x}$?
- Definitions
  - For an inverse to exist, the columns of $\mathbf{A}$ need to span $\mathbb{R}^n$, and they have to be linearly independent, i.e. the matrix is nonsingular.
  - $L^p$ norm: $\|\mathbf{x}\|_p=\left(\sum_i\left|x_i\right|^p\right)^{\frac{1}{p}}$
  - $L^{\infty}$ = $\|\mathbf{x}\|_{\infty}=\max _i\left|x_i\right|$.
  - Frobenius norm: $\|\mathbf{A}\|_F=\sqrt{\sum_{i, j} A_{i, j}^2}$
  - An orthogonal matrix is one whose rows and columns are orthonormal to each other: $\mathbf{A}^{\top} \mathbf{A}=\mathbf{A} \mathbf{A}^{\top}=\mathbf{I}$
- Eigenvectors and eigenvalues
  - The (right) eigenvector of a square matrix $\mathbf{A}$ is a nonzero vector $\mathbf{v}$ s.t. $\mathbf{Av}$ = $\lambda\mathbf{v}$. $\lambda$ is known as the eigenvalue associated with that eigenvector.
    - Think of $\mathbf{A}$ scaling space in the direction of $\mathbf{v}_i$ by $\lambda_i$.
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
- Determinant
  - The determinant is the product of all eigenvalues and can be thought of how much a matrix expands or contracts space.

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
- Similarly, for any matrix $\mathbf{X}$, we have $\nabla_{\mathbf{X}}\|\mathbf{X}\|_F^2=2 \mathbf{X}$.

- Chain Rule: $\frac{\partial y}{\partial x_i}=\frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i}+\frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i}+\cdots+\frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}$ and so 
$\nabla_{\mathbf{x}} y=\nabla_{\mathbf{x}}\mathbf{u} \nabla_{\mathbf{u}} y$ 

### Lagrangian Multipliers
- Suppose we want to maximize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = c$
  - Maximizing $f(\mathbf{x})$ is equivalent to finding the level set of $f(\mathbf{x})$ that is tangent to the level set $g(\mathbf{x}) = c$.
  - Equivalently, $\nabla f(\mathbf{x}) = \lambda\nabla g(\mathbf{x})$ (eqn 1).
  - To know what $\lambda$ is, we plug in $g(\mathbf{x}) = c$ (eqn 2) and we're done!
    - A pretty convoluted way of saying this is to define the Lagrangian $\mathcal{L}(\mathbf{x},\lambda)=f(\mathbf{x}) -\lambda(g(\mathbf{x}) -c)$
    - Then we have that $\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x},\lambda) = 0$ (eqn 1) and $\nabla_{\lambda}\mathcal{L}(\mathbf{x},\lambda) = 0$ (eqn 2)
  - Note that sometimes we get the constraint in the form $g(\mathbf{x}) \leq c$. A common trick is that when the constraint is binding, we set $g(\mathbf{x}) = c$.