# Support Vector Machines

## Kernel Regression

## 

- Prediction = $\operatorname{sign}(\mathbf{w}^{\top}\mathbf{x}+b)$
- We can rewrite $\mathbf{w}^{\top}\mathbf{x}+b = b + \sum_i^n \alpha_i\mathbf{x}^{\top}\mathbf{x}^{(i)}$
- The kernel trick (why is this ok?)
  - Prediction = $\operatorname{sign}(f(\mathbf{x})),$ where $f(\mathbf{x})= b + \sum_i^n\alpha_ik(\mathbf{x}, \mathbf{x}^{(i)})$ and $k(\mathbf{x}, \mathbf{x}^{(i)})$ = $\phi(\mathbf{x})^{\top}\phi(\mathbf{x}^{(i)})$
  - Why is this useful?
    - It enables us to learn models that are nonlinear as a function of $\mathbf{x}$ using convex optimization techniques that are guaranteed to converge efficiently. 
    - The kernel function $k$ admits an implementation that is significantly more computationally efficient than first constructing the $\phi(\mathbf{x})$ vectors.
  - The Gaussian kernel can be thought of increasing the weight of points close (in terms of Euclidean distance) to $\mathbf{x}$.
- A drawback here is that for _every_ prediction, we need to consider all the training samples.
  - However, if most $\alpha_i = 0$, our computation is sped up significantly. $i$ s.t. $\alpha_i \neq 0$ are known as our support vectors. 