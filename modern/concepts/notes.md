# Concepts (Misc)

## The Curse of Dimensionality
- As the number of dimensions increase, the "space" between training points increases significantly

## Local Constancy and Smoothness Regularization
- Many ML models operate under the prior that the function we want to learn should not change very much within a small region. 
- Consider now a checkerboard - are these assumptions sufficient?
- Is it possible to represent a complicated function efficiently, and is it possible for an estimated function to generalize well to new inputs? Yes!
  - A large number of regions $O(2^k)$ can be defined with $O(k)$ examples, so long as we introduce dependencies between the regions through additional assumptions about hte underlying data-generating process.
  - Task-specific assumptions could also be appropriate, such as assuming that the target function is periodic.

## Manifold Learning
- The manifold interpretation is that natural data forms lower-dimensional manifolds in its embedding space. 
  - I view this as the motivation for why we "encode" data into a lower-dimensional space. 
- [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) mentions the inverse: All $n$-dimensional manifolds can be untangled in $2n+2$ dimensions. 
  - I wonder if this is good intuition for first increasing the number of hidden units and then going back down? Like the MLP layers of a transformer...

## Optimization
- Apologies for the bolding inconsistency in this section, I'll clean this up if time permits.
- Newton's method
  - Suppose we want to minimize $f(\mathbf{x})$.
  - We use the following update rule: $\mathbf{x}_{k+1}=\mathbf{x}_k-H\left(\mathbf{x}_k\right)^{-1} \nabla f\left(\mathbf{x}_k\right)$
    - $f\left(x_k+\Delta x\right) \approx f\left(x_k\right)+\nabla f\left(x_k\right)^{\mathrm{T}} \Delta x+\frac{1}{2} \Delta x^{\mathrm{T}} H \Delta x$
    - $\nabla f\left(x_k+\Delta x\right) \approx \nabla f\left(x_k\right)+H \Delta x$
    - Setting the gradient to be zero gives our update function.
  - When the loss function is nonconvex, this method could instead get us to local **maxima**
    - Regularization methods are used to prevent this
  - The time complexity is understandably dominated by inverting the Hessian, which we try to address in Quasi-Newton methods
- Fisher's scoring replaces $H$ with it's expectation, the negative of the Fisher's Information matrix. The advantage of doing so is that this is always positive semi-definite and therefore can aid convergence issues.
- Quasi-Newton methods
  - Quasi-Newton methods avoid computing the inverse of the Hessian by _estimating it through iteration_. Concretely, it uses the following update step:
  - $B_{k+1}\left[\mathbf{x}_{k+1}-\mathbf{x}_k\right]=\nabla f\left(\mathbf{x}_{k+1}\right)-\nabla f\left(\mathbf{x}_k\right)$
    - The intuition here is that we don't want to compute the inverse,  so we iterate on the equation before that.
- Goodfellow's chapter 8 has a lot of interesting discussion here