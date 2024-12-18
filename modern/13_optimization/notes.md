# Optimization

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


## Optimization
- Stochastic Gradient Descent (SGD) is mainly done for computational reasons. The expectation of a minibatch is still unbiased.

### Numerical Analysis Methods

## Regularization
### See Statistical Learning Theory

## Additional Tricks