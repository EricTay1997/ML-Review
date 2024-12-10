# Training

## Commonly Used Functions
- Softplus: $\zeta(x) = \log(1+\exp(x))$, a smoothed version of $x^+ = \max(0, x)$
- Sigmoid: $\frac{d}{d x} \zeta(x)=\sigma(x) = \frac{1}{1+\exp (-x)}$
- Additional properties of softplus and sigmoid:
  - $ \zeta(x)=\int_{-\infty}^x \sigma(y) d y$
  - $ \frac{d}{d x} \sigma(x)=\sigma(x)(1-\sigma(x))$
  - $ 1-\sigma(x)=\sigma(-x)$
  - $ \log \sigma(x)=-\zeta(-x)$
  - $ \forall x \in(0,1), \sigma^{-1}(x)=\log \left(\frac{x}{1-x}\right)$
  - $ \zeta^{-1}(x)=\log (\exp (x)-1)$
  - $ \zeta(x)-\zeta(-x)=x$
- 

## Optimization

### Numerical Analysis Methods

## Regularization

## Additional Tricks