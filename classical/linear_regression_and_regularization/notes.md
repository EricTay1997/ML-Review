# Linear Regression and Regularization

## Parameter Estimation

- $\mathcal{L}(\pmb{\beta}) = ||\mathbf{y}-\mathbf{X}\pmb{\beta}||^2$
  - $\nabla_{\pmb{\beta}}\mathcal{L}(\pmb{\beta}) = 2\mathbf{X}^{\top}\mathbf{X}\pmb{\beta}-2\mathbf{Xy}$ (gradient descent)
  - Which solves to $\pmb{\beta} = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}$


## Optimization 
### To add details regarding SVD and O(n^3) if this hasn't been done


## Regularitzation
### Bayesian perspective


## Testing

* F -tests
* Homoskedasticity - scale/location plot
* Normality - QQ plot
* Outliers - cook's distance
* Multicollinearity - VIF
  * remove variables, linearly combine, PCA/PLS
* Robust Standard Errors
  * White's Estimator
  * Newey-West Estimator (HAC Standard Errors)
  * Driskoll and Kraay Standard Errors