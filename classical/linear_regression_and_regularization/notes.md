# Linear Regression and Regularization



## Parameter Estimation

- Model: $\mathbf{Y} \sim \mathbf{XB} + \pmb{\epsilon}, \pmb{\epsilon} \sim \mathcal{N}(0, \sigma^2\mathbf{I})$
- Assumptions
  - Linearity: $E[\mathbf{Y}] = \mathbf{XB} = \pmb{\mu}$
  - Normality: $\pmb{\epsilon}$ is normally distributed
  - Homoscedasticity: $\operatorname{Var}(\epsilon_i) = \sigma^2$
  - Independence: $\operatorname{Cov}(\epsilon_i, \epsilon_j) = 0$
- Maximum Likelihood Estimation
  - Typically, we have two unknown variables: $\mathbf{B}$ and $\sigma^2$. 
  - We note that the Log Likelihood is given by $\log \mathcal{L}(\mathbf{B}, \sigma^2) = c - \frac{n}{2}\log (\sigma^2) - \frac{1}{2}\frac{||\mathbf{Y} - \mathbf{XB}||^2}{\sigma^2}$
  - Since the $\hat{\pmb{\beta}}$ that maximizes this does not depend on $\sigma^2$, we focus on this first.
    - Notably, $\hat{\pmb{\beta}}$ minimizes $||\mathbf{Y} - \mathbf{XB}||^2$, i.e. **this is the same as minimizing MSE!**
    - The easiest way to do this is to think of this geometrically: 
      - Given that $||\mathbf{Y} - \mathbf{XB}||^2 = ||(\mathbf{I} - \mathbf{P_X})\mathbf{Y}||^2 + ||\mathbf{P_X Y} - \mathbf{XB}||^2$, $\mathbf{X\hat{B}} = \mathbf{P_X Y}$, where $\mathbf{P_X}$ is the projection matrix onto the column space of $\mathbf{X}$. 
      - If $\mathbf{X}^{\top}\mathbf{X}$ is full rank, we then get that $\mathbf{\hat{B}} = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}$
        - In the simple linear regression case, we get the formulae:
          - $\beta_1=\frac{\sum_{i=1}^n\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{\sum_{i=1}^n\left(x_i-\bar{x}\right)^2}$ (We can use the multivariate form to help remember this)
          - $\beta_0 =\bar{y}-(\beta_1 \bar{x})$
  - Now given our choice of $\hat{\mathbf{B}}$, we can then find the $\hat{\sigma^2}$ that maximizes the log likelihood, which is $\frac{\mathbf{e}^{\top}\mathbf{e}}{n}$, where $\mathbf{e} = \mathbf{Y - X\hat{{B}}}_{MLE}$
- Unbiased estimators
  - It is clear that $\hat{\mathbf{B}}$ is unbiased, but it turns out that $\hat{\sigma^2} = \frac{\mathbf{e}^{\top}\mathbf{e}}{n}$ is not!
  - Instead, an unbiased estimate of $\sigma^2$ is instead $\frac{\mathbf{e}^{\top}\mathbf{e}}{n-rank(\mathbf{X})}$ ([Proof](https://www2.stat.duke.edu/courses/Fall19/sta721/lectures/MLES/mles.pdf)).
- t-stats
  - t-stats indicate the significance of each $\hat{B_i}$. 
  - Formally, $\frac{\hat{B_i} - B_i}{SE(\hat{B_i})} \sim t_{n-rank(\mathbf{X})},$ where $SE(\hat{B_i}) = \sqrt{\frac{\mathbf{e}^{\top}\mathbf{e}}{n-rank(\mathbf{X})}[(\mathbf{X^{\top}X})^{-1}]_{ii}}$ 
    - In simple linear regression, we have that $SE(\hat{B_1}) = \sqrt{\frac{\mathbf{e}^{\top}\mathbf{e}}{n-2}\frac{1}{\sum_i(x_i - \bar{x})^2}}$ 
    - In regression libraries, we often set $\beta_i = 0$ as our null hypothesis.
    - Intuition:
      - The higher the errors in our model, the more uncertain we are about _every_ $\hat{B}_i$.
      - Since the first term is our unbiased estimate of $\sigma^2$, the higher the noise in the data generating process $\sigma^2$, the more uncertain we are about _every_ $\hat{B}_i$.
      - **Multicollinearity**: $[(\mathbf{X^{\top}X})^{-1}]_{ii}=\frac{1}{||\epsilon_i||^2},$ where $\epsilon_i$ is the projection of the $i^{th}$ column of $\mathbf{X}$ onto the other columns ([proof](https://math.stackexchange.com/questions/2624986/the-meaning-behind-xtx-1)). 
        - When columns are very correlated, we get high uncertainty for $\hat{B_i}.$
    - The [proof](https://www2.stat.duke.edu/courses/Fall19/sta721/lectures/SamplingDist/sampling.pdf) hinges on two parts:
      - $\hat{\mathbf{B}}\sim\mathcal{N}(\mathbf{B}, \sigma^2(\mathbf{X^{\top}X})^{-1})$ (using $\operatorname{Var}(\mathbf{Ax}) = \mathbf{A}(\operatorname{Var(\mathbf{x})})\mathbf{A}^{\top}$)
      - $\frac{\mathbf{e}^{\top}\mathbf{e}}{\sigma^2} \sim \chi^2_{n-rank(\mathbf{X})}$ (refer to link for more details)
- $R^2$
  - SST (sum of squares total) = TSS = $\sum_i (y_i-\bar{y})^2$
  - SSR (sum of squares regression) = ESS (explained sum of squares) = $\sum_i (\hat{y}_i-\bar{y})^2$
  - SSE (sum of squares error) = RSS (residual sum of squares) = $\sum_i (y_i-\hat{y})^2$
  - SST = SSR + SSE
  - $R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}$ (proportion of variance explained)
    - In simple linear regression, we can show that $R^2$ is the square of the sample Pearson correlation coefficient $r_{x y}=\frac{\sum_{i=1}^n\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{\sqrt{\sum_{i=1}^n\left(x_i-\bar{x}\right)^2} \sqrt{\sum_{i=1}^n\left(y_i-\bar{y}\right)^2}}$ ([proof](https://math.stackexchange.com/questions/129909/correlation-coefficient-and-determination-coefficient)), which yields interesting symmetric properties.

## Optimization 
### Methodologies to Get $\mathbf{B}$
- Moore-Penrose Pseudo-Inverse
  - Instead of calculating $(\mathbf{X}^{\top}\mathbf{X})^{-1}$, which takes around $O(n^{2.4})$ to $O(n^3)$, we use SVD to calculate the Moore-Penrose pseudo-inverse of $\mathbf{X}, \mathbf{X}^+$
    - $\mathbf{X}^+ = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}$
    - SVD takes $O(np^2)$ (assuming $p< n$).
- Gradient Descent
  - If $\mathcal{L}(\mathbf{B}) = ||\mathbf{y}-\mathbf{X}\mathbf{B}||^2$, then $\nabla_{\mathbf{B}}\mathcal{L}(\mathbf{B}) = 2\mathbf{X}^{\top}\mathbf{X}\mathbf{B}-2\mathbf{X^{\top}y}$
  - Full Batch
  - Mini Batch
  - Stochastic Gradient Descent (SGD)

## Testing

* Model Selection
  * Since every additional feature would increase $R^2$, how do we combat over-fitting?
  * [AIC and BIC](../probability_and_info_theory/notes.md)
  * Adjusted $R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$
  * Chow Test (F-test)
    * If a reduced model $R$ with $p_R$ parameters is nested in the full model $F$ with $p_F$ parameters, then 
    * $\frac{(SSE(R) - SSE(F))/\Delta p}{\hat{\sigma}^2_F} \sim F_{\Delta p, n-p_2}$ ($\hat{\sigma}^2_F = \frac{SSE_F}{n-p_F}$)
* Homoskedasticity 
  * Detection: Scale/Location plot
  * Remedies: Robust Standard Errors
    * White's Estimator (heteroskedastic, but still diagonal)
      * $\hat{\operatorname{Var}(\hat{\beta})} =\frac{\mathbf{1}}{\mathbf{n}}\left[\frac{\mathbf{1}}{\mathbf{n}}\left(\mathbf{X}^{\prime} \mathbf{X}\right)\right]^{-\mathbf{1}}\left[\frac{\mathbf{1}}{\mathbf{n}} \mathbf{X}^{\prime} \hat{\pmb{\Omega}} \mathbf{X}\right]\left[\frac{\mathbf{1}}{\mathbf{n}}\left(\mathbf{X}^{\prime} \mathbf{X}\right)\right]^{-\mathbf{1}}$, $\hat{\pmb{\Omega}} =\operatorname{diag}\left(\hat{\mathrm{u}}_1^2, \hat{\mathrm{u}}_2^2, \cdots, \hat{\mathrm{u}}_{\mathrm{n}}^2\right)$ from OLS errors.
    * Newey-West Estimator (HAC Standard Errors, accounts for autocorrelation)
    * Driskoll and Kraay Standard Errors (Accounts for temporal and cross-sectional dependencies)
* Normality - QQ plot
* Outliers - Cook's Distance
  * Outliers matter when they influence our model. The cook's distance sums up the squared changes of predictions. $D_i=\frac{\sum_{j=1}^n\left(\widehat{y}_j-\widehat{y}_{j(i)}\right)^2}{rank(\mathbf{X}) s^2}$
* Multicollinearity 
  * Detection
    * VIF = $\frac{\operatorname{Var}(B_j)_{full}}{\operatorname{Var}(B_j)_{reduced}}$, where the reduced model only has parameter $j$.
    * Another way to detect multicollinearity is with condition numbers, the ratio between the largest and smallest eigenvalue of $\mathbf{X^{\top}X}$. To better understand this relationship, [this discussion](https://stats.stackexchange.com/questions/20386/understanding-condition-index-used-for-finding-multicollinearity) is helpful.
  * Remedies: [Dimensionality Reduction](../dimensionality_reduction/notes.md)
    * Removing correlated variables
    * PCA
    * Partial Least Squares

## Polynomial Regression
- When we include higher degrees of our input features

## Regularization
- Due to the [bias-variance tradeoff](../statistical_learning_theory/notes.md), we may opt for decreased variance at the cost of increased bias. 
- Ridge Regression
  - $\mathcal{L}(\mathbf{B}) = ||\mathbf{Y} - \mathbf{XB}||^2_2 + \lambda||\mathbf{B}||^2_2$
  - $\hat{\mathbf{B}} = (\mathbf{X}^{\top}\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^{\top}\mathbf{y}$
  - $\operatorname{MSE}(\mathbf{B}) = E[||\mathbf{B}-\mathbf{\hat{B}}||^2] = \sigma^2\operatorname{tr}\left[\left(\mathbf{X}^T \mathbf{X}\right)^{-1}\right] =\sigma^2 \sum_{j=1}^p \lambda_j^{-1}$
    - In other words, if we have small eigenvalues, our MSE is high. 
    - However, by adding $\lambda \mathbf{I}$ to $\mathbf{X}^{\top}\mathbf{X}$, we change $\frac{1}{\lambda_j}$ to $\frac{1}{\lambda + \lambda_j}$ (note that there's no free lunch here because we increased bias).
  - Optimization
    - Since we no longer can use the Moore-Penrose Inverse, it is common practice to use the Cholesky decomposition to compute $(\mathbf{X}^{\top}\mathbf{X} + \lambda \mathbf{I})^{-1}$.
- Lasso Regression
  - $\mathcal{L}(\mathbf{B}) = ||\mathbf{Y} - \mathbf{XB}||^2_2 + \lambda||\mathbf{B}||^2_1$
  - Optimization
    - There is no closed form solution for $\mathbf{B}$, so we solve this with a combination of [coordinate descent and subgradients](https://medium.com/@msoczi/lasso-regression-step-by-step-math-explanation-with-implementation-and-example-c37df7a7dc1f). 
    - Looking at the form of the subgradients provides inspiration for why Lasso promotes sparsity.
- Elastic Net
  - $\mathcal{L}(\mathbf{B}) = ||\mathbf{Y} - \mathbf{XB}||^2_2 + \lambda_1||\mathbf{B}||^2_2 + \lambda_2||\mathbf{B}||^2_1$
- Bayesian Perspective
  - Note that the loss functions above permit Bayesian interpretations, where if our prior on $\mathbf{B}$ is a Laplace/Normal zero-mean distribution, we get the lasso/ridge loss functions when computing the posterior likelihood. 
  - This provides additional insight as to why ridge shrinks parameters and lasso induces sparsity. 
- Lagrangian Perspective
  - Note that per our discussion around the [Lagrangian](../linear_algebra_and_calculus/notes.md), our loss functions above are the "dual" interpretation of the respective optimization problems.
  - We can also think of these problems in their "primal" form, e.g. Minimize $||\mathbf{Y} - \mathbf{XB}||^2_2$ subject to $||\mathbf{B}||^2_2 \leq c$.
  - This helps motivate the effect of these methods on $\mathbf{B}$. In particular, ridge regression shrinks coefficients to 0, while lasso regression induces sparsity. ![regularization.png](regularization.png)