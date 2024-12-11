# Statistical Learning Theory

* Capacity
  * One way to quantify the capacity of a binary classifier is the VC dimension.
* Generalization error
  * Roughly, the difference between training and test error is of order $\sqrt{\frac{d_{vc}\log n}{n}}$
    * I.e. generalization error decreases with lower model capacity / more data.
* Bias
  * The bias of an estimator $\hat{\pmb{\theta}}_m=\mathbb{E}\left(\hat{\pmb{\theta}}_m\right)-\pmb{\theta}_{true}$, where the expectation is over the data (seen as samples from a RV)
* MSE & Bias-Variance Tradeoff
  * $\operatorname{MSE}= \mathbb{E}\left[\left(\hat{\theta}_m-\theta\right)^2\right] \\ =\operatorname{Bias}\left(\hat{\theta}_m\right)^2+\operatorname{Var}\left(\hat{\theta}_m\right)$
  * Desirable estimators keep MSE low, and therefore balance bias and variance (bias-variance tradeoff)
  * Note a similar decomposition of the data generating process $y = f(x) + \epsilon$, $\operatorname{Var}(\epsilon) = \sigma^2$
    * $\operatorname{MSE} = \mathbb{E}[(y-\hat{y})^2] = \operatorname{Bias}(\hat{y})^2+\operatorname{Var}(\hat{y})+\sigma^2$, where we term $\sigma^2$ as the irreducible error.
* Consistency
  * $\operatorname{lim}_{m \rightarrow \infty} \hat{\theta}_m=\theta$
  * Biased but consistent: 
    * Suppose $x \sim \mathcal{N}(\mu, \sigma^2)$, and we estimate $\hat{\sigma}^2 = \frac1n\sum^n (x_i - \bar{x})^2$
    * Then $\mathbb{E}[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2$
    * This estimator is consistent because $\operatorname{Var}(\hat{\sigma}^2) = \frac{2\sigma^4 (n-1)}{n^2}$
      * Proof: Using [Cochran's theorem](https://en.wikipedia.org/wiki/Cochran%27s_theorem#Sample_mean_and_sample_variance), we have that $\frac1n\sum^n (x_i - \bar{x})^2 \sim \frac{\sigma^2}{n}\chi^2_{n-1},$ and $\operatorname{Var}[\chi^2_{n-1}] = 2(n-1)$
      * Intuition for $n-1$ is the same as bias for sample variance - using $\bar{x}$ instead of $\mu$ naturally reduces the estimated quantity.
  * Unbiased but not consistent:
    * $\hat{\mu} = x^{(1)}$
* Maximum Likelihood Estimation
  * $\pmb{\theta}_{MLE} = \operatorname{argmax}_{\pmb{\theta}}p(\mathbf{x}; \pmb{\theta})=\underset{\pmb{\theta}}{\arg \max } \mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}} [\log p_{\text {model }}(\mathbf{x} ; \pmb{\theta})]$
    * Note that minimizing this is the same as minimizing $D_{\mathrm{KL}}\left(\hat{p}_{\text {data }} \| p_{\text {model }}\right)=\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}}\left[\log \hat{p}_{\text {data }}(\mathbf{x})-\log p_{\text {model }}(\mathbf{x})\right]$
    * Which is also the same as minimizing the cross-entropy loss $-\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}}\left[\log p_{\text {model }}(\mathbf{x})\right]$
    * Note that in linear regression, this is the same as minimizing MSE.
  * Properties
    * Under appropraite conditions, $\pmb{\theta}_{MLE}$ is consistent:
      * $p_{data}$ must lie within the model family $p_{model}(\pmb{\theta})$
      * The true distribution of $p_{data}$ must correspond to exactly one value of $\pmb{\theta}$
* Efficiency (let's switch to scalars for simplicity)
  * Fisher Information $\mathcal{I}(\theta)=\mathrm{E}\left[\left.\left(\frac{\partial}{\partial \theta} \log f(X ; \theta)\right)^2 \right\rvert\, \theta\right]$ is the variance of the score.
    * Intuitively, when the score has high variance, i.e. the likelihood function varies significantly wrt $\theta$, the distribution is highly peaked. In other words, the data $\mathbf{x}$ conveys _more_ information about $\theta$ than if the pdf was flatter. 
  * The Cramér–Rao lower bound for the scalar unbiased case is then given by $\operatorname{Var}(\hat{\theta})\geq\frac{1}{I(\theta)}$.
    * Again, this makes sense. High $\mathcal{I}(\theta) \rightarrow$ Peaky distribution $\rightarrow$ Smaller lower bound, i.e. our precision for our estimator is higher. 
  * Now, **Efficiency** for an unbiased estimator is then defined as $e(\hat{\theta}) = \frac{I(\theta)^{-1}}{\operatorname{Var}(\hat{\theta})}$. This is bounded above by 1 and it gives a measure of how precise the estimator is (relative to the theoretical maximum precision).
  * Note that for large $n$ (and under certain conditions), the maximum likelihood estimator is both consistent and efficient (achieves the lower bound). 
    * I.e. No consistent estimator has a lower MSE than it, and hence this is an appropriate estimator.
    * **However**, for smaller $n$, regularization may useful to obtain a biased estimator with lower variance