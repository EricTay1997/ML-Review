# Probability and Information Theory

## Probability

* Joint Probability: $ \int \int p(x,y) dx dy = 1$
* Marginal Probability: $P(X = x) = \sum_y P(X = x, Y = y) = \int p(x,y) dy$
* Conditional Probability: 
  * Bayes: $P(X = x|Y = y) = \frac{P(Y = y, X = x)}{P(Y=y)} = \frac{P(Y = y|X = x)P(X = x)}{\sum_{x}(P(Y = y|X=x)P(X=x))}$
    * Or in the continuous case: $p(x|y) = \frac{p(x,y)}{p(y)}$
  * Chain Rule: $P(X = x, Y = y, Z = z) = P(X = x | Y = y, Z = z)P(Y = y | Z = z)P(Z = z)$
  * Conditional Independence: $P(X = x, Y = y | Z = z) = P(X = x | Z = z)P(Y = y | Z = z)$
  * Conditional Expectation: $E[X|Y=y]=\int xp(x|y)dx$
    * We often see the law of iterated expectation: $E[X] = E[E[X|Y]]$. This is essentially saying that $E[X] = \int p(y)E[X|Y=y]dy.$
* Transforming pdfs
  * Suppose $Y = g(X)$
  * Then $p_y(y)=p_x\left(g^{-1}(y)\right)/\left|\frac{\partial g(x)}{\partial x}\right|$ or $p_y(\mathbf{y})=p_x\left(g^{-1}(\mathbf{y})\right)/\left|det(\frac{\partial g(\mathbf{x})}{\partial \mathbf{x}})\right|$
    * Intuition, if $y = 2x$, then we need to half the pdf for it to sum to integrate to 1.
* Expectations and Variance
  * $E_{x \sim p}[x]=\sum_x x P(x) = \int f(x) p(x) d x$
  * $E_{x \sim p}[f(x)]=\sum_x f(x) P(x) = \int f(x) p(x) d x$ (Law of the unconscious statistician)
  * $\operatorname{Var}[X]=E\left[(X-E[X])^2\right]=E\left[X^2\right]-E[X]^2$
  * $\operatorname{Var}_{x \sim p}[f(x)]=E_{x \sim p}\left[f^2(x)\right]-E_{x \sim p}[f(x)]^2$
  * $\operatorname{Cov}(X, Y) = E[(X-E[X])(Y-E[Y])]$
  * $\operatorname{Cov}(f(X), g(Y)) = E[(f(X)-E[f(X)])(g(Y)-E[g(Y)])]$
  * Random Vector $\mathbf{x} = (X_1, \ldots, X_p)^{\top}$
    * Suppose $\pmb{\mu} \stackrel{\text { def }}{=} E_{\mathbf{x} \sim p}[\mathbf{x}]$, then
    * $\pmb{\Sigma} \stackrel{\text { def }}{=} \operatorname{Cov}_{\mathbf{x} \sim p}[\mathbf{x}]=E_{\mathbf{x} \sim p}\left[(\mathbf{x}-\pmb{\mu})(\mathbf{x}-\pmb{\mu})^{\top}\right]$ is the covariance matrix. 
      * $\Sigma_{ij} = \operatorname{Cov}(X_i, X_j)$ 
      * This is closely related to the correlation matrix, where $\Sigma_{ij} = \frac{\operatorname{Cov}(X_i, X_j)}{\sigma(X_i)\sigma(X_j)}$
      * $\pmb{\Sigma}$ is positive semi-definite: 
        * $\mathbf{v^{\top}}\pmb{\Sigma}\mathbf{v} = \sum_i\sum_j v_iv_jCov(X_i, X_j)$, which we recognize to be the formula of $\operatorname{Var}(\mathbf{v^{\top}x}) \geq 0.$
        * In particular, if $\mathbf{v^{\top}}\pmb{\Sigma}\mathbf{v} = 0$, then there exists $X_i$ which is a linear combination of the other $X_j$s.
  * These formulae are relatively easy to apply given a data generating process with specified parameters. In practice, we usually estimate parameters from data:
    * Suppose $X_i$ are iid and $E[X] = \mu$ and $\operatorname{Var}(X) = \sigma^2$
      * Then $\hat{\mu} = \bar{X}$ is unbiased (Of course, other estimators like $X_i$ are also unbiased but this estimator has lower variance, also reference CLT).
      * $\hat{\sigma^2} = \frac{\sum_i\left(X_i-\mu\right)^2}{n}$ if $\mu$ is known is unbiased.
      * $\hat{\sigma^2} = \frac{\sum_i\left(X_i-\bar{X}\right)^2}{n-1}$ if $\mu$ is unknown is unbiased. 
        * Intuition: $\sum_i(X_i-\bar{X})^2 \leq \sum_i(X_i-\mu)^2$, and so we need to inflate this. 
        * Proof: 
          * $E(X^2) = \mu^2 + \sigma^2$
          * $E(\bar{X}^2) = \mu^2 + \frac{\sigma^2}{n}$
            * Since $\operatorname{Var}(\bar{X}) = \frac{\sigma^2}{n}$
            * $E(\bar{X}^2) = E(\frac{\sum_i X_i^2}{n^2} + \frac{\sum_i\sum_{j\neq i} X_iX_j}{n^2}) = \frac{n(\mu^2 + \sigma^2) + (n^2 - n)\mu^2}{n^2} = \mu^2 + \frac{\sigma^2}{n}$
          * Now $E((X_i - \bar{X})^2) = E(X_i^2 - 2X_i\bar{X} + \bar{X}^2) = E(\frac{n-2}{n}X_i^2 - 2\frac{n-1}{n}X_iX_j + \bar{X}^2)$
          * Comparing coefficients, we see that the $\mu^2$ terms cancel out, so we're left with $(\frac{n-2+1}{n})\sigma^2$ as desired.
        * A kinda cool fact is that using [Cochran's theorem](https://en.wikipedia.org/wiki/Cochran%27s_theorem#Sample_mean_and_sample_variance), we have that $s^2 = \frac{1}{n-1}\sum^n (x_i - \bar{x})^2 \sim \frac{\sigma^2}{n-1}\chi^2_{n-1},$ 
    * Note that above, we did not specify the _distribution_ of $X$, but rather just its mean and variance. Now consider the multivariate linear regression case, where we switch conventions from $X$ to $\mathbf{Y}$.
      * $\mathbf{Y = XB} + \pmb{\epsilon}, \pmb{\epsilon} \sim (0, \sigma^2\mathbf{I})$, and $\mathbf{B}$ and $\pmb{\epsilon}$ are unknown. Then:
        * $E(\hat{\mathbf{B}}_{MLE}) = \mathbf{B}$
        * $E(\frac{\mathbf{e}^{\top}\mathbf{e}}{n-rank(\mathbf{X})}) = \sigma^2$, where $\mathbf{e} = \mathbf{Y - X\hat{{B}}}_{MLE}$
        * We include these formulae for completeness, and the proof for these can be found in the [linear regression notes](../06_linear_regression_and_regularization/notes.md).
    * **Sample Covariance Matrix**
      * Suppose we have data $\mathbf{X} \in \mathbb{R}^{n \times p}$, with $n$ rows and $p$ features. 
      * We view each of these features as a random variable $X_j$, which together form the random vector $\mathbf{x} = (X_1, \ldots, X_p)^{\top}$
        * We can view $\mathbf{X}$ as $n$ draws from $\mathbf{x}$, and we denote each draw (row) by $\mathbf{x}_i \in \mathbb{R}^p$, and their sample mean by $\bar{\mathbf{x}}$.
      * The sample covariance matrix $\mathbf{S} \in \mathbb{R}^{p \times p}$ is an unbiased estimate of $\pmb{\Sigma} = \operatorname{Cov}(\mathbf{x})$. I.e. $E[S_{jk}] = \operatorname{Cov}(X_j, X_k)$.
        * Concretely, 
          * $\mathbf{S} = \mathbf{Z^{\top}Z},$ where $\mathbf{Z}$ is $\mathbf{X}$ with demeaned columns.
          * $S_{jk} = S_{kj} :=\frac{1}{n-1} \sum_{i=1}^n[\left(X_{i j}-\bar{X}_j\right)\left(X_{i k}-\bar{X}_k\right)]$ (To avoid confusion, we're dealing with scalars here)
          * Note that this extends the unbiased estimator for population variance above to the unbiased estimator of $\operatorname{Cov}(X_j, X_k)$.
          * Note that this is also very similar to the $\mathbf{X^{\top}X}$ we often use in regression, except that $\mathbf{S}$ notably demeans the columns.
        * Alternatively, $\mathbf{S} = \frac{1}{n-1} \sum_{i=1}^n\left(\mathbf{x}_i-\bar{\mathbf{x}}\right)\left(\mathbf{x}_i-\bar{\mathbf{x}}\right)^{\top}$,  where $\vec{x}_i \in \mathbb{R}^{p}.$ It is helpful to remember that $(\mathbf{x}_i-\bar{\mathbf{x}})$ is the $i^{th}$ row vector of $\mathbf{Z}$, defined as $\mathbf{X}$ with demeaned columns. 
          * This form is very helpful to see why $\mathbf{S}$ is positive semi-definite.
          * For real data, it is likely that $\mathbf{X}$ is rank $p$. This implies that $\mathbf{S}$ is often positive-definite.
            * Proof: Suppose that $\mathbf{v^{\top}Sv} = 0$.
            * $||\mathbf{v}^{\top}(\mathbf{x}_i-\bar{\mathbf{x}})||^2 = 0$ $\forall$ $i$.
            * Let $\mathbf{y}_i = \mathbf{x}_i-\bar{\mathbf{x}}$. Now given that $\mathbf{y}_i$s span $\mathbb{R}^p$, let $\mathbf{v} = \sum_i \alpha_i \mathbf{y}_i$.
            * We now have that $\mathbf{v}^{\top}\mathbf{v}=0$ since $\mathbf{v}^{\top}\mathbf{y}_i = 0$ $\forall$ $i$, which is a contradiction. 
* CLT
  * The CLT states that for $X_i$ with mean $\mu$ and variance $\sigma^2$, $\bar{X}_n=\frac{X_1+\ldots+X_n}{n} \rightarrow \sim N\left(\mu, \frac{\sigma^2}{n}\right)$ and hence $\frac{\bar{X}_n-\mu}{\sigma / \sqrt{n}} \sim N(0,1)$
* Moment Generating Functions 
  * MGFs uniquely identify probability distributions.
  * The MGF of $X$ is $E[\exp(tX)]$.
  * Given that $e^{tX} = 1 + tX + \frac{t^2X^2}{2!} + \ldots$, we can do consecutive differentiations to get the moments of a R.V.
  * The Characteristic Function of $X$ is $E[\exp(itX)]$, which is defined on the entire Real line (as opposed to the MGF). We tradeoff simplicity for this advantage.
  
## Information Theory
* Principles
  * Less likely events should have higher information content.
  * Independent events should have additive information.
* Self-information of event $X = x$ is $I(x) = -\log P(x)$
  * Negative: Each event has positive information, and rare events have higher information content.
* **Shannon entropy**: The uncertainty in an entire probability distribution
  * $H(P)=\mathbb{E}_{\mathrm{x} \sim P}[I(x)]=-\mathbb{E}_{\mathrm{x} \sim P}[\log P(x)]$
  * If $P(x)$ can take on many values, this is high. If not, this is low. 
* **Perplexity**: $2^{H(P)} = \prod_x p(x)^{-p(x)}$ for the discrete case.
  * This is meant to be more intuitive - for a uniform distribution the perplexity is the number of unique outcomes. 
  * Hence, a distribution of perplexity 4 is as random as rolling a 4-sided die. 
* **KL divergence**: The difference between two probability distributions $P(x)$ and $Q(x)$
  * $D_{\mathrm{KL}}(P \| Q)=\mathbb{E}_{\mathbf{x} \sim P}\left[\log \frac{P(x)}{Q(x)}\right]=\mathbb{E}_{\mathbf{x} \sim P}[\log P(x)-\log Q(x)]$
  * This is guaranteed to be nonnegative is 0 $\iff$ the distributions are equal. 
  * This is not symmetric and is therefore not a true distance metric. When then should we use $D_{\mathrm{KL}}(P \| Q)$ vs $D_{\mathrm{KL}}(Q \| P)$?
    * $D_{\mathrm{KL}}(P \| Q):$ $P$ is in numerator. Intuitively, when $P$ is large, we want $Q$ to be large too.  
    * $D_{\mathrm{KL}}(Q \| P):$ $P$ is in denominator. Intuitively, when $P$ is small, we want $Q$ to be small too. 
    * Note: We often do $D_{\mathrm{KL}}\left(\hat{p}_{\text {data }} \| p_{\text {model}}\right)$
* **Cross Entropy**: $H(P,Q)=-\mathbb{E}_{\mathbf{x} \sim P}\log Q(x)=H(P)+D_{\mathrm{KL}}(P \| Q)$
  * If we fix $P$, and we minimize the cross-entropy with respect to $Q$, this is the same as minimizing KL divergence.
  * The minimum cross entropy is when $P = Q$ and the value is then the self-entropy of either distributions.
  * Also see [logistic/softmax regression](../07_naive_bayes_and_logistic_regression_and_glms/notes.md).

