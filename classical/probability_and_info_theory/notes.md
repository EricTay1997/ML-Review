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
  * Suppose $\pmb{\mu} \stackrel{\text { def }}{=} E_{\mathbf{x} \sim p}[\mathbf{x}]$, then
  * $\pmb{\Sigma} \stackrel{\text { def }}{=} \operatorname{Cov}_{\mathbf{x} \sim p}[\mathbf{x}]=E_{\mathbf{x} \sim p}\left[(\mathbf{x}-\pmb{\mu})(\mathbf{x}-\pmb{\mu})^{\top}\right]$ is the covariance matrix. 
    * $\Sigma_{ij} = \operatorname{Cov}(x_i, x_j)$ (Remember that $x_i$ and $x_j$ are RVs) 
    * This is closely related to the correlation matrix, where $\Sigma_{ij} = \frac{\operatorname{Cov}(x_i, x_j)}{\sigma(x_i)\sigma(x_j)}$
  * These formulae are relatively easy to apply given a data generating process with specified parameters. In practice, we usually estimate parameters from data:
    * (To add)
    * Also add why a sample covariance matrix is positive semidefinite by definition
* CLT
  * The CLT states that for $X_i$ with mean $\mu$ and variance $\sigma^2$, $\bar{X}_n=\frac{X_1+\ldots+X_n}{n} \rightarrow \sim N\left(\mu, \frac{\sigma^2}{n}\right)$ and hence $\frac{\bar{X}_n-\mu}{\sigma / \sqrt{n}} \sim N(0,1)$
* Moment Generating Functions 
  * The MGF of $X$ is $E[\exp(tX)]$.
  * Given that $e^{tX} = 1 + tX + \frac{t^2X^2}{2!} + \ldots$, we can do consecutive differentiations to get the moments of a R.V.
  
## Information Theory
* Principles
  * Less likely events should have higher information content.
  * Independent events should have additive information.
* Self-information of event $X = x$ is $I(x) = -\log P(x)$
  * Negative: Each event has positive information
* **Shannon entropy**: The uncertainty in an entire probability distribution
  * $H(P)=\mathbb{E}_{\mathrm{x} \sim P}[I(x)]=-\mathbb{E}_{\mathrm{x} \sim P}[\log P(x)]$
  * If $P(x)$ can take on many values, this is high. If not, this is low. 
* **KL divergence**: The difference between two probability distributions $P(x)$ and $Q(x)$
  * $D_{\mathrm{KL}}(P \| Q)=\mathbb{E}_{\mathbf{x} \sim P}\left[\log \frac{P(x)}{Q(x)}\right]=\mathbb{E}_{\mathbf{x} \sim P}[\log P(x)-\log Q(x)]$
  * This is guaranteed to be nonnegative is 0 $\iff$ the distributions are equal. 
  * This is not symmetric and is therefore not a true distance metric. When then should we use $D_{\mathrm{KL}}(P \| Q)$ vs $D_{\mathrm{KL}}(Q \| P)$?
    * $D_{\mathrm{KL}}(P \| Q):$ $P$ is in numerator. Intuitively, when $P$ is large, we want $Q$ to be large too.  
    * $D_{\mathrm{KL}}(Q \| P):$ $P$ is in denominator. Intuitively, when $P$ is small, we want $Q$ to be small too. 
* **Cross Entropy**: $H(P,Q)=-\mathbb{E}_{\mathbf{x} \sim P}\log Q(x)=H(P)+D_{\mathrm{KL}}(P \| Q)$
  * If we fix $P$, and we minimize the cross-entropy with respect to $Q$, this is the same as minimizing KL divergence.

