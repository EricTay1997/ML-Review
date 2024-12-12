# Testing and Metrics

* Metrics
  * Binary classification
    * ![Classification Statistics](classification_statistics.png)
    * Additionally, we have the following definitions:
      * Recall = TPR = Sensitivity
      * FPR = 1 - Specificity 
    * In cases where both precision and recall are equally important, we can try to optimize $F_1=2 * \frac{\text { precision * recall }}{\text { precision + recall }}$
  * Some other terms (that can be used both in/out of binary classification)
    * The **p-value** is the probability of observing the value of the calculated test statistic under the null hypothesis assumptions. We usually compare the observed p-value to a chosen level of alpha.
    * Alpha: The probability of a type 1 (false positive) error := 1 - Confidence Level.
    * Beta: The probability of a type 2 (false negative) error := 1 - Power. With higher power, we generally get higher sensitivity/recall. However, we get lower specificity. 
  * Losses
    * MSE
    * MAE
* Tests
  * z-test
    * We use the z-test when the population variance $\sigma^2$ is known. 
    * Formally, due to CLT, we say that $z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1)$
  * Student's t-test
    * We use the t-test when the population variance $\sigma^2$ is unknown, and instead estimate this with $s^2$
    * $t = \frac{\bar{x}-\mu_0}{s/\sqrt{n}} \sim t_{n-1}$
  * Welch's t-test
    * [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test) is used when the two samples have unequal variance. 
  * Chi-squared test statistic
    * This is often used in the analysis of contingency tables to see whether two categorical variables are independent of each other.
    * Importantly, we say that $\sum_i \frac{(O_i-E_i)^2}{E_i} \sim \chi^2,$ and [this example](https://en.wikipedia.org/wiki/Chi-squared_test#Example_chi-squared_test_for_categorical_data) is particularly instructive for how we might use this. 
* Multiple Hypothesis Testing
  * When $m > 1$ tests are done, the probability of rejecting the null hypothesis by chance alone can be too high. This is because the significance of individual tests do not represent the Family-Wise Error Rate (FWER).
  * Various correction techniques exist to reduce false positives, but we note that these naturally reduce power.
  * Bonferroni correction: Change $\alpha = \alpha_0 /m $. This is often criticized as being too conservative, so we have other forms of correction: 
  * Šidák correction: $\alpha = 1 - (1-\alpha_0)^{1/m}$
  * Holm-Bonferroni correction:
    * Sort the $m$ p-values, and if $p_i \leq \frac{\alpha_0}{m+1-k}$, we reject $H_i$
    * This applies early stoppage, so the moment we do not reject $H_i$, all subsequent hypotheses are not rejected too.
* Model Statistics
  * Overfitting
    * To combat overfitting, we can use cross-validation and compare training and validation curves
    * AIC ($2k - 2\ln (\hat{L})$) and BIC ($k\ln n - 2\ln (\hat{L})$) are ways to tradeoff training performance and model complexity ($k$ is the number of estimated parameters).