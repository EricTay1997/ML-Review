# Naive Bayes, Logistic Regression and GLMs

- Naive Bayes and Logistic Regression are two models commonly used for classification problems. 
- Logistic Regression provides us with a segue into GLMS.

## Naive Bayes
- $p(y | \mathbf{x}) \propto p(y) \prod_i p(x_i|y)$
  - We evaluate this for every class $y$ and output the class with the highest likelihood.
- Assumptions
  - Conditional independence: The features are conditionally independent of each other, e.g. $p(x_3 | x_2, x_1, y) = p(x_3|y)$.
  - Continuous features are normally distributed within each class
  - Discrete features have multinomial distributions within each class
  - Features are equally important to the prediction of class label

## Logistic Regression 

- $Y_i \sim \operatorname{Bern}(\pi_i),$ $\pi_i = \frac{1}{1+e^{-\mathbf{B^{\top}x_i}}} = \sigma(\mathbf{B^{\top}x_i})$. Alternatively, the log odds, $\log (\frac{\pi_i}{1-\pi_i}) = g(\pi_i) = \mathbf{B^{\top}x_i}$
- Assumptions:
  - Observations are independent
  - Linearity of log odds and dependent variables
  - Low multicollinearity
  - For good estimates, we generally also want
    - Larger sample size
    - And no extreme outliers
- Loss
  - The log likelihood can now be written as $\sum_i [y_i\log(\pi_i) + (1-y_i)\log(1-\pi_i)]$, which is the negative of the **log loss**.
  - Note that this form doesn't explicitly handle imbalances in the dataset, but we can always get our desired sensitivity/precision trade off by changing our threshold. Importantly, we require that the covariates need to have information that distinguishes the two classes.
- Parameter Estimation ([useful reference](https://stats.stackexchange.com/questions/344309/why-using-newtons-method-for-logistic-regression-optimization-is-called-iterati))
  - There is no closed form solution for $\mathbf{B}_{MLE}$, but the loss function is convex. Importantly, we have that
    - $\nabla_{\pmb\theta}\mathcal{L}(\pmb\theta) = \mathbf{X^{\top}}(\pmb\pi - \mathbf{y})$
    - $\nabla^2_{\pmb\theta}\mathcal{L}(\pmb\theta) = \mathbf{X^{\top}DX}$, where $D_{ii} = \pi_i(1-\pi_i)$ 
  - With these forms, we can then use [Newton's method](../../modern/concepts/notes.md) to iteratively update $\mathbf{B}$.
    - The update step turns out to take the form $\mathbf{\hat{B} = (X^{\top}DX)^{-1}X^{\top}DZ}$ ([look familiar?](../linear_regression_and_regularization/notes.md)), which is why this is often termed iterative re-weighted least squares
- Multiple classes
  - The extention of logistic regression to multiple classes is termed softmax regression, which can be viewed as a one-layer neural network. 
  - To compute the probabilities over $k$ classes for $y_i$, we do $\operatorname{softmax}(\mathbf{W}x_i)$, where $W \in \mathbb{R}^{k \times p}$, and $[\operatorname{softmax}(\mathbf{x})]_i = \frac{\exp(x_i)}{\sum_i \exp(x_i)}$
    - The softmax functions as a normalizing function, which also places even higher emphasis on large values (than a simple scaling function).
  - The log likelihood is now $LL =\sum_i\sum_k [y_{ik}\log(\hat{p}_{ik})$], where $y_i \in \mathbb{R}^k$ takes a one-hot encoding form and $\hat{p}_i$ represents our predictions for this data point.
    - Note that in the 2-class case, this is the same as the likelihood for logistic regression
    - If we denote $\operatorname{softmax}(\hat{o}_{ik}) = \hat{p}_{ik},$ then we have that $\frac{\partial LL}{\partial \hat{o}_{ik}} = \hat{p}_{ik} - y_{ik},$ the gradient is the difference between the truth and the predicted probability. 
    - We note that $\sum_k [y_{ik}\log(\hat{p}_{ik})]$ is the cross-entropy loss between $y_i$ and $\hat{p}_i$. 
    - Permitting some abuse of notation, the loss gradient wrt $W$still permits the form $\nabla_{\pmb\theta}\mathcal{L}(\pmb\theta) = \mathbf{X^{\top}}(\pmb\pi - \mathbf{y}),$ where $\pi_i$ is now $\hat{p}_{ik}$, where $k$ is the class that $y_i$ belongs to.

- Comparison vs Naive Bayes
  - Pros for Naive Bayes
    - Computationally easier (no optimization)
    - Converges faster, needs less data
    - Can incorporate priors easily
    - Logistic Regression assumptions may not be valid (e.g. data is not linearly separable)
  - Pros for Logistic Regression
    - Naive Bayes assumptions may not be valid
    - Feature Response relationship can be learned
    - Concept of feature importance

## GLMS
- GLMs extend the concept of logistic regression and assert that $\mathrm{E}(\mathbf{Y} \mid \mathbf{X})=\pmb{\mu}=g^{-1}(\mathbf{X} \mathbf{B})$
  - For logistic regression, we see that $g^{-1} = \sigma$
- MLE estimates are usually found using iteratively reweighted least squares / Newton's, or [Fisher's scoring method](../../modern/concepts/notes.md).