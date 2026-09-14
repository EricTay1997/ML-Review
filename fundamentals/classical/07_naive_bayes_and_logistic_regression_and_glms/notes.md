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

- $Y_i \sim \mathrm{Bern}(\pi_i),$ $\pi_i = \frac{1}{1+e^{-\mathbf{B^{\top}x_i}}} = \sigma(\mathbf{B^{\top}x_i})$. Alternatively, the log odds, $\log (\frac{\pi_i}{1-\pi_i}) = g(\pi_i) = \mathbf{B^{\top}x_i}$
- Assumptions:
  - Observations are independent
  - Linearity of log odds and dependent variables
  - For good estimates, we generally also want
    - Low multicollinearity
    - Larger sample size
    - And no extreme outliers
- Loss
  - The log likelihood can now be written as $\sum_i [y_i\log(\pi_i) + (1-y_i)\log(1-\pi_i)]$, which is the negative of the **[cross-entropy loss](../02_probability_and_info_theory/notes.md)**.
  - Note that this form doesn't explicitly handle imbalances in the dataset, but we can always get our desired sensitivity/precision trade off by changing our threshold. Importantly, we require that the covariates need to have information that distinguishes the two classes.
- Parameter Estimation ([useful reference](https://stats.stackexchange.com/questions/344309/why-using-newtons-method-for-logistic-regression-optimization-is-called-iterati))
  - There is no closed form solution for $\mathbf{B}_{MLE}$, but the loss function is convex. Importantly, we have that
    - $`\nabla_{\mathbf{B}}\mathcal{L}(\mathbf{B}) = \mathbf{X^{\top}}(\pmb\pi - \mathbf{y})`$
    - $`\nabla^2_{\mathbf{B}}\mathcal{L}(\mathbf{B}) = \mathbf{X^{\top}DX}`$, where $`D_{ii} = \pi_i(1-\pi_i)`$ 
  - With these forms, we can then use [Newton's method](../../dl/04_optimization_and_regularization/notes.md) to iteratively update $\mathbf{B}$.
    - The update step turns out to take the form $`\mathbf{\hat{B} = (X^{\top}DX)^{-1}X^{\top}DZ}`$ ([look familiar?](../06_linear_regression_and_regularization/notes.md)), which is why this is often termed iterative re-weighted least squares
- Multiple classes
  - The extention of logistic regression to multiple classes is termed softmax regression, which can be viewed as a one-layer neural network. 
  - To compute the probabilities over $k$ classes for $y_i$, we do $\mathrm{softmax}(\mathbf{Wx}_i)$, where $\mathbf{W} \in \mathbb{R}^{k \times p}$, and $[\mathrm{softmax}(\mathbf{x})]_i = \frac{\exp(x_i)}{\sum_i \exp(x_i)}$
    - The softmax functions as a normalizing function, which also places even higher emphasis on large values (than a simple scaling function).
  - The log likelihood is now $LL =\sum_i\sum_k [y_{ik}\log(\hat{p}_{ik})$], where $y_i \in \mathbb{R}^k$ takes a one-hot encoding form and $\hat{p}_i$ represents our predictions for this data point.
    - Note that this is still the negative of the **[cross-entropy loss](../02_probability_and_info_theory/notes.md)** ($`\sum_k [y_{ik}\log(\hat{p}_{ik})]`$ is the cross-entropy loss between $`y_i`$ and $`\hat{p}_i`$)
    - Note that in the 2-class case, this is the same as the likelihood for logistic regression
      - We can interpret logistic regression as producing unnormalized logits $`\mathbf{B^{\top}x_i}`$.
    - If we denote $`\mathrm{softmax}(\hat{o}_{ik}) = \hat{p}_{ik},`$ then we have that $`\frac{\partial LL}{\partial \hat{o}_{ik}} = y_{ik}-\hat{p}_{ik},`$ the gradient is the difference between the truth and the predicted probability.
      - Proof: write the log-softmax as logit minus logsumexp and use $`\sum_k y_{ik} = 1`$ to pull the logsumexp out of the sum: $`LL_i = \sum_k y_{ik}\left(\hat{o}_{ik} - \log\sum_j e^{\hat{o}_{ij}}\right) = \sum_k y_{ik}\hat{o}_{ik} - \log\sum_j e^{\hat{o}_{ij}}`$. Differentiating wrt $`\hat{o}_{ik}`$ gives $`y_{ik}`$ from the first term and $`e^{\hat{o}_{ik}} / \sum_j e^{\hat{o}_{ij}} = \hat{p}_{ik}`$ from the second.
        - Going through the softmax Jacobian $`\partial \hat{p}_{ij} / \partial \hat{o}_{ik} = \hat{p}_{ij}(\delta_{jk} - \hat{p}_{ik})`$ instead gets the same answer with more cancellation. The logit-minus-logsumexp form is also what a fused cross-entropy kernel computes for the LM head.
    - Permitting some abuse of notation, the loss gradient wrt $`\mathbf{W}_j \in \mathbb{R}^{p}`$ (transpose of $`j^{th}`$ row) still permits the form $`\nabla_{\mathbf{W}_j}\mathcal{L}(\mathbf{W}) = \mathbf{X^{\top}}(\pmb\pi - \mathbf{y}),`$ where $`\pi_i`$ is now $`\hat{p}_{ij}`$, $`y_i =1`$ if the $`i^{th}`$ datapoint's class is $`j`$ and 0 otherwise.
    - In the LLM world, this one-layer network is the LM head: $`\mathbf{x}_i`$ is the residual stream $`\mathbf{h}_t`$ at position $`t`$, $`\mathbf{W}`$ is the unembedding $`\mathbf{W}_U \in \mathbb{R}^{V \times d_{model}}`$, and the entire transformer body is a learned feature extractor for one softmax regression per position.
      - The loss is $`-\log \hat{p}_{y_t}`$ of the observed next token, summed over positions (SFT is the same loss masked to response tokens only).
      - Reading $`\hat{p}_{ik} - y_{ik}`$ per logit: the correct token is pushed up by $`1-\hat{p}_{y_t}`$ (the missing mass), each wrong token $`k`$ is pushed down by $`\hat{p}_{k}`$, and the two sum to the same amount. Tokens the model already gets right contribute almost nothing, so SFT spends its gradient on the tokens it finds surprising (cf. [new-knowledge SFT and hallucinations](../../../llms/post_training/notes.md#hallucinations)).
      - One layer back, $`\nabla_{\mathbf{h}_t}\mathcal{L} = \mathbf{W}_U^{\top}(\hat{\mathbf{p}} - \mathbf{y}) = \mathbb{E}_{k \sim \hat{\mathbf{p}}}[\mathbf{w}_k] - \mathbf{w}_{y_t}`$: the residual stream is moved away from the unembedding vector the model currently expects and toward that of the token that actually came next.
      - Since the target at each position is one-hot, $`H(P)=0`$ and the cross entropy _is_ the forward KL $`D_{KL}(p_{data} \| p_{model})`$ of [SFT-as-distillation](../../../llms/rl/kl_divergence.md#distillation). Pretraining targets real language with $`H(P)>0`$, which is the irreducible floor in pretraining loss.

- Comparison vs Naive Bayes
  - Pros for Naive Bayes
    - Computationally easier (no optimization)
    - Converges faster, needs less data
    - Can incorporate priors easily
    - Logistic Regression assumptions may not be valid (i.e. the true log-odds may not be linear in the features; note LR does not assume separable data — perfect separation actually makes the MLE diverge)
  - Pros for Logistic Regression
    - Naive Bayes assumptions may not be valid
    - Feature Response relationship can be learned
    - Concept of feature importance

## GLMS
- GLMs extend the concept of logistic regression and assert that $\mathrm{E}(\mathbf{Y} \mid \mathbf{X})=\pmb{\mu}=g^{-1}(\mathbf{X} \mathbf{B})$
  - For logistic regression, we see that $g^{-1} = \sigma$
- MLE estimates are usually found using iteratively reweighted least squares / Newton's, or [Fisher's scoring method](../../dl/04_optimization_and_regularization/notes.md).