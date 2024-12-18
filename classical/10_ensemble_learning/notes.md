# Ensemble Learning

- Ensemble learning leverages the idea of a voting mechanism. 

## Random Forests
- Random Forests is an ensemble of decision trees.
- Bagging and Pasting
  - We can get a diverse set of classifiers by diversifying our training data.
  - Bagging/Pasting is when we sample each training set with/without replacement.
  - We can also sample from the set of features to use.
  - Random patches method: Sampling from both samples and features
  - Random subspaces method: Sampling from only features
  - Typically, 
    - Bagging is done for samples, typically with max_samples set to the size of the training set.
    - At each split, we also typically search for the best feature out of $\sqrt{n}$ (rather than all) features.
      - We incur higher bias for lower variance.
- Pruning: we generally do not prune individual trees, since our the combination of sampling methods and aggregation tends to guard against overfitting.
- Feature Importance: Random forests can report how much a feature reduces impurity on average (across nodes it's responsible for), across all trees in a forest.
- Computational Complexity: We can reduce training and inference runtime significantly with parallelism.

## Boosting
- The general idea of boosting is to train each predictor sequentially, each trying to correct its predecessor. 
- AdaBoost
  - Suppose we have $n$ training samples, $K$ classes and $J$ classifiers. 
  - The goal of AdaBoost is to train both the $J$ classifiers and assign weights $\alpha_j$ to each of them, such that our prediction takes the form:
  - $\hat{y}(\mathbf{x}) = \arg\max_k \sum_{j = 1, \hat{y}_j(\mathbf{x}) = k}^K \alpha_j$ (Output the class with the highest weighted vote), where $\hat{y}_j(\mathbf{x})$ represents the prediction of the $j^{th}$ classifier.
  - Algorithm
    - Initialize our _sample_ weights $\mathbf{w} \in \mathbb{R}^n$ with $\frac{1}{n}\mathbb{1}$. 
    - For $j = 1, 2, \ldots, J:$
      - Train classifier $h_j$. 
      - Calculate error rate $r_j = \sum_{i, \hat{y}_j(x_i) \neq y_i} w_i$
      - Store classifier weight $\alpha_j = \eta \log\frac{1-r_j}{r_j}$ (higher weight for better classifiers), $\eta$ is our learning rate
      - For $i = 1, 2, \ldots, n:$
        - If $\hat{y}_j(x_i) \neq y_i$, multiply $w_i$ by $e^{\alpha_j}$ (Upweight samples that predecessor got wrong)
      - Normalize weights such that $\mathbb{1}^{\top}\mathbf{w} = 1.$
  - Interpretations
    - Coordinate descent: [One can interpret Adaboost as an iterative coordinate descent algorithm](https://users.cs.duke.edu/~cynthia/CourseNotes/BoostingNotes.pdf), where the loss is the exponential loss $\frac{1}{n}\sum_i e^{-y_i\hat{y}_i}$ and the coordinates are $j$ and $\alpha_j$. 
    - Two player game (GANs anyone?): We can imagine two agents, one that generates classifiers, and the other than generates sample weights to fool the first.
- Gradient Boosting
  - For gradient boosting, instead of tweaking the sample weights at every iteration, we fit a new predictor to the residual errors made by the previous predictor. 
  - Note that this permits continuous outputs.
- Computational Complexity: Due to the sequential nature of the algorithm, we cannot parallelize training.
## Stacking
  - What if we train a model to aggregate these weak classifiers? Two layer neural network?

## Generalized Additive Models
- Consider the [Generalized Linear Model](../07_naive_bayes_and_logistic_regression_and_glms/notes.md) that says $\mathrm{E}(\mathbf{Y} \mid \mathbf{X})=\pmb{\mu}=g^{-1}(B_0 + B_1x_1 + \ldots + B_px_p)$
- Now let's add many smooth functions $g(\mu) = b_0 + f(x_1) + \ldots + f(x_p)$. 
- One example is polynomial splines, where $f(x_1)$ can define how $x$ varies on various coordinates, and $f(x_2)$ can define how $x^2$ varies over those coordinates, etc.

## Bayesian Model Averaging
- In [Bayesian Model Averaging](https://www2.stat.duke.edu/courses/Fall19/sta721/lectures/BMA/bma.pdf), we can specify a prior on which data generating model is true and calculate appropriate posteriors.
