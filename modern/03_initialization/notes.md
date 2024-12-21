# Initialization

* Two desirable properties
    * The variance of the input should be propagated through the model to the last layer
    * Similarly, we want a gradient distribution with equal variance across layers. Is this gradient wrt nodes or edges?
* Constant initialization
  * Other 3 layers have the same gradient that's close to 0. Why? 
  * Having the same gradient for parameters that have been initialized with the same values means that we will always have the same value for those parameters. This reduces ESS and effectiveness of weights.
* Constant variance initialization
  * Why do activations vanish/explode? Connection to vanishing/exploding gradients? Is vanishing/exploding gradients reference to nodes or edges?
* Xavier initialization
  * Suppose now we want two conditions 
    * The mean of the activations should be zero 
    * The variance of the activations should stay the same across every layer
  * We have the following relation:
    * $\operatorname{Var}(y_i) = \operatorname{Var}(\sum_j w_{ij}x_j) = \sigma_x^2 = \sigma_x^2 d_{x} \operatorname{Var}(w_{ij})$, which gives
    * $\operatorname{Var}(w_{ij}) = \frac{1}{d_{x}}$
  * Now looking at backpropagation, if we wants the variance of the gradients wrt $x_j$ to be the same as that wrt $y_i$, we can repeat the above exercise to get
  * $\operatorname{Var}(w_{ij}) = \frac{1}{d_{y}}$
  * A compromise here leads us to the well-known Xavier initialization 
    * $W \sim \mathcal{N}\left(0, \frac{2}{d_x + d_y}\right)$, or 
    * $W \sim U\left(-\sqrt{\frac{6}{d_x + d_y}}, \sqrt{\frac{6}{d_x + d_y}}\right)$ (we just need to match variance)
* Tanh can be viewed as linear when activations are small and empirical results indicate that activations will become small as layers increase (is this always true?), so Xavier initialization is ok for tanh. 
* For ReLU, we have
  * $\operatorname{Var}(w_{ij}x_j) = \mathbb{E}(w_{ij}^2)\mathbb{E}(x_{j}^2) - \mathbb{E}(w_{ij})^2\mathbb{E}(x_{j})^2 = \operatorname{Var}(w_{ij})\mathbb{E}(x_{j}^2)$
  * Now taking $x_j$ as the post-ReLU output of the previous layer, we get that $\mathbb{E}(x^2) = \frac{1}{2}\operatorname{Var}(\tilde{y}),$ where $\tilde{y}$ is the pre-ReLU activations of the previous layer. 
  * Hence, our variance is now $\frac{2}{d_x}$, which gives us the Kainming initialization. In their paper, they argue that using $d_x$ or $d_y$ both lead to stable gradients throughout the network