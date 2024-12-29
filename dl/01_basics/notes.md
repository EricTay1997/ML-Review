# Notes

- Binary classification
  - Consider the following model with one hidden layer, 
  - $y = \operatorname{sign}(\mathbf{W}_2g(\mathbf{W}_1\mathbf{x} + b_1) + b_2)$, $\mathbf{W}_i \in \mathbb{R}^{p_{i-1} \times p_i}$, $g$ is our activation function, $p_2 = 1$. 
    - $\mathbf{W}_1\mathbf{x} + b_1$ means that we're rotating and shifting the input space. 
      - Note that $\mathbf{W}_1$ can also change dimensions
        - For going from 1D to 2D, we take a line and start rotating it in a 2D plane.
        - For going from 2D to 3D, we take points on a flat piece of paper and let the paper rise/fall whilst tilting it. 
    - A nonlinear function $g$ stretches and squeezes space unevenly. 
    - For a good visualization of the above two steps, see [Olah's blog](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
    - $p_1$ represents the number of dimensions we embed our inputs in, when conducting our classification task. 
    - $f(\mathbf{z}) = \operatorname{sign}(\mathbf{W}_2\mathbf{z} + b_2)$ means that we're drawing a hyperplane (with normal vector $\mathbf{W}_2$) in the $p_1$-dimensional space and classifying points based on which side of the hyperplane they fall on. 
  - Non-linearity
    - With just scaling and shifting, we can intuit that it's impossible to separate the XOR case in both 2D and 3D (see corresponding notebook for more details).
  - Number of dimensions
    - Given that the number of hidden units corresponds to the embedding dimension, we can also intuit the number of units needed for a given problem. 
    - Again, [Olah's blog](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) gives good intuition under the "Topology and Classification" section. 
- Multi-class classification
  - We can interpret multi-class classification as drawing $n$ hyperplanes (where $n$ is the number of classes), and assigning probabilities based on how far each point is to the hyperplane. 
  - This is kind of bogus because:
    - The columns of $\mathbf{W}_2$ are not normalized (so we need to scale the distance)
    - The softmax function we typically apply exponentiates this distance
    - But I thought this was pretty cool!
- PyTorch
  - There quite a few PyTorch intricacies. I'm finding it hard to be both comprehensive and not excessive, so for now I will avoid making notes regarding such details.


