# Notes

- Interpreting MLPs
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
  - There quite a few PyTorch intricacies. I'm finding it hard to be both comprehensive and not excessive, so for now I will avoid making notes regarding most details.
  - Permutations
    - Consider the following $B \times C \times H \times W$ tensor $a$.
    - The order that we read elements would go from last to first.
      - I.e. we go along $W$, changing columns. Then we go along $H$, changing rows, etc. 
      - Suppose we now want to change $a$ into $b$
        - Going from 0 to 4 means that our last dimension is the $C$ dimension. 
        - Going from 4 to 1 means that our 2nd last dimension is the $W$ dimension. 
        - Going from 5 to 8 means that our 3rd last dimension is the $B$ dimension. 
    - It is also important to note that permutations don't change the _order_ of an element
      - The word order here is overloaded, what I mean is that the element at position $ijkl$ will move to some permutation of $ijkl$, e.g. $kjli$, but it'll still be the $k^{th}$ element in a specified direction.  
```
a = torch.tensor([
    [[[ 0,  1],
      [ 2,  3]],

     [[ 4,  5],
      [ 6,  7]]],


    [[[ 8,  9],
      [10, 11]],

     [[12, 13],
      [14, 15]]]])
b = torch.tensor([[ 0,  4,  1,  5],
                  [ 8, 12,  9, 13],
                  [ 2,  6,  3,  7],
                  [10, 14, 11, 15]])
b == a.permute(2,0,3,1).reshape(4,4)
>>> True
```
  - - Reshaping
      - Reshaping is also processed from last dimension to first.
        - I.e. $B \times C \times H \times W \rightarrow B \times C \times HW$ would leave the first 2 dimensions untouched. 
        - [Exercise](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html)
          - Suppose we want to split $a$ into $H'W'$ patches of size $p_H \times p_W$ ($H=p_HH'$)
          - Since reshaping is processed from last to first, we do
            - `a.reshape(B, C, H', pH, W', pW)`
              - Why `(W', pW)` and not `(pW, W')`?
              - Because the last dimension should "read" across columns.
          - Next, we permute $a$ to
            - `(B, H', W', C, pH, pW)`
            - Permuting doesn't "mess up" the ordering of elements, it simply tells us _how_ to read them
              - Why `(H', W')` and not `(W', H')`?
              - This essentially tells us what order to "read" the patches. Here, we're saying read them in Raster Scan order.
    - [Transformer Exercise](https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.ipynb)
      - Line 1: We merge our weight matrices together column-wise
      - Line 2: `qkv` is of size $B \times L \times 3hD$, this just expands the last dimension
        - Note that we can do `(3D, h)` instead of `(h, 3D)`
        - `(3D, h)` implies we do $W = [W_Q ; W_K ; W_V]$, where each $W_Q$ is then a column-wise concatenation of each head. 
      - Line 3: This is ok to do because permutations don't mess up orders. 
      - Line 6: I believe whether we have `(h, 3D)` or `(3D, h)` here is irrelevant and only changes the order of columns for $W_O$. To be consistent with the other weight matrices this ordering seems natural.
```
(1) qkv = nn.Linear(input_dim, 3*embed_dim)(x)
(2) qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
(3) qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
(4) q, k, v = qkv.chunk(3, dim=-1)
(5) values, attention = scaled_dot_product(q, k, v, mask=mask)
(6) values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
(7) values = values.reshape(batch_size, seq_length, self.embed_dim)
(8) o = nn.Linear(embed_dim, embed_dim)(values)
```
- Normalization
  - Batch Normalization
    - BN reparameterizes the per-channel mean and SD as such:
      - $O_{b, c, x, y} = \gamma_c \frac{I_{b, c, x, y}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c$
      - `(x - x.mean(dim = (0,2,3), keepdim = True))/x.std(dim = (0,2,3), unbiased = False, keepdim = True)`
    - While the learnable parameters $\gamma_c$ and $\beta_c$ seem to defeat the purpose of normalization, it offers different learning dynamics where instead of having the mean and SD determined by the complex interaction between layers, we can optimize for this via gradient descent. 
    - Since batch statistics can have high variance especially with small batches,
      - In training, we often compute and use a running mean and variance for normalization.
      - In eval, we use the running mean and variance for normalization.
    - A side note is that convolutional layers need not have a bias anymore if it precedes a BN layer.
  - Layer Normalization
    - LN normalizes over the last dimension - for each batch, for each token, normalize its features.
    - `(x - x.mean(dim = (-1), keepdim = True))/x.std(dim = (-1), unbiased = False, keepdim = True)`
    - This makes for easier parallelizability without any batch-wise dependence.
    - This could also be more relevant when [batch statistics have large variance](https://arxiv.org/pdf/2003.07845v1).