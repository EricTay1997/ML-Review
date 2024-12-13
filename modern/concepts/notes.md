# Concepts (Misc)

## The Curse of Dimensionality
- As the number of dimensions increase, the "space" between training points increases significantly

## Local Constancy and Smoothness Regularization
- Many ML models operate under the prior that the function we want to learn should not change very much within a small region. 
- Consider now a checkerboard - are these assumptions sufficient?
- Is it possible to represent a complicated function efficiently, and is it possible for an estimated function to generalize well to new inputs? Yes!
  - A large number of regions $O(2^k)$ can be defined with $O(k)$ examples, so long as we introduce dependencies between the regions through additional assumptions about hte underlying data-generating process.
  - Task-specific assumptions could also be appropriate, such as assuming that the target function is periodic.

## Manifold Learning
- The manifold interpretation is that natural data forms lower-dimensional manifolds in its embedding space. 
  - I view this as the motivation for why we "encode" data into a lower-dimensional space. 
- [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) mentions the inverse: All $n$-dimensional manifolds can be untangled in $2n+2$ dimensions. 
  - I wonder if this is good intuition for first increasing the number of hidden units and then going back down? Like the MLP layers of a transformer...

## Optimization
- Newton's method
- Quasi-Newton's methods
- Goodfellow's chapter 8 has a lot of interesting discussion here