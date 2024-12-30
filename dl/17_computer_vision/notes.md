# Computer Vision

## Autoregressive Image Modeling

- For autoregressive image modeling, we predict pixel by pixel in raster scan order. 
- Let's study the PixelCNN Model
- To prevent looking ahead, we use masked convolution kernels
  - ![masked_kernel.png](masked_kernel.png)[Source](https://arxiv.org/pdf/1606.05328)
  - However, these introduce blind spots, and so we instead use horizontal and vertical convolutions
    - ![blind_spot.png](blind_spot.png)[Source](https://arxiv.org/pdf/1606.05328)
    - Additional explanation
      - Blind spots for 3x3 masked filters: for every feature map (not on the boundary) at position (i,j) does not depend feature maps on (i-1,j+2) in the previous layer.
      - Horizontal stacks look left only, including current spot.
      - Vertical stacks look at all rows at/above the current spot.
        - Note that causality within the same row for vertical stack is not ensured
      - The trick to ensure overall causality is that i-th row in vertical stack will only be used in the computation of (i+1)-th row in horizontal stack.
- Loss Function
  - In PixelCNN, for each pixel, we output 256 logits for each possible pixel value, then compute cross entropy loss.
    - This is costly in terms of memory.
    - In addition, the model doesn't know that pixel value $x$ is close to $x-1$. 
  - In PixelCNN+, we instead assume that the output is a mixture of logistic distributions
    - $P(x \mid \pi, \mu, s)=\sum_{i=1}^K \pi_i\left[\sigma\left(\left(x+0.5-\mu_i\right) / s_i\right)-\sigma\left(\left(x-0.5-\mu_i\right) / s_i\right)\right]$
    - The model then tries to learn the mean and scale parameters for each mixture.

## Vision Transformers 

- 