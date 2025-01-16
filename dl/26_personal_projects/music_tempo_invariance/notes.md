# Using Tempo Invariance In Pop Music Modelling 

## Hypothesis

We assert that modifying the input or model architecture to account for [tempo](https://en.wikipedia.org/wiki/Tempo) differences in pop music would lead to higher quality latent representations, ultimately leading to:
- A higher quality of generated music
- Possibly a better understanding of our latent representations (interpretability)

## Problem 

- Common encoders like [SoundStream](https://arxiv.org/pdf/2107.03312) convert audio into a mel-spectrogram and process it like an image. 
- The use of a uniform 2D convolutional kernel fails to capture the inherent temporal structure in most pop music, which is mostly in [4/4 time](https://en.wikipedia.org/wiki/Time_signature).
- In particular, if two songs only differ in their tempo, the second spectrogram will look very similar to a "horizontally stretched out" version of the first.
- Intuitively, if we reshape the width of these "images" (input modification), or scale the width of our kernels (model architecture modification) to account for tempo differences, we would have easier time learning kernels that generalize over multiple samples. 

## Additional Work

- [Scale-invariant convolutions](https://arxiv.org/pdf/2102.02282) are useful for downbeat tracking.