# Video

- Vs Image
  - Video generation is more complex because of the added dimension of time and the requirement for motion to be consistent over time. 
  - In generation, video models either condition each grame on previous ones, or generate a sequence as a whole, incorporating the dynamics of movement and change acros time. 
  - 3D convolutional kernels are commonly used
- Encoding
  - [Flamingo](https://arxiv.org/pdf/2204.14198) samples frames and encodes them independently to which learned temporal embeddings are added. 
- Decoding
  - [Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
  - Zeroscope