# Autoencoders

- In general, an autoencoder consists of an encoder that maps the input $\mathbf{x}$ to a lower-dimensional feature vector $\mathbf{z}$, and a decoder that reconstructs the input $\hat{\mathbf{x}}$ from $\mathbf{z}$.
- $\mathbf{z}$ is useful for tasks like dimensionality reduction, feature extraction, and anomaly detection.

## Variational Autoencoders (VAEs)

- In vanilla autoencoders, we do not have any restrictions on the latent vector.
  - As the autoencoder was allowed to structure the latent space in whichever way it suits the reconstruction best, there is no incentive to map every possible latent vector to realistic images.
  - Hence, it is unsuitable for generative tasks. 
- In contrast, VAEs regularizes the latent space to follow a Gaussian distribution, and thus is suitable for generative tasks. 
- Concretely, for each data point $\mathbf{x}_i$:
  - The encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ assumes $q_\phi(\mathbf{z}_i \mid \mathbf{x}_i) \sim \mathcal{N}(\pmb\mu_i, \sigma^2_i\mathbf{I})$, and learns to map each $\mathbf{x}_i$ to its associated $\pmb\mu_i$ and $\sigma_i$. 
  - The decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$ also assumes $p_\theta(\mathbf{x}_i \mid \mathbf{z}_i) \sim \mathcal{N}(\tilde{\pmb\mu}_i, \tilde{\sigma}^2_i\mathbf{I})$, and learns to reconstruct $\mathbf{x}_i$ from $\mathbf{z}_i$.
  - The loss to minimize is then as follows:
    - $\text{ELBO}=\mathcal{L}_{\pmb{\theta}, \pmb{\phi}}(\mathbf{x}) = \mathbf{E}_{\mathbf{z} \sim q_\phi\left(\mathbf{z} \mid \mathbf{x} \right)}\left[\log p_\theta\left(\mathbf{x} \mid \mathbf{z}\right)\right]-D_{K L}\left(q_\phi\left(\mathbf{z} \mid \mathbf{x} \right) \| p_\theta(\mathbf{z})\right)$
    - The first term is the reconstruction loss, and for the second term, the decoder assumes a prior of $p_\theta(\mathbf{z}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, which causes the encoder to find latent representations closer to the unit ball. 
  - How did we get to this loss? [(An Introduction to Variational Autoencoders)](https://arxiv.org/pdf/1906.02691)
    - We start by wanting to generate samples from our decoder, $\mathbf{x} \sim p_\theta(\mathbf{x})$, and we hope that our decoder approximates the true distribution of the data $p_\theta(\mathbf{x}) \approx p(\mathbf{x})$. 
    - We introduce latent variables $\mathbf{z}$ following a pre-specified distribution $p_\theta(\mathbf{z})$, so our decoder can now be specified by having both $p_\theta(\mathbf{x} \mid \mathbf{z})$ and $p_\theta(\mathbf{z})$. 
    - How do we know if our decoder is good? 
      - We want to maximize the likelihood $p_\theta(\mathbf{x})$.
      - Assuming our data is independent, and taking one sample, 
      - Likelihood $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})p_\theta(\mathbf{z})d\mathbf{z}$
      - Consider when our $p_\theta(\mathbf{x} \mid \mathbf{z})$ is a neural network - we need to run it for _all_ $\mathbf{z}$! This is intractable.
  - Addressing intractability with variational inference
    - For any inference model $q_\phi(\mathbf{z} \mid \mathbf{x})$,
    - $\begin{aligned} \log p_{\pmb{\theta}}(\mathbf{x}) & =\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\pmb{\theta}}(\mathbf{x})\right] \\ & =\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\pmb{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right] \\ & =\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\pmb{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})} \frac{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right] \\ & =\underbrace{\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[p_{\pmb{\theta}}(\mathbf{x} \mid \mathbf{z})\frac{p_{\pmb{\theta}}(\mathbf{z})}{q_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{\text{ELBO}=\mathcal{L}_{\pmb{\theta}, \pmb{\phi}}(\mathbf{x}) }+\underbrace{\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=D_{K L}\left(q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})\right)}\end{aligned}$
    - Given that KL Divergence $\geq 0$, the ELBO (as defined above) forms a lower bound for the log likelihood. The hope is therefore that maximization of the ELBO w.r.t. the parameters $\pmb\theta$ and $\pmb\phi$ would approximately maximize the likelihood. 
  - Additional implementational details:
    - We use the reparameterization trick $\mathbf{z}_i = \pmb\mu_i + \sigma_i\pmb\epsilon$, $\pmb\epsilon \sim \mathcal{N}(\mathbf{0, I})$ to allow gradients to update $\pmb\phi$
    - Since we assume that $p_\theta(\mathbf{x} \mid \mathbf{z})$ is gaussian, we minimize MSE in reconstruction.
    - There is a closed form solution for $D_{K L}\left(q_\phi\left(\mathbf{z}_i \mid \mathbf{x}_i \right) \| p_\theta(\mathbf{z})\right)$, which we plug in directly to the loss. 
    - [Understanding Transposed Convolutions](https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967)
  - ToDo:
    - Understand more about why we can approximate the loss with empirical quantities. 
