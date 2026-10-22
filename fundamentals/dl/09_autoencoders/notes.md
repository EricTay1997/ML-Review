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
  - The objective to maximize (the loss to minimize is its negative) is:
    - $`\text{ELBO}=\mathcal{L}_{\pmb{\theta}, \pmb{\phi}}(\mathbf{x}) = \mathbf{E}_{\mathbf{z} \sim q_\phi\left(\mathbf{z} \mid \mathbf{x} \right)}\left[\log p_\theta\left(\mathbf{x} \mid \mathbf{z}\right)\right]-D_{K L}\left(q_\phi\left(\mathbf{z} \mid \mathbf{x} \right) \| p_\theta(\mathbf{z})\right)`$
    - The first term is the reconstruction loss, and for the second term, the decoder assumes a prior of $`p_\theta(\mathbf{z}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`$, which causes the encoder to find latent representations closer to the unit ball. 
  - How did we get to this loss? [(An Introduction to Variational Autoencoders)](https://arxiv.org/pdf/1906.02691)
    - We start by wanting to generate samples from our decoder, $`\mathbf{x} \sim p_\theta(\mathbf{x})`$, and we hope that our decoder approximates the true distribution of the data $`p_\theta(\mathbf{x}) \approx p(\mathbf{x})`$. 
    - We introduce latent variables $`\mathbf{z}`$ following a pre-specified distribution $`p_\theta(\mathbf{z})`$, so our decoder can now be specified by having both $`p_\theta(\mathbf{x} \mid \mathbf{z})`$ and $`p_\theta(\mathbf{z})`$. 
    - How do we know if our decoder is good? 
      - We want to maximize the likelihood $`p_\theta(\mathbf{x})`$.
      - Assuming our data is independent, and taking one sample, 
      - Likelihood $`p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})p_\theta(\mathbf{z})d\mathbf{z}`$
      - Consider when our $`p_\theta(\mathbf{x} \mid \mathbf{z})`$ is a neural network - we need to run it for _all_ $`\mathbf{z}`$! This is intractable.
  - Addressing intractability with variational inference
    - For any inference model $`q_\phi(\mathbf{z} \mid \mathbf{x})`$,
    - $`\begin{aligned} \log p_{\pmb{\theta}}(\mathbf{x}) & =\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\pmb{\theta}}(\mathbf{x})\right] \\ & =\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\pmb{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right] \\ & =\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\pmb{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})} \frac{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right] \\ & =\underbrace{\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[p_{\pmb{\theta}}(\mathbf{x} \mid \mathbf{z})\frac{p_{\pmb{\theta}}(\mathbf{z})}{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{\text{ELBO}=\mathcal{L}_{\pmb{\theta}, \pmb{\phi}}(\mathbf{x}) }+\underbrace{\mathbb{E}_{q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=D_{K L}\left(q_{\pmb{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\pmb{\theta}}(\mathbf{z} \mid \mathbf{x})\right)}\end{aligned}`$
    - Given that KL Divergence $`\geq 0`$, the ELBO (as defined above) forms a lower bound for the log likelihood. The hope is therefore that maximization of the ELBO w.r.t. the parameters $`\pmb\theta`$ and $`\pmb\phi`$ would approximately maximize the likelihood. 
  - Additional implementational details:
    - We use the reparameterization trick $`\mathbf{z}_i = \pmb\mu_i + \sigma_i\pmb\epsilon`$, $`\pmb\epsilon \sim \mathcal{N}(\mathbf{0, I})`$ to allow gradients to update $`\pmb\phi`$
    - Since we assume that $`p_\theta(\mathbf{x} \mid \mathbf{z})`$ is gaussian, we minimize MSE in reconstruction.
    - There is a closed form solution for $`D_{K L}\left(q_\phi\left(\mathbf{z}_i \mid \mathbf{x}_i \right) \| p_\theta(\mathbf{z})\right)`$, which we plug in directly to the loss. 
    - [Understanding Transposed Convolutions](https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967)
  - ToDo:
    - Understand more about why we can approximate the loss with empirical quantities. 
- Variants
  - Vector Quantised VAEs (VQ-VAEs) use a discrete latent space, in which the encoder output is mapped to the nearest vector in a learned codebook, preventing collapse by forcing the encoder to output semantically meaningful, discrete codes.
  - Residual Vector Quantised VAEs (RVQ-VAEs) address limitations of VQ-VAEs, namely the fixed codebook size and potentially limited codebook utilisation.
    - Uses multiple VQ layers in a hierarchical fashion
    - Each layer quantises the residual error from the previous layer, allowing for more fine-grained representations.

## The ELBO

- Latent-variable model
  - Generate by sampling $`\mathbf{z} \sim p(\mathbf{z})`$ from a fixed prior (e.g. $`\mathcal{N}(\mathbf{0}, \mathbf{I})`$), then $`\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})`$.
  - The likelihood marginalizes out $`\mathbf{z}`$: <div align="center">
    $`\displaystyle p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})\,d\mathbf{z} = \mathbb{E}_{p(\mathbf{z})}\left[p_\theta(\mathbf{x} \mid \mathbf{z})\right]`$ </div>
  - With e.g. a neural-net decoder $`p_\theta(\mathbf{x} \mid \mathbf{z})`$, this can be intractable.
- Importance sampling
  - One solution to intractability is Monte Carlo: draw $`K`$ samples $`\mathbf{z}_k \sim p(\mathbf{z})`$ and average, $`\hat{p}_\theta(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K} p_\theta(\mathbf{x} \mid \mathbf{z}_k)`$.
  - Unbiased, but high variance: only a tiny region of $`\mathbf{z}`$ plausibly generates a given $`\mathbf{x}`$, so almost every prior sample gives $`p_\theta(\mathbf{x} \mid \mathbf{z}_k) \approx 0`$ and the estimate is dominated by rare hits.
    - Roughly, sampling from the prior needs $`K \approx \exp\left(D_{KL}(p_\theta(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))\right)`$ samples ([Chatterjee and Diaconis](https://arxiv.org/pdf/1511.01437)).
  - Instead, sample from any $`q_\phi(\mathbf{z} \mid \mathbf{x})`$, and reweight: <div align="center">
    $`\displaystyle p_\theta(\mathbf{x}) = \int q_\phi(\mathbf{z} \mid \mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\,d\mathbf{z} = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[w\right], \quad w = \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}`$ </div>
    - Still unbiased. If $`q_\phi(\mathbf{z} \mid \mathbf{x})`$ concentrates on the $`\mathbf{z}`$ that explain $`\mathbf{x}`$, the importance weights $`w`$ are far less extreme than $`p_\theta(\mathbf{x} \mid \mathbf{z}_k)`$→ lower variance.
  - Example: estimate $`\mathbb{E}_p[f(z)]`$ with $`z \in \{A, B\}`$, $`p(A) = 0.99`$, $`p(B) = 0.01`$, $`f(A) = 0`$, $`f(B) = 100`$. True value is $`1`$.
    - Sample from $`p`$: 99% of samples give $`0`$, 1% give $`100`$ → variance $`99`$.
    - Sample from $`q(A) = q(B) = 0.5`$ and use $`pf/q`$: half give $`0`$, half give $`2`$ → mean still $`1`$, variance $`1`$.
    - Sample from $`q(B) = 1`$: every sample gives exactly $`1`$ → variance $`0`$.
  - **Note that the ideal sampler is the true posterior**, $`q^* = p_\theta(\mathbf{z} \mid \mathbf{x})`$.
    - For $`f \geq 0`$, the zero-variance proposal is $`q^* \propto pf`$ (the $`q(B) = 1`$ case above). Here that is $`p(\mathbf{z})\,p_\theta(\mathbf{x} \mid \mathbf{z}) = p_\theta(\mathbf{x}, \mathbf{z}) \propto p_\theta(\mathbf{z} \mid \mathbf{x})`$, which makes every weight $`w = p_\theta(\mathbf{x})`$ exactly, so one sample suffices.
- Optimizing $`\log p_\theta(\mathbf{x})`$
  - We want to maximize $`\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[w]`$ over the dataset.
  - For computational efficiency, we'd like to take just one sample $`\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})`$ per $`\mathbf{x}`$ and use $`\log w`$ (unbiased estimate of the ELBO ELBO).
  - But $`\log`$ is concave, so by Jensen's inequality this systematically under-estimates the log-likelihood: <div align="center">
    $`\displaystyle ELBO = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[\log w\right] \leq \log \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[w\right] = \log p_\theta(\mathbf{x})`$ </div>
  - Aside: can we do better? Take $`K`$ samples per $`\mathbf{x}`$, average the weights, _then_ take the log. This is the [IWAE](https://arxiv.org/pdf/1509.00519) (importance weighted autoencoder) bound: <div align="center">
    $`\displaystyle \mathcal{L}_K = \mathbb{E}_{\mathbf{z}_1, \dots, \mathbf{z}_K \sim q_\phi(\mathbf{z} \mid \mathbf{x})}\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{p_\theta(\mathbf{x}, \mathbf{z}_k)}{q_\phi(\mathbf{z}_k \mid \mathbf{x})}\right]`$ </div>
    - $`ELBO = \mathcal{L}_1 \leq \mathcal{L}_2 \leq \dots \leq \log p_\theta(\mathbf{x})`$, and $`\mathcal{L}_K \to \log p_\theta(\mathbf{x})`$ as $`K \to \infty`$: less biased, but $`K\times`$ the decoder passes per $`\mathbf{x}`$.
  - Back to one sample/ELBO: how bad is the bias?
- ELBO
  - Factorization 1 (the bias): substitute $`p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{z} \mid \mathbf{x})\,p_\theta(\mathbf{x})`$ (full derivation in the VAE section above): <div align="center">
    $`\displaystyle \log p_\theta(\mathbf{x}) - \mathcal{L}_{\theta,\phi}(\mathbf{x}) = D_{KL}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right)`$ </div>
    - $`\log p_\theta(\mathbf{x})`$ doesn't depend on $`\phi`$, so **maximizing the ELBO over the encoder minimizes the gap/bias**
      - Note that I motivated sampling from $q$ via importance sampling, but we could technically get here just via the bias lens (that $`D_{KL}\left(p(\mathbf{z}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right)`$ is large). I just find that unintuitive since we seem to introduce a lot of terms "for no reason".
      - For completeness, we also discuss the expectation-maximization (EM) view below, where **$`q`$ is our guess for the missing $`\mathbf{z}`$**.
        - If we knew the $`\mathbf{z}`$ behind each $`\mathbf{x}`$, we'd just maximize $`\log p_\theta(\mathbf{x}, \mathbf{z})`$. We don't, so EM guesses $`\mathbf{z}`$ with the posterior at the current parameters $`\theta_\text{old}`$ (E-step), then fits $`\theta`$ on those guesses (M-step): <div align="center">
          $`\displaystyle \theta_\text{new} = \arg\max_\theta\, \mathbb{E}_{p_{\theta_\text{old}}(\mathbf{z} \mid \mathbf{x})}\left[\log p_\theta(\mathbf{x}, \mathbf{z})\right]`$ </div>
        - Note that $`\displaystyle \log p_\theta(\mathbf{x}) = \mathbb{E}_{p_\theta(\mathbf{z} \mid \mathbf{x})}\left[\log p_\theta(\mathbf{x}, \mathbf{z})\right] + H\left(p_\theta(\mathbf{z} \mid \mathbf{x})\right)`$
          - EM drops $`H`$ because the posterior is held fixed at $`\theta_\text{old}`$ while $`\theta`$ is updated, so its entropy is a constant.
        - A VAE can't compute the posterior, so it guesses with a learned $`q_\phi(\mathbf{z} \mid \mathbf{x})`$ instead. Now $`H(q_\phi)`$ depends on $`\phi`$ and has to stay, and it's where the ELBO's denominator comes from: <div align="center">
          $`\displaystyle \mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[\log p_\theta(\mathbf{x}, \mathbf{z})\right] + H(q_\phi) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right]`$ </div>
          - Without $`H`$, the best $`q_\phi`$ would collapse onto the single most likely $`\mathbf{z}`$.
          - The price of guessing is the gap above, $`D_{KL}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right)`$, which is zero when $`q_\phi`$ is the posterior (recovering the identity above).
    - Over $`\theta`$, the gap is not neutral: $`\nabla_\theta \mathcal{L}_{\theta,\phi}(\mathbf{x}) = \nabla_\theta \log p_\theta(\mathbf{x}) - \nabla_\theta D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x}))`$, so the decoder is also rewarded for having posteriors that $`q_\phi(\mathbf{z} \mid \mathbf{x})`$ can fit easily.
      - Extreme case: a decoder that ignores $`\mathbf{z}`$ has posterior = prior, so the gap is zero. This is one explanation of posterior collapse, where $`\mathbf{z}`$ carries no information about $`\mathbf{x}`$ ([Variational Lossy Autoencoder](https://arxiv.org/pdf/1611.02731)).
  - Factorization 2 (the VAE loss): henceforth ignoring the gap, substitute $`p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})`$ instead: <div align="center">
    $`\displaystyle \mathcal{L}_{\theta,\phi}(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right]}_{\text{reconstruction (explain the data)}} - \underbrace{D_{KL}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})\right)}_{\text{prior matching (be simple)}}`$ </div>
    - This is the VAE objective above: Gaussian decoder → MSE reconstruction; Gaussian encoder with $`\mathcal{N}(\mathbf{0}, \mathbf{I})`$ prior → closed-form KL.
