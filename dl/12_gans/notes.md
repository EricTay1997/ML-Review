# Generative Adversarial Networks

## Adversarial Attacks

- Adversarial attacks are usually grouped into "white-box" and "black-box" attacks. 
- White-box attacks assume that we have access to the model parameter and can, for example, calculate the gradients with respect to the input (similar as in GANs).
- Black-box attacks on the other hand have the harder task of not having any knowledge about the network, and can only obtain predictions for an image, but no gradients or the like.
  - While this seems hard to do, we find that white-box attacks are surprisingly transferable to other models, especially if both models share similarities, e.g. being trained on the same data. 
  - Knowledge of model internals is therefore very valuable. 
- The Fast Gradient Sign Method allows us to create an adversarial example by maximizing the loss:
  - $\tilde{x} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta,x,y))$
- Adversarial Patches
  - Here, we train patches by iteratively placing patches randomly on different samples, and running gradient descent.
- Why are models susceptible to these attacks? 
  - The network learns to classify points on a manifold, but much fo the input space is unexplored.
  - Activations: Given that the output range of a ReLU neuron can be arbitrarily high, a patch or noise that causes a very high value for a single neuron can overpower many other features in the network.
- What can we do?
  - Uncertainty quantification could be helpful. 
  - Defensive distillation: Instead of training the model on the dataset labels, we train a secondary model on the softmax predictions of the first one. This way, the loss surface is "smoothed" in the directions an attacker might try to exploit, and it becomes more difficult for the attacker to find adversarial examples.
  - I wonder if extensive data augmentation could be useful here too.

## GANs 

- GANs consist of a generator and a discriminator which play a minimax game. 
  - $\min _G \max _D L(D, G)=\mathbb{E}_{x \sim p_r(x)}[\log D(x)]+\mathbb{E}_{z \sim p_z(z)}[\log (1-D(G(z)))]$
- The generator generates fake images from noise and the discriminator distinguishes fake and real images. 
-  For each batch, training for GANs generally look like:
    - Train discriminator on real data
    - Generate fake data
    - Train discriminator on fake data
    - Train generator to fool discriminator.
- Problems
  - Hard to achieve Nash equilibrium
  - Balancing act between discriminator and generator
    - When discriminator is too strong, the gradient of the loss function vanishes and learning is slow/halted.
    - When discriminator is too weak, the generator does not have enough feedback. 
  - Low dimensional supports
    - Real data is concentrated in a lower dimensional manifold, the generator generates images from a lower dimensional noise vector. 
    - Both of these are then likely to be disjoint and might bias the training toward having too strong a discriminator. 
  - Mode collapse 
    - The generator may collapse to a setting where it always produces the same outputs.
  - Lack of an evaluation metric. When do we stop training?
- Improved Training 
  - Feature matching
    - The discriminator checks to see if the generator's output matches the expected statistics of the real samples. 
  - Minibatch discrimination
    - The discriminator is additionally given data about other training points
  - Historical averaging
    - This prevents model parameters from updating too quickly.
  - One-sided label smoothing
    - Use softened values like 0.9 and 0.1 for the discriminator. 
  - Virtual Batch Normalization 
    - Normalize each data sample based on a fixed batch of data rather than within its minibatch
  - Adding Noise
    - Add noise in the discriminator to "spread out" the distribution from the lower dimensional manifolds.
      - Alternatively, one can say that this prevents the generator and discriminator from simply memorizing training examples.
    - This also perhaps weakens the discriminator, which is probably helpful especially in the earliest phases.
  - Adding additional information to prevent mode collapse
    - AC-GANs supply both the generator and discriminator with class labels to produce class conditional samples
    - I have implemented the AC-GAN on CIFAR-10 data at https://github.com/EricTay1997/AC-GAN_CIFAR10
  - DCGANs
    - Replaces pooling functions with strided convolutions.
  - WGANs
    - Instead of training a discriminator to tell real and fake samples apart, we have a critic that asks if the samples look similar. 
    - This changes the loss function to the negative difference of logits between real and fake samples. 