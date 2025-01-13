# Post-Training

- Post-Training  is a set of processes and techniques that refine and optimize a machine learning model after it's been trained.

## Model Optimization

- Quantization
  - Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with lower-precision data types.
    - Quantization-aware Training (QAT) is a way of training that simulates quantization whilst training
    - Also see [Mixed Precision Training](../25_compuational_performance/notes.md)
  - Double quantization is when we quantize the scaling factors from the first quantization.
    - QLoRA combines double quantization with LoRA.
- Pruning
  - Pruning is a technique that removes less important connections, neurons, or structures from a trained model 

## Guidance

- Take an existing model and steer the generation process at inference time for additional control.
  - It's important to remember that we're _not_ changing the model internals. 
    - For example, the model already has the ability to generate the conditioned outputs, we're instead restricting the inputs to guide this generation process. 
- (input) -> (output) -> (compute loss of output against objective), then backpropagate.
- Soft-prompting (typically classified in parameter efficient fine-tuning, but I think it fits better here)
  - "Prompting" is discrete because changing a prompt represents a discrete "jump" from embedded vectors to another
  - Soft-prompting is then editing (e.g. concatenating) an input with a continuous vector that we can edit in backpropagation
  - Prompt-Tuning: Concatenate prefix vector only to input embedding
  - P-Tuning: Inserts vector anywhere, only to input embedding
  - Prefix-Tuning: Concatenate prefix vector for each transformer block
- See more specific examples in [Diffusion](../10_diffusion/notes.md) and [NLP](../17_nlp/post_training.md). 

## Fine-Tuning

- Re-train existing models on new data to change the type of output they produce.
- This usually involves one of the following options:
  - Adding/Replacing a few output layers
  - Freezing a portion of weights (Parameter Efficient Fine-Tuning)
- Parameter Efficient Fine-Tuning (PEFT)
  - Adapters: Adding layers after FFN layers in a transformer block.
  - LoRA
    - Suppose we only finetune our linear layers, excluding bias terms, converting weight $W \in \mathbb{R}^{a \times b}$ to $W'$
    - We can reparameterize $W'$ as $W + \Delta W$
    - LoRA is a low rank approximation of $\Delta W = AB$, $A \in \mathbb{R}^{a \times r}, B \in \mathbb{R}^{r \times b}$
    - We can then freeze our original model, and focus on training $A$ and $B$ matrices for each of our linear layers.
    - LoRA does not increase inference latency because weights can be merged with the base model. 
  - IA3
    - Rescales (element-wise) activations of key vectors, query vectors and MLP hidden activations.
- See examples in [Diffusion](../10_diffusion/notes.md), [NLP](../17_nlp/post_training.md), and [CV](../16_computer_vision/notes.md).

## Reinforcement Learning with Human Feedback (RLHF)

- In pretraining process, it is hard to incorporate additional (human) preferences
- 4 steps
  - Pretraining a language model (LM)
  - Use human responses to fine-tune LM to follow instructions
  - Gathering data and training a reward model
    - Gather data
      - Prompt LMs with prompts $x$
      - Gather responses $y$
      - Get human rankings
    - Train a reward model
      - Loss is based on $P(y_1 > y_2 \mid x) = \sigma(r(x,y_1) - r(x, y_2))$
      - Can be any model
      - Why do we need this? Ideally in the next step, we can ask a human to generate a reward/rank for any $y \mid x$, but that's prohibitively expensive.
  - Fine-tuning the LM with reinforcement learning
    - ![rlhf.png](rlhf.png)[Source](https://huggingface.co/blog/rlhf)
    - Some parameters of the LM are frozen because fine-tuning an entire 10B or 100B+ parameter model is prohibitively expensive
    - State: $x$
    - Action: $y$
    - Policy: $\pi_{PPO}(y \mid x)$
    - Why is this RL? 
      - If we have a dataset of $(y,x)$ pairs, this can be couched as supervised learning.
      - The key here is that the model itself generates $y \mid x$. 
        - We then also need the KL divergence term to prevent the model from just generating gibberish that just tricks the imperfect reward model. 

## Direct Preference Optimization (DPO)

- Similar to RLHF, but we skip generation of the reward model
  - ![dpo.png](dpo.png)[Source](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
- Loss is based on $P(y_1 > y_2 \mid x) = \sigma(\beta(\log\frac{\pi_{PPO}(y_2\mid x)}{\pi_{base}(y_2\mid x)} - \log\frac{\pi_{PPO}(y_1\mid x)}{\pi_{base}(y_1\mid x)}))$
  - $\beta$ is a temperature parameter. Higher $\beta$ means that model is more sensitive to rankings.
- The simplicity of not needing to model a reward model comes at the cost of DPO being more prone to overfitting to preferences and ending up generating nonsense.
  - While the loss above does have some flavor of minimizing the divergence between $\pi_{PPO}$ and $\pi_{base}$, we find that this KL-regularization is actually insignificant when preferences are very strong, which is exacerbated by our finite data regime (Section 4.2 of [$\Psi$PO paper](https://arxiv.org/pdf/2310.12036))
    - The paper argues that the reward model is useful as a regularizer because it underfits preferences, preventing this problem. 

## $\Psi$PO

- The objective $\max _\pi \underset{\substack{x \sim \rho \\ y \sim \pi(. \mid x) \\ y^{\prime} \sim \mu(. \mid x)}}{\mathbb{E}}\left[\Psi\left(p^*\left(y \succ y^{\prime} \mid x\right)\right)\right]-\tau D_{\mathrm{KL}}\left(\pi \| \pi_{\mathrm{ref}}\right)$ generalizes both RLHF's and DPO's objective functions. 
- DPO corresponds to when $\Psi(q) = \log(\frac{q}{1-q})$, and the unboundedness of this $\Psi$ causes DPO to overfit. 
- The [paper](https://arxiv.org/pdf/2310.12036) proposes taking $\Psi$ to be the identity, but unlike RLHF and like DPO, proposes an empirical solution for this optimization problem. 
  - Sampled IPO
    - We minimize $\underset{\left(y_w, y_l, x\right) \sim D}{\mathbb{E}}\left(\log \left(\frac{\pi_{PPO}(y_w \mid x) \pi_{\mathrm{base}}\left(y_l \mid x\right)}{\pi_{PPO}\left(y_l \mid x\right) \pi_{\mathrm{base}}(y_w \mid x)}\right)-\frac{\tau^{-1}}{2}\right)^2$
    - Intuitively, when the weight on the KL-divergence term $\tau$ is larger, we penalize deviations from our base model, which prevents overfitting.