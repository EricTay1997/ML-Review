# Transformers

I've found transformers to be _very confusing_. To that end, these notes aim to be an end-to-end tutorial of what transformers are. While these notes should be comprehensive, I've found the following resources to be particularly instructive:
- [3Blue1Brown Videos, DL5, DL6 and DL7](https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#combination-of-local-and-global-context)

## Encoders and Decoders

- The [transformer paper](https://arxiv.org/pdf/1706.03762) was released with an encoder-decoder architecture. 
  - ![architecture.png](architecture.png) (Encoder is left, Decoder is right)
- Traditionally, "the main difference is that encoders are designed to learn embeddings that can be used for various predictive modeling tasks such as classification. In contrast, decoders are designed to generate new texts, for example, answering user queries." [Source](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)
- I've found this to be confusing especially since:
  - Encoder-only models like BERT also "decode" embeddings into output tokens or text during training.
  - Decoder-only models like GPT do alter token embeddings during training.  
- Ultimately, I think the traditional understanding is still useful in comprehending what encoders/decoders are, but would like to point out two other distinctions that may help in categorization / model selection. 
  - For encoders, output token length is the same as input token length. 
  - For decoders, training is done in an autoregressive fashion. 
- Today, many tasks that were originally achieved with encoder-decoder models can be achieved with decoder-only models. 

## Attention

- Due to the success of decoder-only models, the main innovation of the transformer is in attention. 
- Terminology:
  - $d$ : The model size / hidden state dimension / positional encoding size.
  - $h$ : The number of heads in multi-head attention layer.
  - $L$ : The segment length of input sequence.
  - $\mathbf{X} \in \mathbb{R}^{L \times d}$ : The input sequence where each element has been mapped into an embedding vector of dimension $d$. Note that this represents _one_ sample of training data.
  - $\mathbf{x}_i \in \mathbb{R}^{1 \times d}$ : The $i^{th}$ input token, $i^{th}$ row in $\mathbf{X}$.
  - $\mathbf{W}^{k,i}, \mathbf{W}^{q,i} \in \mathbb{R}^{d \times d_k}$: The key and query weight matrix for head $i$. 
  - $\mathbf{W}^{v,i} \in \mathbb{R}^{d \times d_v}$: The value weight matrix for head $i$.
  - $\mathbf{W}^{o,i} \in \mathbb{R}^{d_v \times \ d}$: The output weight matrix for head $i$.
  - $\mathbf{W}^{o} \in \mathbb{R}^{hd_v \times \ d}$: The overall output weight matrix, formed by concatenating each $\mathbf{W}^{o,i}$ row-wise. 
  - $\mathbf{K}^i = \mathbf{XW}^{k,i}, \mathbf{Q}^i = \mathbf{XW}^{q,i} \in \mathbb{R}^{L \times d_k}$: The key and query input embeddings for head $i$.
  - $\mathbf{V}^i = \mathbf{XW}^{v,i} \in \mathbb{R}^{L \times d_v}$: The value input embeddings for head $i$. 
  - $\mathbf{k}_j^i, \mathbf{q}_j^i \in \mathbb{R}^{d_k}, \mathbf{v}_j^i \in \mathbb{R}^{d_v}$: The key, query and value embeddings for the $j^{th}$ token in the input sequence ($j^{th}$ row).
- Goal:
  - The goal of training is to obtain parameters for $\mathbf{W}^{k,i}, \mathbf{W}^{q,i}, \mathbf{W}^{v,i}, \mathbf{W}^{o,i}$, and MLP layers (not discussed yet).
- Multi-head self-attention
  - An attention block adds (because of residual connection) to input $\mathbf{X}$: 
    - $\mathbf{Y} = \sum_i \left[ \operatorname{softmax}(\frac{\mathbf{Q}^i\mathbf{K}^{i\top}}{\sqrt{d_k}})\mathbf{V}^i\mathbf{W}^{o,i} \right]$ (softmax is applied such that row sums are 1)
  - Naming
    - Multi-head: Summing over $i$ heads
    - Self: Using the same $\mathbf{X}$ for $\mathbf{Q}^i, \mathbf{K}^i$ and $\mathbf{V}^i$ matrices
    - Attention: $\operatorname{attn}(\mathbf{Q}^i, \mathbf{K}^i, \mathbf{V}^i) = \operatorname{softmax}(\frac{\mathbf{Q}^i\mathbf{K}^{i\top}}{\sqrt{d_k}})\mathbf{V}^i$
  - Confusion
    - Typically, we see the following formula: $[\operatorname{attn}(\mathbf{Q}^1, \mathbf{K}^1, \mathbf{V}^1); \dots; \operatorname{attn}(\mathbf{Q}^h, \mathbf{K}^h, \mathbf{V}^h)]\mathbf{W}^o$. This is [mathematically equivalent](https://transformer-circuits.pub/2021/framework/index.html#architecture-attn-independent) to the formula we presented.
  - Multi-head
    - Having multiple heads allows us to learn different attention patterns (see intuition)
    - Each attention head can be computed in parallel.
    - Usually, we set $d_k = d_v = \frac{d}{h}$.
  - Permutation invariance
    - Swapping rows $i$ and $j$ in $\mathbf{X}$ would swap rows $i$ and $j$ in $\mathbf{Y}$.
- Intuition
  - $\mathbf{q}_j^i, \mathbf{k}_j^i$ and $\mathbf{v}_j^i$ are lower-dimensional representations of the $j^{th}$ input token. 
    - I'm not sure if there's a compelling reason for it to be lower dimensional outside of computational cost. 
  - $A = \operatorname{softmax}(\frac{\mathbf{Q}^i\mathbf{K}^{i\top}}{\sqrt{d_k}})$ is a matrix where $A_{ab}$ represents how similar $\mathbf{k}_b^i$ is to $\mathbf{q}_a^i$, relative to the other keys. 
  - What attention head $i$ does to the $a^{th}$ row of $\mathbf{X}$, is to _add_ additional context to its embedding, given by $\sum_b (A_{ab}\mathbf{v}_b^{i\top}\mathbf{W}^{o,i})$
  - Here, we see that every input token can now absorb context from any other input token in the same sequence (limited by $L$). This addresses a major weakness in RNNs, which have difficulty remembering long inputs in seq2seq tasks.
  - Why do we need matrices to convert $\mathbf{X}$ into these $\mathbf{q}_j^i, \mathbf{k}_j^i$ and $\mathbf{v}_j^i$ vectors?
    - This allows us to more flexibly query and match queries. One such example is to find the following word for the last time we encountered the current word. (See [Q and K Composition](../23_safety/02_interpretability.md))
  - Why do we model $\mathbf{v}_b^i$ and $\mathbf{W}^{o,i}$ separately?
    - My intuition is that it's computational. 
- Softmax and Temperature
  - The softmax ensures that the row sums are 1. 
  - The temperature is related to how large our denominator $\sqrt{d_k}$ is. As this denominator increases, our weights are more even. 
  - Motivation for $\sqrt{d_k}$: Supposing our entries of $\mathbf{q}_j^i$ and $\mathbf{k}_j^i$ are variance one, this ensures that the entries of our matrix are variance one. Therefore, it's a reasonable scalar to prevent concentrating too much on the most similar keys. 

## Positional Embeddings

- Note that in many language tasks, storing the relative positions is important. 
- However, because the self-attention operation is permutation invariant (if we just use token embeddings), we need to provide additional order information. 
- We do this by _adding_ a matrix $\mathbf{P} \in \mathbb{R}^{L \times d}$
- Types of positional embedding
  - Sinusoidal
    - $P_{ij} = \sin(\frac{i}{10000^\frac{j}{d}})$ if $j$ even, and 
    - $P_{ij} = \cos(\frac{i}{10000^\frac{j-1}{d}})$ if $j$ odd
    - Under this formulation, the wavelengths vary from $2\pi$ to $10000\times2\pi$.
  - Other [types](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#positional-encoding) of embedding include learned, relative and rotary embeddings.
- Note, while it feels more natural to concatenate these embeddings, this is easier to implement and [perhaps](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) we may think of $d$ being large enough to store position and semantic information in different dimensions.

## MLP

- It is important to remember that while attention is the key innovation of transformers, 2/3 of parameters in a transformer is in its linear layers. 
- [This video](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8) provides some useful intuition for what MLP layers do. 
  - Suppose we have input to MLP $\mathbf{X} \in \mathbb{R}^{L \times d}$
  - The MLP layers do $(\operatorname{ReLU}(\mathbf{X}\mathbf{W}^1))\mathbf{W}^2$, where $\mathbf{W}^1 \in \mathbb{R}^{d \times d'}, d' > d$.
  - Note that in such a transformation, _unlike attention_, each of the $L$ tokens do _not_ influence each other, i.e. we can process each token in parallel. 
  - $\mathbf{W}^1$ projects $\mathbf{X}$ into a higher-dimensional vector before $\mathbf{W}^2$ projects it down. 
    - One hypothesis is that each entry (column) of $\mathbf{x}_i\mathbf{W}^1 \in \mathbb{R}^{1 \times d'}$ represents a question, e.g. is $\mathbf{x}_i$ correlated with "Michael Jordan"?
    - $\mathbf{W}^2$ then says, if $\mathbf{x}_i$ is correlated with "Michael Jordan", then add a particular vector (e.g. "basketball") to $\mathbf{x}_i$'s embedding. 
    - As [research on SAEs indicate]((https://transformer-circuits.pub/2023/monosemantic-features/index.html)), the neurons in $\operatorname{ReLU}(\mathbf{X}\mathbf{W}^1))$ are often polysemantic, i.e. under the question paradigm, they correspond to multiple questions.  
  - Potentially unanswered question: Why do we need the MLP layers? What does it do that the attention heads cannot? 
    - Suppose that $\operatorname{softmax}(\frac{\mathbf{Q}^i\mathbf{K}^{i\top}}{\sqrt{d_k}})$ is the identity matrix. 
    - Then we're essentially adding $\mathbf{XW}^{v,i}\mathbf{W}^{o,i}$ to $\mathbf{X}$, which looks very similar?
    - Notable differences
      - $\mathbf{XW}^{v,i}$ is lower dimensional than $\mathbf{XW}^1$, so maybe interference makes it harder to add the correct embeddings? 
      - Lack of activation: If something has a negative activation to "Michael Jordan" (which can happen with 0 correlation but negative bias), what the attention head would do is _subtract_ the "basketball" embedding, which may not be what we want to do. 
    - Or maybe it's just more parameter efficient, since we don't need to compute and store a huge identity matrix. 

## Additional details
- Residual connections
  - This helps to mitigate the vanishing gradients problem. 
  - This also helps us think of transfomers as "adding to the residual stream"
- Learning-Rate Warm Up
  - When training a transformer, we usually gradually increase the learning rate from 0 on to our originally specified learning rate in the first few iterations.
  - Explanations
    - Adam uses the bias correction factors which however can lead to a higher variance in the adaptive learning rate during the first iterations. Improved optimizers like RAdam have been shown to overcome this issue.
    - The iteratively applied Layer Normalization across layers can lead to very high gradients during the first iterations, which can be solved by using Pre-Layer Normalization.
- Layer Normalization
  - We typically use [Layer Normalization](../01_basics/notes.md) to stabilize the network and reduces the training time
  - We don't use batch normalization here because batches tend to be small for language tasks, which could induce high variance in batch statistics.
  - Pre-LN Transformer
    - ![pre_ln.png](pre_ln.png)[Source](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf)
    - In the original (Post-LN) Transformer, gradients in certain layers can be very large.
    - The Pre-LN Transformer normalizes these and eliminate the need for warm up.
- Initialization
  - Xavier initialization should be appropriate, but [BERT and GPT2 initializes weights with a smaller SD of 0.02](https://aclanthology.org/D19-1083.pdf)
  - GPT2 also scales weights of residual layers by $1/\sqrt{N}$, to account for the accumulation on the residual path.
    - ToDo: Understand this better.

## Extensions

- I think [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#combination-of-local-and-global-context) provides a great summary of extensions to the transformer. I will summarize a subset of ideas briefly here.
- A key bottleneck is in the computation of the $\mathbf{Q}^i\mathbf{K}^{i\top}$ matrix, which is $O(L^2d)$. 
- This is why larger context lengths are a big deal! (But also note that they allow for [increased vulnerabilities](../23_safety/03_alignment.md))
- Methods to address this include:
  - Memory methods to "cache" information
  - Methods to selectively incorporate _some_ global context
  - Attention free transformers
    - $Y=f(X) ; Y_t=\sigma_q\left(Q_t\right) \odot \frac{\sum_{t^{\prime}=1}^T \exp \left(K_{t^{\prime}}+w_{t, t^{\prime}}\right) \odot V_{t^{\prime}}}{\sum_{t^{\prime}=1}^T \exp \left(K_{t^{\prime}}+w_{t, t^{\prime}}\right)}$
      - I personally wonder how similar this is to normal transformers. To me, the hadamard product would distort values significantly. 
