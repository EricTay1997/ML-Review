# Interpretability

## Beauty and Curiosity

While its potential use case in safety is certainly very meaningful, the selfish part of myself truly finds interpretability research inherently interesting. Here, I've resonated very strongly with [Olah's thoughts on the topic](https://transformer-circuits.pub/2023/interpretability-dreams/index.html):

"While our goal is safety, we also believe there is something deeply beautiful hidden inside neural networks, something that would make our investigations worthwhile even in worlds with less pressing safety concerns. With progress in deep learning, interpretability is **the** research question which is just crying out to be answered! ... The success of deep learning is often bemoaned as scientifically boring. One just stacks on layers, makes the models bigger, and gets lower loss. Elegant and clever ideas are often unnecessary. ... Neural networks are full of beautiful structure, if only we care to look for it."

## General Techniques
- LIME
  - Local surrogate models are trained to approximate the predictions of the underlying black box model.
- Feature importance
  - Shapley Values
    - We can randomize the inputs of a particular feature to see how important it was in creating accurate predictions.
  - Model-specific techniques, e.g. impurity reduction in trees, attention maps
    - Note: Especially when looking at how these features affect "localized" statistics, without considering its effect on overall predictions, we could potentially draw erroneous conclusions. 
      - See [Attention is not Explanation](https://arxiv.org/pdf/1902.10186) for more, and [Anthropic's article regarding transformers](https://transformer-circuits.pub/2021/framework/index.html#related-work) for an example of possible misinterpretation. 
- [Causal ML](../../classical/14_causal_inference/notes.md) 

## Mechanistic Interpretability 
- [Zoom In: An Introduction to Circuits (2020)](https://distill.pub/2020/circuits/zoom-in/)
  - Takeaways
    - 3 speculative claims about neural nets:
      - Features: Features are the fundamental unit of neural networks.
They correspond to directions.
      - Circuits: Features are connected by weights, forming circuits.
      - Universality: Analogous features and circuits form across models and tasks.
- [A Mathematical Framework for Transformer Circuits (2021)](https://transformer-circuits.pub/2021/framework/index.html)
  - Useful Videos
    - [Attention in transformers, visually explained | DL6](https://www.youtube.com/watch?v=eMlx5fFNoYc&pp=ygUPM2JsdWUxYnJvd24gZGw2)
    - [A Walkthrough of A Mathematical Framework for Transformer Circuits](https://www.youtube.com/watch?v=KV5gbOmHbjU)
  - Takeaways
    - Reverse engineering results
      - Zero layer transformers model bigram statistics. The bigram table can be accessed directly from the weights.
      - One layer attention-only transformers are an ensemble of bigram and “skip-trigram” (sequences of the form "A… B C") models. The bigram and skip-trigram tables can be accessed directly from the weights, without running the model. 
      - Two layer attention-only transformers can implement much more complex algorithms using compositions of attention heads. These compositional algorithms can also be detected directly from the weights. Notably, two layer models use attention head composition to create “induction heads”, a very general in-context learning algorithm.
        - The paper notes that induction heads can be used to get the next word for the previous instance of the destination token. 
        - [Here](https://transformer-circuits.pub/2021/exercises/index.html) are some exercises for how we can do this:
          - Let the pattern be [A1][B]...[A2]->[B], referencing [Ma's notes](https://johnma2006.github.io/papernotes.html)
          - Q-Composition: $W_Q$ reads in a subspace affected by the previous head. (Exercise 2) 
            - Head 1:
              - QK circuit: [A2] attends to [A1] through content.
              - OV circuit: Puts position of [A1] in [A2]'s residual stream.
            - Head 2:
              - QK circuit: $W_Q$ reads position of [A1] and shifts forward (Q-composition), $W_K$ stores positions.
              - OV circuit: Puts content of [B] in [A2]'s residual stream.
          - K-Composition: $W_K$ reads in a subspace affected by the previous head. (Exercise 3)
            - Head 1:
              - QK circuit:[B] attends to prev token [A1] through position.
              - OV circuit: Puts content of [A1] in [B]'s residual stream.
            - Head 2:
              - QK circuit: [A2] attends to [B] through content (K-composition).
              - OV circuit: Puts content of [B] in [A2]'s residual stream.
    - Conceptual takeaways
      - Attention heads can be understood as independent operations, each outputting a result which is added into the residual stream.
        - Note that authors also show how "adding" can be "deleting"
      - Attention-only models can be written as a sum of interpretable end-to-end functions mapping tokens to changes in logits. 
      - Attention heads can be understood as having two largely independent computations: a QK (“query-key”) circuit which computes the attention pattern, and an OV (“output-value”) circuit which computes how each token affects the output if attended to.
      - Composition of attention heads greatly increases the expressivity of transformers. There are three different ways attention heads can compose, corresponding to keys, queries, and values. Key and query composition are very different from value composition.
- [Toy Models of Superposition (2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
  - Useful Videos
    - [The Geometry of AI Minds in Superposition](https://www.youtube.com/watch?v=qGQ5U3dkZzk)
    - [A Walkthrough of Toy Models of Superposition w/ Jess Smith](https://www.youtube.com/watch?v=R3nbXgMnVqQ)
  - Takeaways
    - Definitions:
      - Feature
        - Intuition: Interpretable properties of the input we observe neurons (or word embedding directions) responding to.
        - 3 Working definitions
          - Arbitrary functions of the input. 
          - Interpretable properties.
          - Neurons in sufficiently large models. 
        - Directions. Regardless of the definition, we view features as directions in activation space.
      - Superposition
        - Superposition is when the neural network represents more features than dimensions by tolerating a bit of interference.
        - Intuitively, superposition is a form of lossy compression.
        - This gives rise to polysemanticity, where a neuron responds to multiple unrelated features.
        - The real world has a limitless number of features, and if we could, we may prefer an extremely large neural network composed of monosemantic neurons.
      - Privileged Basis
        - A privileged basis is a meaningful basis for a vector space. The main important example of a privileged basis is the basis of neuron directions immediately after an elementwise non-linearity.
    - Results
      - In experimenting with a toy model, authors found that:
        - Superposition could be observed
        - Two key aspects of a feature affects superposition: its importance and sparsity. 
          - Generally, problems with many sparse, unimportant features will show significant superposition.
        - Superposition seems to "cluster" different features together
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
  - Useful Videos
    - [Reading an AI's Mind with Sparse Autoencoders](https://www.youtube.com/watch?v=krINuMZhJmU&t=562s)
  - Takeaways
- [How to use and interpret activation patching (2024)](https://arxiv.org/pdf/2404.15255)
  - Takeaways
- [Mapping the Mind of a Large Language Model (2024)](https://www.anthropic.com/news/mapping-mind-language-model)
  - Takeaways