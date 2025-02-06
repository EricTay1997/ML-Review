# Mechanistic Interpretability

## Introduction

- Definitions
  - Per [Sharkey](https://arxiv.org/pdf/2501.16496), there are 3 threads of interpretability:
    - Build AI systems that are interpretable by design - think simpler models, or explaining the sensitivity of machine learning models to inputs and training data.
      - Feature importance
        - Shapley Values
          - We can randomize the inputs of a particular feature to see how important it was in creating accurate predictions.
        - Model-specific techniques, e.g. impurity reduction in trees, attention maps
          - Note: Especially when looking at how these features affect "localized" statistics, without considering its effect on overall predictions, we could potentially draw erroneous conclusions. 
            - See [Attention is not Explanation](https://arxiv.org/pdf/1902.10186) for more, and [Anthropic's article regarding transformers](https://transformer-circuits.pub/2021/framework/index.html#related-work) for an example of possible misinterpretation. 
    - Why did my model make this particular decision?
      - LIME: Local surrogate models are trained to approximate the predictions of the underlying black box model.
    - Mech Interp: How did my model solve this general class of problems? 
      - Emphasis on the mechanisms underlying neural network generalization
- Approaches
  - Reverse engineering: decompose the network into components and then attempt to identify the function of those components
    - Think SAEs
  - Concept-based interpretability: set of concepts that might be used by the network and then looks for components that appear to correspond to those concepts
    - Think probes
- Applications
  - Monitor AI systems for signs of cognition related to dangerous behavior
  - Modify internal mechanisms and edit model parameters to adapt their behavior to better suit our needs
  - Predict how models will act in unseen situations or predict when a model might learn specific abilities
  - Improve model inference, training, and mechanisms to better suit our preferences
  - Extract latent knowledge from AI systems so we can better model the world

## Beauty and Curiosity

While its potential use case in safety is certainly very meaningful, the selfish part of myself truly finds interpretability research inherently interesting. Here, I've resonated very strongly with [Olah's thoughts on the topic](https://transformer-circuits.pub/2023/interpretability-dreams/index.html):

"While our goal is safety, we also believe there is something deeply beautiful hidden inside neural networks, something that would make our investigations worthwhile even in worlds with less pressing safety concerns. With progress in deep learning, interpretability is **the** research question which is just crying out to be answered! ... The success of deep learning is often bemoaned as scientifically boring. One just stacks on layers, makes the models bigger, and gets lower loss. Elegant and clever ideas are often unnecessary. ... Neural networks are full of beautiful structure, if only we care to look for it."

## Tools

- Tools like Garcon or TransformerLens are very useful in doing work here.
- Demos ([link](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb), [link](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)) show how we can
  - Cache, access and modify activations for attribution/ablation
  - Visualize attention heads to identify induction heads
  - Reverse engineer induction circuits

## Papers
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
  - Authors use a sparse autoencoder (SAE) to decompose the MLP activations of a one-layer transformer into relatively interpretable features.
    - The SAE has two layers:
      - The first layer (“encoder”) maps the MLP activations to a higher-dimensional layer via a learned linear transformation followed by a ReLU nonlinearity. We refer to the units of this high-dimensional layer as “features.” 
      - The second layer (“decoder”) attempts to reconstruct the model activations via a linear transformation of the feature activations. 
    - The model is trained to minimize a combination of (1) reconstruction error and (2) an L1 regularization penalty on the feature activations, which incentivizes sparsity.
  - Why not architectural approaches:
    - Superposition is defined as when a neural network represents more independent "features" of the data than it has neurons by assigning each feature its own linear combination of neurons
    - The paper shows that even without explicit assignment (superposition), cross-entropy loss can still promote polysemanticity. 
      - Models achieve lower loss by representing multiple features ambiguously (in a polysemantic neuron) than by representing a single feature unambiguously and ignoring the others.
      - I think the argument really is that encouraging activation sparsity does not prevent polysemanticity.
  - Results
    - Sparse Autoencoders extract relatively monosemantic features
    - Sparse autoencoders produce interpretable features that are effectively invisible in the neuron basis
    - Sparse autoencoder features can be used to intervene on and steer transformer generation
    - Sparse autoencoders produce relatively universal features
    - Features appear to "split" as we increase autoencoder size
    - Just 512 (MLP activation) neurons can represent tens of thousands of features
    - Features connect in "finite-state automata"-like systems that implement complex behaviors
- [How to use and interpret activation patching (2024)](https://arxiv.org/pdf/2404.15255) 
  - Activation patching
    - Technique of replacing internal activations of a neural net
    - 2 types of experiments:
      - Exploratory: patch components one at a time, get an idea of which parts of a model are involved in the task in question, and may be part of the corresponding circuit.
      - Confirmatory: patch many model components at a time, confirm a hypothesised circuit by verifying that it actually covers all model components needed to perform the task in question.
    - This work focuses on communicating useful practical advice for activation patching:
      - What kind of patching experiments provide which evidence?
      - How should you interpret activation patching results?
      - What metrics you can use, what are common pitfalls?
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
  - Useful Summary
    - [Mapping the Mind of a Large Language Model](https://www.anthropic.com/news/mapping-mind-language-model)
  - Researchers were able to scale SAEs to extract interpretable features from Claude 3 Sonnet. 
  - Key Results
    - Sparse autoencoders produce interpretable features for large models.
    - Scaling laws can be used to guide the training of sparse autoencoders.
      - Over the ranges tested, given the compute-optimal choice of training steps and number of features, loss decreases approximately according to a power law with respect to compute.
      - As the compute budget increases, the optimal allocations of FLOPS to training steps and number of features both scale approximately as power laws.
    - The resulting features are highly abstract: multilingual, multimodal, and generalizing between concrete and abstract references.
    - Researchers were able to measure a kind of "distance" between features based on which neurons appeared in their activation patterns. Looking for features that are "close" to each other seemed to yield features that were conceptually related.
    - There appears to be a systematic relationship between the frequency of concepts and the dictionary size needed to resolve features for them.
      - If a concept is present in the training data only once in a billion tokens, then we should expect to need a dictionary with on the order of a billion alive features in order to find a feature which uniquely represents that specific concept. 
    - **Features can be used to steer large models. This extends prior work on steering models using other methods. (!!)**
    - **We observe features related to a broad range of safety concerns, including deception, sycophancy, bias, and dangerous content. (!!)**