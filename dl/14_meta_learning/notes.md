# Meta Learning

- Meta learning is where a model is trained to adapt and learn new tasks quickly with minimal data. 
- Few-shot classification is an instantiation of meta-learning in the field of supervised learning. 
  - The dataset $\mathcal{D}$ is often split into two parts, a support set $\mathcal{S}$ for learning and a prediction set $\mathcal{B}$ for training or testing, 
  - K-shot N-class classification task: the support set contains K labelled examples for each of N classes.
- There are three common approaches to meta-learning: Metric-based, model-based and optimization-based

## Metric-Based
- The predicted probability is a weighted sum of labels of support set samples:
  - $P_\theta(y \mid \mathbf{x}, S)=\sum_{\left(\mathbf{x}_i, y_i\right) \in S} k_\theta\left(\mathbf{x}, \mathbf{x}_i\right) y_i$
  - This is similar to nearest neighbors algorithms and kernel density estimation.
  - ProtoNet:
    - We first embed each datapoint $f_\theta(\mathbf{x}_i)$
    - For each class $c$, we define the _prototype_ $\mathbf{v}_c$ for that class as the mean of embeddings in that class. 
    - $p(y=c\mid\mathbf{x}) = \operatorname{softmax}(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_c))$, where we typically use the squared euclidean distance for our distance metric $d_\varphi$.

## Model-Based
- We depend on a model designed specifically for fast learning - a model that updates its parameters rapidly with a few training steps.
- Memory-Augmented Neural Networks
  - The intuition here is that we use external memory as a sort of "cache". 
  - The model's parameters are sensitive to what's stored in the cache, and this cache is frequently updated with recent samples. 
  - For predictions, we then use attention where the query is a function of input $\mathbf{x}$, and the memory matrix are our keys. 

## Optimization-Based
- Instead of having a model designed specifically for fast learning, we use an optimization algorithm that allows a model to be good at learning with a few examples. 
- Model-Agnostic Meta Learning (MAML)
  - The idea is to find a good initialization that is suitable to learning any task quickly. 
  - How good is an initialization $\theta$? Well we can sample a bunch of support and prediction sets, and calculate the loss when finetuning on them, when starting from $\theta$.
  - Concretely, we update $\theta$ with gradient descent:
    - Sample a batch of tasks $\mathcal{T}_i\sim p(\mathcal{T})$
    - Compute loss when finetuning for $\mathcal{T}_i$, starting from $\theta$
      - $g(\theta, \mathcal{T}_i) = \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(\theta)) := \mathcal{L}_{\mathcal{T}_i}(\theta_i)$
    - Update $\theta = \theta - \beta\nabla_\theta\left[\sum_{\mathcal{T}_i \sim p(\mathcal{T})}g(\theta,\mathcal{T}_i)\right]$
  - First-order MAML
    - The above is computationally expensive because it requires second-order gradients
    - An approximation: $\theta = \theta - \beta\sum_{\mathcal{T}_i \sim p(\mathcal{T})} \nabla_{\theta_i} \mathcal{L}_{\mathcal{T}_i}(\theta_i)$
- Proto-MAML
  - MAML intializes $\theta$ the same regardless of which $\mathcal{T}_i$ is chosen. 
  - Proto-MAML uses the concept of prototypes to initialize $\theta$ based on the task at hand. 
