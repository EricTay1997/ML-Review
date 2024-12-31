# Contrastive Learning

- Contrastive Learning is when models learn to differentiate between similar and dissimilar data points in an unsupervised manner. 
- This "differentiation" is expressed by learning how to embed data points. 
- Such embeddings are useful for downstream tasks like classification. 
- SimCLR
  - ![simclr.png](simclr.png)[Source](https://simclr.github.io)
  - Loss:
    - $
\ell_{i,j}=-\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)}
$
    - A similarity metric like cosine similarity can be used.
  - Data Augmentation
    - Crop-and-resize, and color distortion are particularly useful especially when used together. 
    - The former without the latter may allow the model to simply pick up non-subject matter details only relevant to a particular image.