# Unsupervised Clustering

- K-means
  - K-means initializes with $k$ clusters and assigns each $\mathbf{x}_i$ to a cluster, aiming to minimize the distance from every point to the centroid of it's cluster.
  - Algorithm:
    - Initialize $k$ centers randomly. 
    - Assign each point to the center closest to it. 
    - Update the center to the centroid of all points assigned to it.
    - Repeat from step 2 until converge.
  - To pick $k$, we can plot the loss versus $k$ and use the "elbow" method. 
  - K-means may sometimes have high variance and be sensitive to initialization. 
  - K-means also considers distance in the euclidean space, i.e. clusters are circular.
- DBSCAN works as follows (per [Géron](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975)):
  - For each instance, the algorithm counts how many instances are located within a small distance ε (epsilon) from it. This region is called the instance’s 
ε-neighborhood.
  - If an instance has at least min_samples instances in its ε-neighborhood (including itself), then it is considered a core instance. In other words, core instances are those that are located in dense regions. 
  - All instances in the neighborhood of a core instance belong to the same cluster. This neighborhood may include other core instances; therefore, a long sequence of neighboring core instances forms a single cluster. 
  - Any instance that is not a core instance and does not have one in its neighborhood is considered an anomaly.
- Spectral Clustering
  - We quote an [_excellent_ tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf)
  - ![Screenshot 2024-12-15 at 8.00.29 AM.png](Screenshot%202024-12-15%20at%208.00.29%20AM.png)
  - The linked article elaborates how this is related to the RatioCut problem.
    - A key piece of intuition is that $U_{ik} > 0$ implies that the $i^{th}$ datapoint is likely to belong to the $k^{th}$ cluster. 
- GMM and EM
  - Soft probabilities
  - Elliptical
  - Data generation and priors
  - EM