# Graph Neural Networks

- Types of problems
  - Graphs-level: Property of entire graph
  - Node-level: Property of nodes
  - Edge-level: Property of edge
- Technical Details
  - Each layer is a graph with the same adjacency properties
  - Going from layer to layer, we can have an MLP for node to node, edge to edge, and master node to master node
  - Node to node MLPs
    - GCN
      - $H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)$,
      - where $\hat{A}=A+I$, $A$ being the adjecency matrix, and $\tilde{D}$ is a diagonal matrix with $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ denoting the number of neighbors node $i$ has
    - GraphConv
      - Addresses potential issue in GCN where the network forgets node-specific information
      - $
\mathbf{x}_i^{(l+1)} = \mathbf{W}^{(l + 1)}_1 \mathbf{x}_i^{(l)} + \mathbf{W}^{(\ell + 1)}_2 \sum_{j \in \mathcal{N}_i} \mathbf{x}_j^{(l)}
$
    - Graph Attention
      - Idea is to not just 'average' across neighbors, but to weight edges with attention.
      - Attention is implemented as a one-layer MLP
      - $h_i'=\sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}h_j\right)$
      - $\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_j\right]\right)\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_k\right]\right)\right)}$, where $\alpha_{ij}$ is the attention weight from node $i$ to $j$, having been masked appropriately for adjacency $A$
      - $\mathbf{a} \in \mathbb{R}^{1 \times 2d}$
      - $d$ being the dimensionality of $\mathbf{W}h_i$
      - Leaky ReLU is needed to ensure that attention is dependent of the node itself, if not nodes with the same neighbors will have the same attention.
- Additional thoughts:
  - Intuition - why is a GNN useful? By maintaining the adjacency properties in each layer, you may “pool” (message passing) information effectively to leverage the graph structure
  - GNNs vs CNNs: Perhaps we can view CNNs as a special case of GNNs, where CNNs draw edges between pixels (nodes) that are close to each other. Edit: Seems like there’s a decent amount of literature on this topic.
  - Generalization - The output of the graph is an understanding of how a node relates to neighboring nodes. This is then extendable to graphs with different structures (e.g. # nodes)
- Real-life use cases
  - [AI trends in 2024: Graph Neural Networks](https://www.assemblyai.com/blog/ai-trends-graph-neural-networks/)