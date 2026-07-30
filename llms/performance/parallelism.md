# Parallelism

Training and serving across multiple devices. Primary source: the JAX scaling book — [sharding](https://jax-ml.github.io/scaling-book/sharding/), [training](https://jax-ml.github.io/scaling-book/training/), [applied training](https://jax-ml.github.io/scaling-book/applied-training/). Also some from [Lippe's notes](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/overview.html). See also [Basics](basics.md), [Inference](inference.md), [TPUs & Rooflines](tpus.md).

> **Skeleton status.** Section scaffolding + scope hints below; existing Lippe-derived content is kept in place and marked _(existing)_. Redundancy watch is at the bottom — things to delete once the scaling-book notes land.
>
> Note on the [transformers chapter](https://jax-ml.github.io/scaling-book/transformers/): its params/FLOPs accounting (6ND, per-layer FLOPs table, attention share) is already written up in [Basics §Training vs Inference](basics.md#training-vs-inference). The part still needed *here* is the per-layer weight/activation **shapes**, since those set the communication volumes below.

## Sharding and collectives

_From [sharding](https://jax-ml.github.io/scaling-book/sharding/). This is the primitives layer — everything in §Strategies is an application of it._

### Notation

_To cover: device mesh + axis names · sharding spec and how a logical array maps to per-device local shapes · local vs global shape · the unreduced `{U}` suffix · JAX `NamedSharding`/`PartitionSpec`._

### Which sharding needs which collective

_To cover: the four matmul cases by where the contracting dimension is sharded — (1) neither sharded → no comms, (2) one sharded → AllGather, (3) both sharded → AllReduce / ReduceScatter, (4) same axis on both non-contracting dims → AllGather to fix. The point to extract: **the sharding determines the collective**, mechanically._

### The collectives and their costs

_To cover, each with its cost model and whether it's bandwidth- or latency-bound: AllGather · ReduceScatter · AllReduce (= AllGather ∘ ReduceScatter, 2× cost) · AllToAll (~¼ of AllGather). Also: ReduceScatter is AllGather's transpose, which is why they swap in the backward pass._

- _(existing)_ Gather vs Scatter
  - Gather "gathers" data spread across multiple processors such that each processor has a copy.
  - Scatter does not copy data, rather it transmits $`\frac{n-1}{n}`$ of its data to other devices.
  - ![gather_scatter.png](images/gather_scatter.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_simple.html)
  - Note that here we reverse the order of our activations ($`p \times n`$ rather than $`n \times p`$)
  - To reduce both the communication needed (?) and the amount of data stored on each device, gather/scatter is more suitable when $`\mathbf{x}`$ has fewer/more features than $`\mathbf{y}`$.
- _(existing)_ Ring AllReduce: if we want all nodes to contain all activations, use scatter-reduce followed by an allgather — this sums individual arrays on all nodes, and eventually every node has a copy of the sum.
  - Gather ![gather.png](images/gather.png) · Scatter ![scatter.png](images/scatter.png) [Source](https://arxiv.org/pdf/2302.05442)

### Overlapping communication with compute

_To cover: collective matmul / how a sharded matmul hides its AllGather behind the compute; when overlap is possible vs when you stall._

- _(existing)_ Asynchronous layers
  - In the gather strategy, we first need to communicate all the features of $`\mathbf{x}`$ before we can compute the output.
    - ![async_gather.png](images/async_gather.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_async.html)
  - In the scatter strategy, we need to compute the output on all devices before we can communicate results and sum them.
    - ![async_scatter.png](images/async_scatter.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_async.html)
  - Asynchronous layers allow us to overlap communication with computation and reduce downtime.
- _(existing)_ In PyTorch, functions like `to()` and `copy_()` admit an explicit `non_blocking` argument. We can also do this in JAX.

## Strategies

_From [training](https://jax-ml.github.io/scaling-book/training/). For each strategy the same three questions: **what is sharded**, **what is communicated per step**, **at what per-device batch size does it go bandwidth-bound**. Worth keeping the answers in that parallel shape so the summary table at the end writes itself._

### Data parallelism

_To cover: params/optimiser replicated, activations sharded by batch · gradient AllReduce each step · the per-device batch threshold for staying compute-bound · what it does **not** fix (memory)._

- _(existing)_ Overview
  - Each device will hold the same model and parameters, and process a different batch of data in parallel.
  - After obtaining the gradients for each batch, we aggregate the gradients over the devices and update our model.
    - This is synchronous SGD, but this may be slowed down due to communication overhead.
    - Asynchronous SGD can be used, although there may be gradient staleness. However, when weight matrices are large, most updates are sparse and gradient staleness may be ok.
    - `DP` has all communication go through a master process, which is slower than `DDP`, which uses Ring-AllReduce.

### Fully-sharded data parallelism (FSDP / ZeRO-3)

_To cover: params + optimiser state also sharded · AllGather weights just-in-time per layer, ReduceScatter gradients · why this is ~the same total comms as DP but far less memory · the optimal-sharding expression for how many axes to give FSDP._

- _(existing)_ Parameter Sharding
  - Storing _all_ of a model's data (optimizer state, gradients, parameters) can be costly in terms of memory
  - Each device can instead store a portion of parameters
  - Before executing a layer, a device can then communicate with other devices to receive the parameters it needs

### Tensor parallelism

_To cover: activations sharded along the model dim, weights along the feedforward dim · comms **per layer** (not per step) — the reason it needs fast interconnect · the condition on the TP axis size beyond which it goes comms-bound._

- _(existing)_ Overview
  - Tensor parallelism splits the model across the feature dimension.
  - It does not face the pipeline bubble issue, but requires more communication across devices.

### Pipeline parallelism

_Note: the scaling book is deliberately light here (TPU topologies rarely need PP), so the existing Lippe material is the substance for this section rather than something to replace. To add from the book: why pipelining is comparatively unattractive on TPU, and where its comms sit relative to TP._

- _(existing)_ Overview
  - Pipeline parallelism splits the model across devices, whilst introducing minimal communication across devices, although also facing the pipeline bubble issue.
  - ![pipeline1.png](images/pipeline1.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
- _(existing)_ Micro-Batching
  - Micro-Batching mitigates the pipeline bubble issue.
  - ![pipeline2.png](images/pipeline2.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
- _(existing)_ Looping Pipelines
  - Looping mitigates the pipeline bubble issue further.
  - ![pipeline3.png](images/pipeline3.png)![pipeline4.png](images/pipeline4.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_looping.html)
  - We can process mini-batches breadth-first (each GPU processes a batch fully before moving on to the next) or depth-first (process a mini-batch the moment it is ready)
  - [Poirier](https://arxiv.org/pdf/2211.05953) argues that when combining data parallelism and pipeline parallelism, because the former requires us to communicate and sum across devices, breadth-first pipeline parallelism is faster since we can start this communication earlier.

### Combining them

_To cover: FSDP + TP as the standard 2D mesh (shard batch on one axis, model on the other) · why the combination raises the usable chip count · the minimum batch-to-chip ratio it requires · where PP enters to make it 3D._

- _(existing)_ We can combine all 3 parallelism types for increased computational gains.
  - ![3d.png](images/3d.png)[Source](http://web.ecs.baylor.edu/faculty/dong/elc5396_DeepLearning/DeepLearningSignalProcessingH3.pdf)
  - [DeepSpeed](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

## Choosing a strategy

_From [training §Takeaways](https://jax-ml.github.io/scaling-book/training/). The payoff section: one table of strategy → what's sharded → comms → the threshold that kills it, so the decision becomes lookup rather than derivation. Also: scaling across pods over DCN and the batch-size requirement that imposes._

## Case study: LLaMA-3-70B on TPU v5p

_From [applied training](https://jax-ml.github.io/scaling-book/applied-training/). Worth doing end-to-end as the integration test of everything above: param/FLOP count → memory budget (params, optimiser, checkpoints) → why pure FSDP runs out → the mixed TP+FSDP choice from the derived formulas → MFU (~40%) → wall-clock estimate (~44 days on a full pod)._

## Placement on GPUs: TP within a node, PP across nodes

_(existing — from the [vLLM blog](https://www.aleksagordic.com/blog/vllm); complementary to the book, which is TPU/ICI-centric rather than node-centric.)_

- A node is a server that may contain one or multiple GPUs.
- If a model doesn't fit on one GPU, the first option is to shard it across multiple GPUs on the same node using tensor parallelism (e.g. TP=8). If the model still doesn't fit, the next step is pipeline parallelism across nodes.
- Intranode bandwidth is significantly higher than internode, which is why TP is generally preferred over PP (it is also true that PP communicates less data than TP):
  - Tensor Parallelism usually stays within a single node (intra-node communication) because it involves fine-grained operations with very high bandwidth and low-latency needs (like splitting matrix multiplications).
  - Pipeline Parallelism usually spans across nodes (inter-node communication) because it splits the model into larger chunks (layers or blocks), and the data passed between chunks is relatively smaller and less frequent, making it more tolerant to slower communication across nodes.
- The next step is to scale out: enable data parallelism (DP > 1) replicating the model across nodes, add a lightweight DP coordination layer, introduce load balancing across replicas, and place one or more API servers in front to handle incoming traffic.

---

### Redundancy watch

Things to prune once the scaling-book notes land, so the two treatments don't sit side by side:

- **Gather vs Scatter** (§Collectives) — Lippe's framing of the same primitives the book covers as AllGather/ReduceScatter with proper cost models. Keep the figures, drop the prose once the collectives section exists. The "(?)" in that bullet should resolve on the way.
- **Ring AllReduce** — the book derives AllReduce = AllGather ∘ ReduceScatter with costs; this bullet becomes a one-line implementation note.
- **DP overview** and **Parameter Sharding** — will be superseded by the DP/FSDP sections' cost analysis. The bits worth *keeping* are the ones the book doesn't cover: sync vs async SGD, gradient staleness, and `DP` vs `DDP` (PyTorch-specific).
- **TP overview** — two lines that the TP section will subsume entirely.
- **3D parallelism** — one figure plus a link; the book's FSDP+TP analysis replaces the prose.
- Also check for overlap against [Basics](basics.md): the ridge-point / arithmetic-intensity machinery is already there, and the training chapter's thresholds are all instances of it — cross-reference rather than restate.
