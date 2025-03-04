# Computational Performance

As models and data scale in size, optimizing for more efficient processes becomes more and more imperative. This section will cover non-algorithmic ways we may do so, drawing heavily from [Lippe's notes](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/overview.html) on this topic.

![overview.png](overview.png)(Adapted from [Lippe](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/overview.html))

ToDo: Add notes from [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)

## Single Processor

- CPU vs GPU
  - CPUs and GPUs are processors
  - GPUs have many smaller, more specialized cores, which make it suited for parallel processing, e.g. tensor cores which compute matrix-matrix multiplications quickly.
  - Terminology differences
    - A GPU is formed by multiple units named SM (Streaming Multiprocessors), these function like CPU cores.
    - Each SM can execute many threads concurrently.
    - Threads are grouped into warps, a basic execution unit, where each warp contains 32 threads. These function like CPU threads.
    - While CPU threads can each execute different tasks at the same time, all GPU threads in a single warp can only execute one same task. 
    - A threadblock is a collection of warps. The threads in the same thread block run on the same SM. 
  - PyTorch (amongst other libraries) allow us to use these tensor cores for training DL models.
- Row-major vs Column-major
  - Row/column-major means that consecutive elements in a row/column are stored next to each other in memory. 
  - NumPy/PyTorch/CSV are row-major, Parquet is column-major. 
  - In a sample $\times$ feature matrix, it is faster to access samples/features in row/column-major formats.
- Vectorization
  - Vectorization refers to single instruction, multiple data (SIMD) operations. 
    - I.e. One instruction carries our the same operation on a number of operands in parallel.
  - NumPy enables vectorization when we write code in a way that operates on entire arrays rather than looping through individual elements. 
- Imperative vs Symbolic programming
  - Imperative programming makes it easy to design new models since it is possible to write code with control flow and the ability to use a large amount of the Python software ecosystem.
  - Symbolic programming requires that we specify the program and compile it before executing it. The benefit is improved performance.
- Asynchronous Computation
  - For PyTorch, by default, GPU operations are asynchronous.
  - Broadly speaking, PyTorch has a frontend for direct interaction with the users, e.g., via Python, as well as a backend, e.g. via C++, used by the system to perform the computation.
  - Thus, there is little impact on the program’s overall performance, regardless of Python’s performance.
  - Conversions to NumPy are blocking because NumPy has no notion of asynchrony.
- `torch.compile` makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels
  - A Just-In-Time (JIT) compiler compiles code at runtime into a fast executable
  - The `max-autotune` configuration with profile the model with different optimization configurations and generate optimized machine code for the model using the best found configuration
- JAX
  - JAX is a numerical computing library that has various desirable characteristics for the computations done in DL. 
    - Provides a unified NumPy-like interface to computations that run on CPU, GPU, or TPU, in local or distributed settings
    - Features Just-In-Time (JIT) compilation via Open XLA
      - XLA significantly increases execution speed and lowers memory usage by fusing low-level operations
      - Warning: The intermediate `jaxpr` representation is specialized to the shapes of input arguments 
        - Hence, running a jitted function with different input shapes requires multiple recompilations. 
        - We can use padding to prevent re-compilations, but when this needs to be done extensively (e.g. NLP with many different sentence lengths), the overhead could outweigh the benefits.
      - While compilation time could be a significant bottleneck, we can use the `scan` transformation to write a for-loop with a single compilation of the inner step.
    - Efficiently evaluates gradients via its automatic differentiation transformations
      - `jaxpr` representations give us analytical forms of gradients
      - Allows us to efficiently compute higher-order gradients
    - Supports automatic vectorization of functions
      - Allows for vectorization of functions not written in "vectorized forms".
      - It also allows us to support additional batch dimensions.
    - A note: JAX is designed to be functional. 
      - Writing code with side effects is dangerous because an error will _not_ be thrown and JAX will just ignore such instructions.
- What are we bounded by?
  - [He's article](https://horace.io/brrr_intro.html)
  - Memory
    - Size of DRAM
      - Solution: See below
  - Bandwidth
    - Time spent transferring tensors within a GPU
      - Solution: Operator fusion
  - Compute (on SRAM)
    - Time spent on your GPU computing actual floating point operations (FLOPS)
      - Solution: More tensor cores
  - Overhead
    - Everything else
      - Solution: Asynchronous computation
- Memory reduction
  - Memory vs compute
    - We discuss methods to reduce this memory constraints (sometimes at the cost of increased computational cost)
  - Mixed Precision Training
    - Use 32-bit floating-point numbers for weight updates and final loss computation
    - Use 16-bit floating-point numbers for most computations
      - Loss scaling may be needed because `float16` may induce underflow/overflow issues
      - `bfloat16` has a larger range but lower precision, and is an alternative to avoid loss scaling
    - This reduces both memory and compute costs
  - Quantization
    - We represent the weights and activations with lower-precision data types
    - Quantization-aware Training (QAT) is a way of training that simulates quantization whilst training
    - Double quantization is when we quantize the scaling factors from the first quantization.
      - QLoRA combines double quantization with [LoRA](../22_post_training/notes.md).
  - Gradient Checkpointing / Activation Recomputation
    - Trade compute for memory by recomputing activations during the backward pass.  
  - Gradient Accumulation
    - We can accumulate gradients over batches and take steps once every few batches.
    - This to me doesn't feel like it "speeds up" a forward pass. Rather, it just remedies the instability induced by memory limitations that force smaller batch sizes than we would like.
  - Pruning
    - Pruning is a technique that removes less important connections, neurons, or structures from a trained model 
  - Donating buffers (JAX-specific)
    - Since JAX employs functional programming, we cannot modify variables in place.
    - If we don't need our input variables, JAX provides a mechanism to donate buffers, which allows us to reuse the memory of the input arguments for the output arguments.

## Multiple Processors

- Parallel Computation and Communication
  - In PyTorch, functions like `to()` and `copy_()` admit an explicit `non_blocking` argument. 
  - We can also do this in JAX

### Data Parallelism

- Overview
  - Each device will hold the same model and parameters, and process a different batch of data in parallel.
  - After obtaining the gradients for each batch, we aggregate the gradients over the devices and update our model. 
    - This is synchronous SGD, but this may be slowed down due to communication overhead.
    - Asynchronous SGD can be used, although there may be gradient staleness. However, when weight matrices are large, most updates are sparse and gradient staleness may be ok.
    - `DP` has all communication go through a master process, which is slower than `DDP`, which uses Ring-AllReduce. 
- Parameter Sharding (Fully-sharded data parallelism)
  - Storing _all_ of a model's data (optimizer state, gradients, parameters) can be costly in terms of memory
  - Each device can instead store a portion of parameters
  - Before executing a layer, a device can then communicate with other devices to receive the parameters it needs

### Pipeline Parallelism

- Overview
  - Pipeline parallelism splits the model across devices, whilst introducing minimal communication across devices, although also facing the pipeline bubble issue. 
  - ![pipeline1.png](pipeline1.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
- Micro-Batching
  - Micro-Batching mitigates the pipeline bubble issue.
  - ![pipeline2.png](pipeline2.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
- Looping Pipelines
  - Looping mitigrates the pipeline bubble issue further.
  - ![pipeline3.png](pipeline3.png)![pipeline4.png](pipeline4.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_looping.html)
  - We can process mini-batches breadth-first (each GPU processes a batch fully before moving on to the next) or depth-first (process a mini-batch the moment it is ready)
  - [Poirier](https://arxiv.org/pdf/2211.05953) argues that when combining data parallelism and pipeline parallelism, because the former requires us to communicate and sum across devices, breadth-first pipeline parallelism is faster since we can start this communication earlier. 

### Tensor Parallelism

- Overview
  - Tensor parallelism splits the model across the feature dimension. 
  - It does not face the pipeline bubble issue, but requires more communication across devices.
  - Gather vs Scatter
    - Gather "gathers" data spread across multiple processors such that each processor has a copy. 
    - Scatter does not copy data, rather it transmits $\frac{n-1}{n}$ of its data to other devices. 
    - ![gather_scatter.png](gather_scatter.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_simple.html)
    - Note that here we reverse the order of our activations ($p \times n$ rather than $n \times p$)
    - To reduce both the communication needed (?) and the amount of data stored on each device, gather/scatter is more suitable when $\mathbf{x}$ has fewer/more features than $\mathbf{y}$.
- Asynchronous layers
  - In the gather strategy, we first need to communicate all the features of $\mathbf{x}$ before we can compute the output. 
    - ![async_gather.png](async_gather.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_async.html)
  - In the scatter strategy, need to compute the output on all devices before we can communicate results and sum them. 
    - ![async_scatter.png](async_scatter.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_async.html)
  - Asynchronous layers allow us to overlap communication with computation and reduce downtime. 
  - Gather
    - ![gather.png](gather.png)[Source](https://arxiv.org/pdf/2302.05442)
  - Scatter
    - ![scatter.png](scatter.png)[Source](https://arxiv.org/pdf/2302.05442)
  - If we want all nodes to contain all activations, consider Ring Allreduce, which uses the scatter-reduce above, and then an allgather. 
    - This sums individual arrays on all nodes, and eventually every node will have a copy of this sum.

### 3D Parallelism

- We can combine all the 3 parallelism types for increased computational gains.
  - ![3d.png](3d.png)[Source](http://web.ecs.baylor.edu/faculty/dong/elc5396_DeepLearning/DeepLearningSignalProcessingH3.pdf)
  - [DeepSpeed](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

## Inference

- We can break down LLM inference into two stages: prefill and decoding.
  - Terminology
    - For pre-fill, we are concerned with the **time to first token (TFT)**, or **latency**.
      - Human visual reaction time is around **200ms**, so [Baseten recommends](https://www.baseten.co/blog/understanding-performance-benchmarks-for-llm-inference/) <200ms latency.
    - For decoding, we are concerned with the **time per output token**, or **tokens per second (TPS)**, or **throughput**.
      - Reading time averages between 3-5 words per second. 
      - For most LLMs, 4 tokens approximately equals 3 words. 
      - [Baseten recommends](https://www.baseten.co/blog/understanding-performance-benchmarks-for-llm-inference/) around 30 tokens per second. 
  - Quick math (based on [Cursor's article](https://www.cursor.com/blog/llama-inference#prompt-processing-is-really-cheap) of serving Llama-2-70B)
    - Compute
      - FLOPS per token $\approx$ 2*70B
        - We technically also have attention calculations, which scale linearly with sequence length $N$, but this is small for large models (70B) with relatively short sequences (8k). I have not checked what this means for longer context lengths. 
        - Compute therefore **scales linearly** with $N$
    - Memory
      - Storing model params $\approx$ 140GB
      - Storing kv cache $\approx 4BNn_gn_ld_{head} \approx 320BN$ KB. 
    - Memory bandwidth
      - Key: Every parameter is passed from HBM to SRAM around once (especially with flash attention), so this is usually **the same** as memory (per second)
      - All model params needs to be passed only once for prefill, but **once per token** for decoding
        - For prefill, bandwidth is **generally constant** wrt $N$.
      - If memory is mostly dominated by model params / kv cache, we would expect memory bandwidth to be mostly dominated by model params / kv cache.
  - Inference characteristic differences
    - Since model params needs to be passed only once for prefill, but one per token for decoding,
      - Pre-fill tends to be compute bound (for $N > 156$ on an A100, per [Cursor's calculations]((https://www.cursor.com/blog/llama-inference#prompt-processing-is-really-cheap))).
      - Decoding tends to be bandwidth bound.
      - Costs:
        - When using open-sourced models, we pay **per second**, and when using closed-sourced models, we pay **per token**.
        - Due to the different bounds for pre-fill/decoding, Cursor found it cheaper to use open-sourced models for prompt processing and closed-sourced models for completion-heavy tasks.
  - Levers
    - Increasing Batch Size
      - Increasing batch size will increase compute linearly, but bandwidth sublinearly (only KV cache part). This can increase throughput. 
      - Batch size is upper bounded by GPU memory, since KV cache memory grows linearly with batch size.
      - Increasing compute worsens latency
        - [Disaggregated serving](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) is helpful because of the different characteristics of prefill and decoding.
      - So far, we've been focusing on total TPS. Perceived TPS considers what an individual user sees. Increasing batch size, generally decreases perceived TPS, and we're lower bounded by non-functional requirements.
      - Increasing batch size is sometimes not feasible for startups with "bursty" request profiles. 
    - Number of GPUs
      - If we increase the number of GPUs, we can shard model weights, and therefore increase our batch size limit. 
      - It is important to note that parallelism is therefore not only done out of necessity, but **useful** in increasing throughput. 
      - Additional parallelism also incurs communication cost, however.
- Tensor-RT
  - TensorRT works by taking a model description, such as an ONNX file, and compiling the model to run more efficiently on a given GPU (optimized runtime engines).
  - As opposed to the more general `torch.compile` mentioned above, it is optimized specifically for NVIDIA hardware. 
    - `torch.compile` does allow us to specify the Tensor-RT backend.
- vLLM
  - Tailored for efficient LLM inference, while Tensor-RT supports a broader range of model types.
  - Designed to be more flexible in terms of hardware, while Tensor-RT is optimized specifically for NVIDIA GPUs.
- LoRA Swapping
  - We usually build a single engine that works for the foundation model and swap LoRAs in and out as needed at inference time.
  - LoRAs can be stored on 
    - GPU memory: the LoRA is actively being used on the GPU to fulfill an inference request. 
      - Capacity: 10s/100s of LoRAs 
      - Load time: Instant 
    - Host/CPU memory: the LoRA is cached on system memory within the model serving instance. 
      - Capacity: 1000s of LoRAs 
      - Load time: 1-2 ms 
    - Disk: the LoRA is cached on container storage attached to the model serving instance. 
      - Capacity: Effectively unlimited 
      - Load time: 10-100ms 
    - Network: the LoRA is living in an HF repo, S3 bucket, etc. 
      - Capacity: Effectively unlimited 
      - Load time: ~100ms
- Batching
  - No batching: each request is processed one at a time.
  - Static batching: requests are placed in batches that are run when full.
  - Dynamic batching: requests are placed in batches as they’re received and batches run once full or once enough time has elapsed since the first request.
    - Dynamic batching is great for live traffic on models like Stable Diffusion XL, where each inference request takes about the same amount of time. 
    - For LLMs, however, output sequences will vary in length. If you use a dynamic batching approach, each batch of requests is going to need to wait for the longest output to finish before the next batch can begin.
  - Continuous (in-flight) batching: requests are processed token-by-token, with new requests getting processed as older requests finish and free up space on the GPU.
    - Model servers like TGI and VLLM offer continuous batching, while TensorRT-LLM uses “in-flight batching” to essentially the same effect.
    - This, however, increases TFT
      - Since the prefill phase takes compute and has a different computational pattern than generation, it cannot be easily batched with the generation of tokens
      - Continuous batching frameworks currently manage this via hyperparameter: waiting_served_ratio, or the ratio of requests waiting for prefill to those waiting end-of-sequence tokens.
- Speculative decoding 
  - The process of coordinating a large LLM (the target model) and a smaller LLM (the draft model) on the same GPU to combine the quality of the large model with the speed of the small model. Some ways of creating draft models are: 
    - Using a single model for both draft and target, and training the model from the start.
    - Letting the draft model use part of the target model, and training the draft model.
    - Distilling the knowledge from the target model into the draft model.
  - The idea is to additionally have the large LLM validate the drafts - if it accepts the drafts then throughput is increased.
    - The idea hinges on the fact that decoding tends to be memory bound. 
    - Hence, we can parallelize $f(x_1)$ and $f(\hat{x}_2) = f(f^*(x_1))$. If $f(x_1) \approx x_2$, we can output 2 tokens, and if not we simply output 1. 
- Chunked prefill
  - Continuous batching can introduce latency as the decode phases are delayed until the prefill requests are completed.
  - ![chunked_prefill.png](chunked_prefill.png)[Source](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)
  - Chunked prefill prevents the prefill phase from becoming a bottleneck, enables more parallelization with decode phase tokens, and increases GPU utilization.
  - Using prefill chunks also decouples memory consumption from the context length of incoming requests
- With sliding window attention, we can chunk and parallelize the prefill process.