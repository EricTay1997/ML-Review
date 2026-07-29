# Inference

LLM serving: batching and serving-system internals. Prefill/decode characteristics — the memory/bandwidth/compute cost model and batch-size trade-offs — live in [Basics §Training vs Inference](basics.md#training-vs-inference). See also [GPUs](gpus.md) for the hardware model and [Parallelism](parallelism.md) for sharding.

## Serving Engines

- Tensor-RT
  - TensorRT works by taking a model description, such as an ONNX file, and compiling the model to run more efficiently on a given GPU (optimized runtime engines).
  - As opposed to the more general `torch.compile`, it is optimized specifically for NVIDIA hardware. 
    - `torch.compile` does allow us to specify the Tensor-RT backend.
  - See `bloom_tensorrt_llm.ipynb` in this folder for a hands-on TensorRT-LLM walkthrough.
- vLLM
  - Tailored for efficient LLM inference, while Tensor-RT supports a broader range of model types.
  - Designed to be more flexible in terms of hardware, while Tensor-RT is optimized specifically for NVIDIA GPUs.
  - See [vLLM Internals](#vllm-internals) below.

## vLLM Internals

[Aleksa Gordić's vLLM deep-dive](https://www.aleksagordic.com/blog/vllm)

### Continuous Batching

- Put all requests into one sequence and process all at once

### Paged attention

- KV caches in paged memory
- Ease of retrieval - think continuous batching! 

### Chunked prefill 

- Chunks prefill and computes KV cache for each (so long sequences don't slow down everyone)

### Prefix Caching

- Hashes each prefill chunk

### Guided decoding 

- Masking logics

### Speculative Decoding

- Small model drafts k tokens
- Run forward pass for large model over prompt tokens + k tokens and do accept/reject
- n-gram, EAGLE, Medusa
- The idea is to additionally have the large LLM validate the drafts - if it accepts the drafts then throughput is increased.
  - The idea hinges on the fact that decoding tends to be memory bound. 
  - Hence, we can parallelize $`f(x_1)`$ and $`f(\hat{x}_2) = f(f^*(x_1))`$. If $`f(x_1) \approx x_2`$, we can output 2 tokens, and if not we simply output 1. 

### Disaggregated P/D
  - Prefill workers write KV to a dedicated KV-cache service; decode workers read from it. This isolates long, bursty prefill from steady, latency-sensitive decode.

### UniProcExecutor to MultiProcExecutor
  - TP within a node, PP across nodes, then DP to scale out — see [Parallelism](parallelism.md#placement-tp-within-a-node-pp-across-nodes).