## Notation

- $`b`$ — batch size; $`s`$ — sequence length; $`h`$ — hidden dimension (d_model); $`a`$ — number of attention heads ($`d_{head} = h/a`$); $`n_{kv}`$ — number of KV heads (< $`a`$ under GQA); $`L`$ — number of layers
- $`N`$ — total parameter count; $`D`$ — number of training tokens
- Bytes: bf16/fp16 = 2, fp32 = 4. Any stored activation tensor of shape $`(b, s, h)`$ costs $`2sbh`$ bytes in bf16 — most per-layer memory terms below are multiples of this

## GPUs (more in [GPUS](./gpus.md))

- Where things live during training
  - HBM (GPU) — model parameters, optimizer states, activations, gradients
  - CPU (RAM) — dataset / dataloader
  - Disk — full dataset, checkpoints
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

## Training vs Inference

- Training
  - **Compute-bound**: large batches give every weight read (bandwidth) a lot of FLOPs to amortize over
  - Memory: params + gradients + optimizer states + activations
    - Mixed-precision Adam ≈ **16 bytes/param**: bf16 params (2) + bf16 grads (2) + fp32 master weights (4) + fp32 Adam $`m, v`$ (4+4) → a 7B model is ~112 GB before activations
    - Activations (every intermediate tensor the backward pass needs): per layer ≈ $`sbh(34 + 5as/h)`$ bytes ([Korthikanti et al., arXiv 2205.05198](https://arxiv.org/abs/2205.05198)); the $`5as^2b`$ term is the materialized $`s \times s`$ attention matrices, which flash attention (later) eliminates — leaving $`34sbh`$/layer ≈ 18 GB at $`b{=}1, s{=}4096`$ for 7B (the $`s^2`$ term would have added ~34 GB) — this is what flash attention and gradient checkpointing attack
      - Unlike the 16 bytes/param (fixed), activations scale with **batch size** (and sequence length) — at large $`b`$ they dominate training memory, which is why batch size is the memory knob (see [Batch Size](#batch-size))
  - Bandwidth: per step ~O(params) traffic several times over (forward read, backward read, optimizer read/write of all 16 bytes/param) plus activation traffic $`O(b \cdot s \cdot h)`$ (flash attention exists precisely to fix the one bandwidth-bound exception: the $`s^2`$ attention traffic)
  - Compute: ≈ $`6ND(1 + s/8h) \approx 6ND`$ FLOPs total ($`N`$ params, $`D`$ tokens): $`2N`$ per token forward + $`4N`$ backward — e.g. Llama 3 70B × 15T tokens ≈ $`6.3 \times 10^{24}`$ FLOPs
    - Derivation ([scaling book](https://jax-ml.github.io/scaling-book/transformers/)): a dot product of length $`P`$ is $`P`$ multiplies + $`P`$ adds = $`2P`$ FLOPs, so a matmul $`[T, P] \times [P, M]`$ costs $`2TPM`$ — **2 FLOPs per parameter per token** in the forward pass, since each token "touches" each weight exactly once in a multiply-add
    - The backward pass does two matmuls of the same shape per weight matrix — $`\partial L/\partial W = x^\top (\partial L/\partial y)`$ (weight grads) and $`\partial L/\partial x = (\partial L/\partial y) W^\top`$ (activation grads, to keep backpropagating) — so backward = $`2\times`$ forward = $`4N`$/token, total $`6N`$/token
    - The $`(1 + s/8h)`$ factor is the attention scores/values FLOPs, which the parameter count misses: ≈ $`4sh`$/token/layer (2 for $`qK^\top`$, 2 for the value sum). Assuming a gated (SwiGLU-style) MLP with $`F{=}4h`$, per-layer params = $`4h^2`$ (QKVO) + $`12h^2`$ (3 MLP matrices) = $`16h^2`$, so the attention share = $`4sh / (2 \cdot 16h^2) = s/8h`$ — ~12.5% at $`s{=}8k, h{=}8k`$, dominant past $`s > 8h`$ (the constant shifts with MLP width/gating; causal-aware kernels halve the attention side)
    - Where it goes: MLP dominates (~¾ of per-layer params/FLOPs under the $`F{=}4h`$ gated MLP; attention projections ~¼) — and the scores/values share above grows linearly with context
- Inference
  - **Bandwidth-bound** (decode): generating one token requires reading _all_ weights + that sequence's KV cache once, while doing only ~$`2N`$ FLOPs with them (arithmetic intensity ~1 vs an H100 ridge of ~295 FLOPs/byte)
  - Memory: params + **KV cache** (activations are negligible — only one token's worth at a time in decode)
    - KV cache per token: $`2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot 2`$ bytes → 320 KiB/token for 70B (the [harnesses note](../agents/harnesses.md) does this arithmetic) → 40 GiB at 128k context: the cache rivals the weights at long context or large batch
  - Bandwidth: sublinear in batch size — the weights read is shared across the whole batch (sublinear per-token cost) but each sequence's KV cache read is its own (linear)
  - Compute: **prefill** is training-forward-like — all prompt tokens in parallel, big matmuls, compute-bound ($`2N(1 + s/8h) \approx 2N`$ FLOPs/token); **decode** does the same FLOPs per token but one token at a time, leaving compute idle waiting on memory (hence [disaggregated serving](https://docs.vllm.ai/en/latest/features/disagg_prefill.html))
    - What the KV cache buys: without it, generating token $`t`$ means re-running the forward pass over all $`t`$ prior tokens — $`O(t)`$ recompute per token, $`O(T^2)`$ per sequence. The cache makes the _parameter_ FLOPs flat (~$`2N`$/token); the attention term ($`2N \cdot s/8h`$, linear in context) necessarily remains — but in decode it arrives welded to the KV-cache _read_ (~2 FLOPs per byte moved), so it's bandwidth-bound and context growth is felt through the memory/bandwidth story above

## Batch Size

How memory/bandwidth/compute scale with $`b`$ is covered in [Training vs Inference](#training-vs-inference); this section is about _choosing_ $`b`$.

- Training
  - Bigger batch → better GPU utilization
    - Saturates the GPU: more parallel work to fill all compute units
    - Amortizes fixed costs: kernel launch overhead, memory transfers, optimizer step
    - Better memory coalescing: larger contiguous memory accesses are more efficient on GPUs
  - Batch just needs to be **big enough**: kernels saturate early (at $`s{=}4k`$ even $`b{=}1`$ streams thousands of tokens per weight read) and per-step fixed costs (optimizer step, DP all-reduce) amortize past a few thousand tokens/step — beyond that gains are marginal, and the _global_ batch is set by learning dynamics (critical batch size, with LR scaled to match; see [Optimization](../optimization/notes.md))
- Inference
  - As $`b`$ increases, compute scales linearly but bandwidth scales sublinearly, because weight reads are amortized across the batch
  - Bigger batch → higher **total** throughput: decode moves up the roofline toward compute-bound
    - Below a saturation batch $`B_{sat}`$, step time is dominated by streaming the weights layer-by-layer from HBM, so it's nearly flat — computing 1 vs 10 tokens can take a similar time; beyond $`B_{sat}`$ the kernels become compute-bound and step time grows roughly with $`b`$ ([Gordić's vLLM deep-dive](https://www.aleksagordic.com/blog/vllm))
    - Throughput = $`b / T(b)`$, so: below $`B_{sat}`$ ($`T`$ ≈ constant) throughput grows ~linearly with $`b`$; above it ($`T \propto b`$) throughput **plateaus** at the compute roofline — zero marginal throughput, linearly worse latency. $`B_{sat}`$ is the operating point, not just a landmark
    - Caveat 1 — KV reads never amortize (per-sequence): bandwidth cost = weights (constant) + KV (linear in $`b`$). At long context the KV term dominates ($`b \cdot s \cdot 320`$ KiB rivals the 140 GB weight read at $`b \cdot s \approx`$ 440k tokens, e.g. $`b{=}14`$ at $`s{=}32k`$) → throughput saturates _before_ the compute knee, at a lower plateau (GQA/MLA raise it by shrinking KV bytes)
    - Caveat 2 — memory often binds first: $`B_{sat}`$ is a few hundred sequences, and at real context lengths that many KV caches may not fit in HBM — many deployments never reach the knee (why paged attention / prefix caching are _throughput_ features, not just memory features)
    - Same saturation in token units: prefill goes compute-bound past only ~156 tokens on an A100 ([Cursor's Llama-2-70B math](https://www.cursor.com/blog/llama-inference#prompt-processing-is-really-cheap)) — sequence length saturates prefill "for free," which is why the batching question is really a _decode_ question
  - But worse **per-user** latency: perceived TPS (what one user sees) drops as batch grows — batch size is the throughput↔latency knob, lower-bounded by latency non-functional requirements
    - Why: a user's next token arrives once per decode step, and step time = bytes moved / bandwidth — every added sequence adds its KV read to _every_ step (mild creep below $`B_{sat}`$, then step time $`\propto b`$ above it)
  - Upper bounds: GPU memory (each sequence adds its own KV cache), and the request profile — a "bursty" low-traffic service may simply not have concurrent requests to batch

## Attention

The parameter matmuls are covered above; this is the scores/values part — the only cost that _grows with context_. Mechanism details live in [Attention Variants](../architecture/attention.md); this section is just the cost model.

- **Naive attention**
  - Training: compute $`4s^2h`$/layer (the $`s/8h`$ share above); the $`s \times s`$ matrix is _materialized to HBM_ → the $`5as^2b`$ activation-memory term, and the same matrix is written/re-read around the softmax — few FLOPs per byte, so long-context attention goes **bandwidth-bound** even in training
  - Inference: KV cache grows $`4 \cdot n_{kv} \cdot d_{head}`$ bytes/token/layer (the 320 KiB/token above); decode reads the whole cache every token — the $`O(s)`$ bandwidth term
- **Flash attention** ([Dao et al., arXiv 2205.14135](https://arxiv.org/abs/2205.14135)) — same math, different schedule
  - Tiles Q, K, V through SRAM with an online (running-max) softmax; the $`s \times s`$ matrix never touches HBM
  - Training: kills the $`5as^2b`$ memory term (→ activations linear in $`s`$) and the matrix traffic (→ compute-bound again); FLOPs _unchanged_ — still $`O(s^2)`$, slightly more in backward (tiles are recomputed rather than stored)
  - Inference: same win for prefill; decode unchanged — the KV read is irreducible (flash-decoding parallelizes it, doesn't shrink it)
- **Sliding-window attention (SWA)** — each query sees only the last $`w`$ tokens (Mistral, Gemma 3, gpt-oss)
  - Training: compute $`4s^2h \to 4swh`$ — linear in $`s`$
  - Inference: KV cache capped at $`w`$ tokens — memory, per-token reads, and per-token FLOPs all **constant** in context
  - Cost: no direct access past $`w`$ (receptive field grows ~$`wL`$ across depth, but recall degrades); shipped interleaved with full-attention layers (Gemma 3 at 5:1, gpt-oss alternating)
- **Linear attention / gated DeltaNet** (Qwen3-Next; analyzing just the SSM part) — replace softmax-over-history with a **fixed-size recurrent state** $`S`$ ($`d_k \times d_v`$ per head): each step decays $`S`$, writes $`k_t v_t^\top`$, reads $`o_t = S_t^\top q_t`$ (delta rule + gating refine the write/decay; see [Attention Variants](../architecture/attention.md))
  - Training: no $`s^2`$ anywhere — the attention part is linear in $`s`$ (~$`6ad^2`$ FLOPs/token, small next to QKVO); chunked-scan formulations keep it matmul-shaped for tensor cores
  - Inference: **no KV cache** — the state is ~2 MiB/layer ($`a{=}64, d{=}128`$), equivalent to a KV cache of only ~512 tokens/layer; beyond that it's pure win, and per-token reads/FLOPs are constant in context
  - Cost: the state is a _lossy compression_ of the entire history — exact long-range recall (needle-style retrieval) degrades; hence hybrids (Qwen3-Next interleaves 3 GDN : 1 full-attention layer)
- Summary (per layer, one sequence):

| | Train compute | Train act. memory | Decode cache/state | Decode reads+FLOPs per token | Recall |
|---|---|---|---|---|---|
| Naive | $`O(s^2)`$ | $`O(s^2)`$ | $`O(s)`$ | $`O(s)`$ | exact |
| Flash | $`O(s^2)`$ | $`O(s)`$ | $`O(s)`$ | $`O(s)`$ | exact |
| SWA | $`O(sw)`$ | $`O(s)`$ | $`O(w)`$ | $`O(w)`$ | window only |
| Linear/GDN | $`O(s)`$ | $`O(s)`$ | $`O(1)`$ | $`O(1)`$ | lossy |

## Optimization Methods

- Parallelism - see [Parallelism](./parallelism.md)
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
      - QLoRA combines double quantization with [LoRA](../post_training/notes.md).
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