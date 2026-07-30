## Notation

- $`b`$ — batch size; $`s`$ — sequence length; $`h`$ — hidden dimension (d_model); $`a`$ — number of attention heads ($`d_{head} = h/a`$); $`n_{kv}`$ — number of KV heads (< $`a`$ under GQA); $`L`$ — number of layers
- $`N`$ — total parameter count; $`D`$ — number of training tokens
- Bytes: bf16/fp16 = 2, fp32 = 4. Any stored activation tensor of shape $`(b, s, h)`$ costs $`2sbh`$ bytes in bf16 — most per-layer memory terms below are multiples of this

## Basics

- Where things live during training (GPUs)
  - HBM (GPU) — model parameters, optimizer states, activations, gradients
  - CPU (RAM) — dataset / dataloader
  - Disk — full dataset, checkpoints
- What are we bounded by?
  - [He's article](https://horace.io/brrr_intro.html)
  - Memory
    - Size of DRAM
      - Solution: See below
  - (HBM) Bandwidth
    - Time spent transferring tensors within a GPU (from GPU memory - HBM, to compute cores)
      - Solution: Operator fusion
  - Inter-chip Communication
    - PCIe, NVLink / NVSwitch (ICI), InfiniBand / RoCE (DCN)
  - Compute (on SRAM)
    - Time spent on your GPU computing actual floating point operations (FLOPS)
      - Solution: More tensor cores
  - Overhead
    - Everything else
      - Solution: Asynchronous computation
- Which bound am I hitting? **Arithmetic intensity** (source: [scaling book — roofline](https://jax-ml.github.io/scaling-book/roofline/))
  - $`\text{Arithmetic Intensity} = \dfrac{\text{Computation FLOPs}}{\text{Communication Bytes}}`$
  - Compare against the accelerator's **ridge point** = peak FLOPs/s ÷ bandwidth. Above it → compute-bound; below → bandwidth-bound
  - Worked example — matmul $`X[B, D] \times Y[D, F] \to Z[B, F]`$, all bf16 (local notation: $`B`$ is the per-replica batch in _tokens_, $`D, F`$ are model dims)
    - Load $`2BD + 2DF`$ bytes, perform $`2BDF`$ FLOPs, write $`2BF`$ bytes back:
    - $`\text{Intensity(matmul)} = \dfrac{2BDF}{2BD + 2DF + 2BF} = \dfrac{BDF}{BD + DF + BF}`$
    - Assuming $`B \ll D, F`$: $`\dfrac{BDF}{BD + DF + BF} \approx \dfrac{BDF}{DF} = B`$ — intensity _is_ the token batch size
    - So compute-bound requires $`B > \text{Intensity(accelerator)} = \dfrac{1.97 \times 10^{14}}{8.20 \times 10^{11}} = 240`$ tokens
      - Those numbers are a **TPU v5e MXU** (197 bf16 TFLOP/s, 820 GB/s HBM); for an H100 the ridge is ~295 FLOPs/byte
    - Reasonable for transformer matmuls: per-replica $`B < 1024`$ tokens (_not_ sequences) while $`D, F > 8000`$
  - Caveats
    - The $`B \ll D, F`$ assumption fails when $`D, F`$ are small — the $`BD`$ / $`BF`$ terms stop being negligible, intensity falls below $`B`$, and the critical batch size needed to become compute-bound rises
    - Dtype-specific: int8 halves the bytes (intensity $`\approx 2B`$) but also raises peak FLOPs/s (v5e: 394 TOP/s vs 197 TFLOP/s), so the ridge moves too — on v5e the two cancel to the same ~240-token threshold, at 2× the throughput

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

- High Level
  - As $`b`$ increases, compute scales linearly but bandwidth scales sublinearly, because weight reads are amortized across the batch
  - We want to increase batch size until we're compute-bound, because throughput plateaus at this ridge
  - This $`B_{sat}`$ is much lower for training because the model weight read is amortized over all training tokens, whilst for inference this needs to happen for every decode token.
- Training
  - Bigger batch → better GPU utilization
    - Saturates the GPU: more parallel work to fill all compute units
    - Amortizes fixed costs: kernel launch overhead, memory transfers, optimizer step
    - Better memory coalescing: larger contiguous memory accesses are more efficient on GPUs
  - Batch just needs to be **big enough**: kernels saturate early (at $`s{=}4k`$ even $`b{=}1`$ streams thousands of tokens per weight read) and per-step fixed costs (optimizer step, DP all-reduce) amortize past a few thousand tokens/step — beyond that gains are marginal, and the _global_ batch is set by learning dynamics (critical batch size, with LR scaled to match; see [Optimization](../optimization/notes.md))
- Inference
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

Summary (per layer, one sequence; all in $`s`$ — see the GDN caveat for the $`s \to d`$ trade). Prefill tracks the training-compute column:

**Training**

| | Compute | Bandwidth | Act. memory |
|---|---|---|---|
| Naive | $`O(s^2)`$ | $`O(s^2)`$ | $`O(s^2)`$ |
| Flash | $`O(s^2)`$ | $`O(s^2d^2/M)`$ | $`O(s)`$ |
| SWA | $`O(sw)`$ | $`O(sw)`$ | $`O(sw)`$ |
| Linear/GDN | $`O(s)`$ | $`O(s)`$ | $`O(s)`$ |

**Decode (per token)**

| | Compute | Bandwidth | Cache/state | Recall |
|---|---|---|---|---|
| Naive | $`O(s)`$ | $`O(s)`$ | $`O(s)`$ | exact |
| Flash | $`O(s)`$ | $`O(s)`$ | $`O(s)`$ | exact |
| SWA | $`O(w)`$ | $`O(w)`$ | $`O(w)`$ | window only |
| Linear/GDN | $`O(1)`$ | $`O(1)`$ | $`O(1)`$ | lossy |

- Flash is the only row that improves _nothing_ in the compute columns — it's a schedule change, so its win is entirely in bandwidth and training memory, and decode is untouched
- SWA and GDN improve every column, because they change the algorithm; their difference is the decode floor ($`O(w)`$ truncated-exact vs $`O(1)`$ compressed-lossy)

### Naive attention

- Training
  - Compute: $`O(s^2)`$: forward $`4s^2h`$/layer — $`2s^2h`$ for the scores $`QK^\top`$ + $`2s^2h`$ for the values $`PV`$ (per head $`4s^2d`$, summed over $`a`$ heads with $`h = a \cdot d_{head}`$)
  - fwd+bwd = $`12s^2h`$ — **6 matmuls** of $`2s^2d`$ per head: 2 forward + 4 backward, since each forward matmul needs one gradient per operand (both operands are activations here, attention having no weights). From $`O = PV`$: $`dV = P^\top dO`$, $`dP = dO V^\top`$; from $`S = QK^\top`$: $`dQ = dS K`$, $`dK = dS^\top Q`$. Same 3× rule as $`6ND`$
    - The softmax and its backward are elementwise $`O(s^2)`$ non-matmul work, which runs on the slow vector units rather than tensor cores
  - Bandwidth: $`O(s^2)`$: the $`s \times s`$ matrix is _materialized to HBM_ and round-trips several times around the softmax (write $`S`$ → read → write $`P`$ → read) — so it moves $`O(s^2)`$ bytes to do $`O(s^2 d)`$ FLOPs, an intensity of only ~$`d`$ vs the ~240–295 ridge
    - → long-context attention is **bandwidth-bound even in training**, while the parameter matmuls around it are comfortably compute-bound
  - Memory: $`O(s^2)`$: the $`5as^2b`$ activation term is the dominant term at long context

### Flash attention

[Dao et al., arXiv 2205.14135](https://arxiv.org/abs/2205.14135) — same math, different schedule: tile Q, K, V through SRAM with an online (running-max) softmax so the $`s \times s`$ matrix never touches HBM. **Trades compute for bandwidth** — the winning trade, because attention was bandwidth-bound.

- Primer
  - By materializing the $`s \times s`$ $`QK^\top`$ matrix we incur $`O(s^2)`$ **bandwidth** (it round-trips HBM around the softmax) and $`O(s^2)`$ **memory** (it's kept for the backward pass)
  - Flash attention avoids that materialization, hence $`O(s)`$ memory. The matrix is only ever needed to produce the probability scalers that weight each token's linear combination of $`V`$ vectors — and **that linear combination can be accumulated without ever holding the matrix** 
  - Note: if SRAM were big enough to hold Q, K, V, bandwidth would be $`O(sd)`$ — read each once and done. The problem is that it isn't
  - Mechanism: hold Q, then stream K, V past it in blocks. Per token, carry three things — the running max $`m`$, the running sum $`\ell`$, and a running linear combination of $`V`$. Each new block rescales that accumulated combination (by $`\exp(m_{old} - m_{new})`$ and adds a new component. After the last block, normalising by $`\ell`$ gives exactly the intended output
  - So why $`O(s^2d^2/M)`$ bandwidth rather than $`O(sd)`$? Because SRAM can't hold all of Q either. Q must be tiled, and **each Q tile re-streams all of K and V** — so bandwidth = (one pass over K, V) × (number of Q tiles) = $`O(sd) \times O(sd/M) = O(s^2d^2/M)`$
    - $`sd/M`$ is just "how many SRAM-fuls of Q there are"
- Training
  - Compute: $`O(s^2) \to O(s^2)`$ (unchanged; **~+17%** constant): the backward recomputes $`S = QK^\top`$ from tiles instead of reading it, a 7th matmul → $`14s^2h`$ vs $`12s^2h`$. Under causal masking it also skips fully-masked tiles (~2× saving), which can more than pay that back
  - Bandwidth: $`O(s^2) \to O(s^2d^2/M)`$ ($`M`$ = SRAM capacity, so ~$`M/d^2`$ less traffic — ~6–24× on an A100): the $`s^2`$ round-trips vanish; K, V are re-read once per Q block instead — intensity climbs above the ridge and attention becomes **compute-bound**
  - Memory: $`O(s^2) \to O(s)`$: kills the $`5as^2b`$ term → activations linear in $`s`$
- Inference (nothing new here)
  - Compute: prefill $`O(s^2)`$, decode $`O(s)`$/token: prefill like training-forward ($`4s^2h`$/layer); decode $`4sh`$/token/layer (one query vs $`s`$ cached keys) — the $`s/8h`$ share above
  - Bandwidth: $`O(s)`$/token: decode reads the whole KV cache every token — intensity ~1 (each cached K element feeds one multiply-add for the score, each V element one for the sum), lifted only to $`a/n_{kv}`$ by GQA (~8 for 70B) — still far under the ridge
  - Memory: $`O(s)`$: KV cache grows $`4 \cdot n_{kv} \cdot d_{head}`$ bytes/token/layer (the 320 KiB/token above)

### Sliding-window attention (SWA)

Each query sees only the last $`w`$ tokens (Mistral, Gemma 3, gpt-oss).

- Training
  - Compute: $`O(s^2) \to O(sw)`$: $`4s^2h \to 4swh`$ forward ($`12swh`$ fwd+bwd)
  - Bandwidth: $`O(s^2) \to O(sw)`$
  - Memory: $`O(s^2) \to O(sw)`$: $`5as^2b \to 5aswb`$
- Inference
  - Compute: prefill $`O(s^2) \to O(sw)`$, decode $`O(s) \to O(w)`$/token: $`4swh`$/layer prefill, $`4wh`$/token/layer decode — constant in context
    - Prefill is where the win is largest for long prompts: at $`s{=}32k, w{=}4k`$ that's 8× fewer attention FLOPs, dropping attention from ~50% of prefill FLOPs ($`s/8h`$) to ~6%
  - Bandwidth: prefill $`O(s^2) \to O(sw)`$, decode $`O(s) \to O(w)`$: reads only $`w`$ tokens of KV per step
  - Memory: $`O(s) \to O(w)`$: KV cache capped at $`w`$ tokens/layer
- Cost: no direct access past $`w`$ (receptive field grows ~$`wL`$ across depth, but recall degrades); shipped interleaved with full-attention layers (Gemma 3 at 5:1, gpt-oss alternating)

### Linear attention / gated DeltaNet

Qwen3-Next; analyzing just the SSM part — replace softmax-over-history with a **fixed-size recurrent state** $`S`$ ($`d_k \times d_v`$ per head): each step decays $`S`$, writes $`k_t v_t^\top`$, reads $`o_t = S_t^\top q_t`$ (delta rule + gating refine the write/decay; see [Attention Variants](../architecture/attention.md)).

- Training (over $`s`$ tokens)
  - Compute: $`O(s^2) \to O(s)`$: ~$`6ad^2`$ FLOPs/token (state decay + outer-product write + read), small next to QKVO; chunked-scan formulations keep it matmul-shaped for tensor cores
  - Bandwidth: $`O(s^2) \to O(s)`$: no $`s^2`$ traffic; within a chunk the state stays in SRAM/registers
  - Memory: $`O(s^2) \to O(s)`$: activations are the per-chunk states
- Inference (per token)
  - Compute: $`O(s) \to O(1)`$: ~$`6ad^2`$, independent of context
  - Bandwidth: $`O(s) \to O(1)`$: read/write the fixed state per step
  - Memory: $`O(s) \to O(1)`$: **no KV cache** — the state is ~2 MiB/layer ($`a{=}64, d{=}128`$), equivalent to a KV cache of only ~512 tokens/layer; past that it's pure win
- Caveat — **the real trade**: $`s`$ for $`d`$. The complexities above are all in $`s`$, but per token per layer attention compute costs $`4sh`$ while GDN costs about $`6hd`$ — the sequence-length dependence becomes a state-dimension dependence, so GDN is only cheaper once $`s >> d`$
  - Per head that reads as $`4sd \to 6d^2`$ — quadratic in $`d`$ because the state is a _matrix_ (a key→value linear map), and touching a $`d \times d`$ matrix costs $`d^2`$, whereas attention only manipulates vectors ($`O(d)`$ each) but must touch $`s`$ of them
- Cost: the state is a _lossy compression_ of the entire history — exact long-range recall (needle-style retrieval) degrades; hence hybrids (Qwen3-Next interleaves 3 GDN : 1 full-attention layer)

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