# Parallelism

Primary source: the JAX scaling book — [sharding](https://jax-ml.github.io/scaling-book/sharding/), [training](https://jax-ml.github.io/scaling-book/training/), [applied training](https://jax-ml.github.io/scaling-book/applied-training/), [GPUs](https://jax-ml.github.io/scaling-book/gpus/). Also some from [Lippe's notes](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/overview.html). Inference is a different problem (no backward pass, decode is bandwidth-bound) and has its own sharding rules in [Inference §Sharding for inference](inference.md#sharding-for-inference). See also [Basics](basics.md), [TPUs & Rooflines](tpus.md), [GPUs](gpus.md).

## Sharding and collectives

_From [sharding](https://jax-ml.github.io/scaling-book/sharding/). This is the primitives layer — everything in §Strategies is an application of it._

### Notation

- **Mesh**: the device mesh `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))` tells us we have 4 TPUs in a 2×2 grid, with axis names $`X`$ and $`Y`$
- **Sharding**: $`A[I_X, J_Y]`$ tells us to shard the first axis $`I`$ along the mesh axis $`X`$, and the second axis $`J`$ along the mesh axis $`Y`$. This sharding tells us that each shard holds $`1/(|X| \cdot |Y|)`$ of the array
- <img src="images/sharding_data_devices.png" width="420">[Source](https://jax-ml.github.io/scaling-book/sharding/)
- Global shape (before sharding) vs local shape (after sharding, i.e. what each device actually holds):
  - <img src="images/sharding_global_vs_local.png" width="520">[Source](https://jax-ml.github.io/scaling-book/sharding/)

### Which sharding needs which collective

- **Case 1**: neither input is sharded along the contracting dimension. We can multiply local shards without any communication
  - $`A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]`$
- **Case 2**: one input has a sharded contracting dimension. We typically "AllGather" the sharded input along the contracting dimension
  - $`A[I, J_X] \cdot B[J, K] \to C[I, K]`$
  - $`\textbf{AllGather}_X[I, J_X] \to A[I, J]`$, then $`A[I, J] \cdot B[J, K] \to C[I, K]`$
- **Case 3**: both inputs are sharded along the contracting dimension. We can multiply the local shards, then "AllReduce" the result
  - $`A[I, J_X] \cdot_{\text{LOCAL}} B[J_X, K] \to C[I, K]\{U_X\}`$ — each device along the $`X`$ dimension will be left with different partial sums of this final desired product
  - $`\textbf{AllReduce}_X C[I, K]\{U_X\} \to C[I, K]`$
- **Case 4**: both inputs have a non-contracting dimension sharded along the same axis. We cannot proceed without AllGathering one of the two inputs first
  - $`A[I_X, J] \cdot B[J, K_X] \to C[I_X, K_X]`$ is not allowed — we need to AllGather one term first

### The collectives and their costs

- **AllGather** copies the shards spread across devices onto EACH device along that axis. In our notation, it removes the sharding along an axis (drops a subscript)
  - Cost: <div align="center">
    $`\displaystyle T_{total} = \max\left[\dfrac{T_{min} \cdot \sum_i |X_i|}{2}, \dfrac{V}{W_{ici} \cdot N_{axes}}\right]`$ </div>
  - The bandwidth term (second) doesn't depend on the number of shards $`|X|`$ — the more shards, the more hops, but the fewer bytes that need to move per hop
  - Or more simply, just $`T_{total} = \dfrac{V}{W_{ici}}`$
- **ReduceScatter** sums an unreduced/partially summed array, such that each device now has a shard of the fully summed array
- Generally, an **AllReduce** is twice as expensive as an AllGather. One way to see this is to note that an AllReduce can be expressed as a composition of two other primitives: a ReduceScatter and an AllGather
  - $`\textbf{ReduceScatter}_{Y,J} : A[I_X, J]\{U_Y\} \to A[I_X, J_Y]`$
  - $`\textbf{AllGather}_Y : A[I_X, J_Y] \to A[I_X, J]`$
- **AllToAll**: whereas the AllGather not only moves shards but ensures all devices have the full copy, the AllToAll ends with each device still holding a shard — so there is less work (no copy)
  - $`\textbf{AllToAll}_{X,J} A[I_X, J] \to A[I, J_X]`$
  - General cost: $`T_{\text{comms per AllToAll}} = \dfrac{V \cdot \max(A, B, C, \dots)}{4 \cdot N \cdot W_{ici}}`$. For a 1D mesh this reduces to $`V / (4 \cdot W_{ici})`$ — $`\frac14`$ the cost of an AllGather
  - **Why the factor of 4, from not needing a copy**: AllGather must *replicate* — every byte needs to reach all $`N`$ devices, which costs a path of length $`N`$. AllToAll only needs to *relocate* — each byte has exactly one destination, so it takes the shortest path on the bidirectional ring: at most $`N/2`$ (left or right), and the **mean** element travels half that, $`N/4`$.
- **On GPUs** the same four collectives (NCCL, "nickel", and NVSHMEM) run on a switched tree instead of a ring, with _two_ bandwidths — $`W_{GPU}`$ (NVLink egress per GPU, inside a node) and $`W_{node}`$ (InfiniBand egress per node) — see [GPUs §Networking](gpus.md#networking). Three things change:
  - **AllGather / ReduceScatter within a node: nothing.** $`T = \text{bytes} \cdot (N-1)/(N \cdot W_{GPU}) \to \text{bytes}/W_{GPU}`$ — the ring cost with $`W_{ici} \to W_{GPU}`$; the switch gives every GPU its full egress bandwidth to whoever it's sending to. AllReduce = RS + AG = 2× as usual
  - **AllToAll within a node is a direct send** (full all-to-all connectivity): each GPU holds $`B/N`$ bytes and sends $`B/N^2`$ to each of $`N-1`$ peers: <div align="center">
    $`\displaystyle T_{AllToAll} = \frac{B \cdot (N-1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}`$ </div>
    vs $`B/4W`$ on the ring → at $`N = 8`$ a 2× theoretical speedup ($`B/8W`$ vs $`B/4W`$). MoE wants a **sparse / ragged AllToAll** — at most $`k`$ of the $`N`$ output shards are non-zero, since each token goes to $`k`$ experts — costing $`\min(k/N, 1) \cdot B/(W \cdot N)`$
  - **Across nodes, reductions go up the tree**: first within the node, then at the leaf, then at the spine, running the normal algorithm at each level. For an AllReduce this moves _less_ data overall — after the node-level reduce only $`B`$ bytes egress up to the leaf instead of $`B \cdot N`$. To first order $`T_{AG\ \text{or}\ RS} \approx \text{bytes}/W_{node}`$. 
    - The precise rule for a tree $`\textbf{AllGather}_X(A_Y\{U_X\})`$ with $`Y`$ the inner axis: <div align="center">
    $`\displaystyle T = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]`$ </div>
    with $`D_i`$ = children at depth $`i`$, $`W_{\text{link } i}`$ = bandwidth of the link from each child up to level $`i`$, and $`S_i`$ = size of the subtree below level $`i`$. Roughly: the more GPUs or nodes we span, the more aggregate bandwidth we get — but only within that level of the tree. 
    - **In-network reductions (SHARP)**: the InfiniBand switches can do the reduction themselves — in theory close to halving an AllReduce, ~30% in practice

| Operation | Syntax | TPU ring ($`W_{ici}`$) | GPU within a node ($`W_{GPU}`$) | GPU across nodes ($`W_{node}`$) |
|---|---|---|---|---|
| **AllGather** (removes a subscript) | $`[A_X, B] \to [A, B]`$ | bytes / ($`W_{ici}`$ × num axes) | bytes / $`W_{GPU}`$ | bytes / $`W_{node}`$ — tree, reduce within the node first |
| **ReduceScatter** (sums, adds a subscript) | $`[A, B]\{U_X\} \to [A_X, B]`$ | same as AllGather | same | same |
| **AllReduce** (removes a $`\{U_X\}`$) | $`[A_X, B]\{U_Y\} \to [A_X, B]`$ | 2 × AllGather | 2 × AllGather | 2 × AllGather, less with SHARP |
| **AllToAll** (reshards without replicating) | $`[A, B_X] \to [A_X, B]`$ | AllGather / 4 (bidirectional ring) | $`B / (N \cdot W_{GPU})`$ — AllGather / 8 at $`N{=}8`$ | only the $`(Z-8)/Z`$ of shards on other nodes leaves the node — see [EP](#expert-parallelism-ep) |

## Recipe

> **Intuition first, mechanics later.** This section states the decision procedure up front and leans on results derived further down — the compute-bound conditions in [Strategies](#strategies), the PP / ZeRO-3 interaction in [Pipeline parallelism](#pipeline-parallelism), the two worked runs in the [case study](#case-study-llama-3-70b-on-tpu-pods-vs-h100-clusters). The framing follows the Ultra-Scale Playbook's [Finding the best training configuration](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=step_1:_fitting_a_training_step_in_memory) — three steps: fit, batch, throughput; its [cheatsheet](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ultra-cheatsheet.svg) is the one-page version — and the thresholds are the scaling book's ([GPU chapter TLDR](https://jax-ml.github.io/scaling-book/gpus/)). Training only — inference has its own rules in [Inference §Sharding for inference](inference.md#sharding-for-inference).

### Arithmetic Intensity

- Every compute-bound condition in [Strategies](#strategies) is the same inequality: **work per device per step must exceed $`\alpha = C/W`$** — peak FLOPs/s over the bandwidth of the link the collective runs on - the network's arithmetic intensity.
- What differs between hardware is only _which_ $`W`$, and what multiplies it:

| Fabric | $`C`$ (bf16 FLOPs/s) | $`W`$ (bytes/s) | $`\alpha`$ |
|---|---|---|---|
| TPU v5p, one ICI axis (bidirectional) | $`4.59 \times 10^{14}`$ | $`1.8 \times 10^{11}`$ | 2550 |
| TPU v5e, one ICI axis | $`1.97 \times 10^{14}`$ | $`9 \times 10^{10}`$ | ~2200 |
| TPU v5p, DCN between pods (per slice) | | | ~73,440 |
| H100, NVLink within a node (per-GPU egress) | $`9.9 \times 10^{14}`$ | $`4.5 \times 10^{11}`$ | 2200 |
| H100, InfiniBand across nodes (per-node egress) | $`9.9 \times 10^{14}`$ | $`4 \times 10^{11}`$ | 2475 |

- **Takeaway: ~2,200–2,550 tokens per chip on every fabric.** $`C`$ and $`W`$ have scaled together, so the answer hasn't moved across a hardware generation or across vendors
- Multipliers on $`W`$ — the one place the topologies genuinely differ:
  - TPU: giving a collective $`M`$ mesh axes multiplies $`W_{ici}`$ by $`M`$ (the $`M_X, M_Y`$ in [Combining FSDP and TP](#combining-fsdp-and-tp)). A torus has 2–3 axes to spend, and they're uniform — no cliff between "near" and "far" chips until you leave the pod for DCN
  - GPU: no axis multiplier — past the node there is exactly one 400 GB/s pipe. What you get instead is the $`(n-1)/n`$ ring factor we usually drop: a collective over only 2 nodes moves half the bytes, so its threshold halves

### The three questions

Underneath the three steps below are three questions, asked in order:

1. **Does it fit in memory (for the number of GPUs we have)?** This is hard constraint:
- FSDP, TP, PP and EP divide parameters + optimizer state
- TP and CP divide activations
- PP _concentrates_ activations at early stages rather than dividing them
- Pick the axis that shrinks whatever is too big
2. **Is every collective compute-bound or bandwidth-bound?** We want to be compute-bound. 
- The global batch (number of tokens in a batch) is set by optimization. I.e. batch $B$ is a scarce resource (fixed)
- DP/FSDP is **batch-hungry** ($`B/X > \alpha`$ tokens per device). Shard too much and we're no longer compute-bound.
- TP and EP are **batch-free** ($`F/Y > \alpha`$ and the $`F_{expert}`$ regimes have no $`B`$ in them). How much we can shard is determined by hidden dimension. 
- PP needs microbatches, but they come out of the same batch.
- Batch-free approaches complement batch-hungry ones and allow us to shard more whilst staying compute-bound.
3. **Which link does each axis run on?** 
- Per-layer, critical-path traffic (TP, EP's AllToAll, CP's KV ring) goes on the fastest link
- Once-per-step, per-stage traffic and non-critical path traffic (DP/FSDP, PP) can cross the slow one. 
- That gives Llama 3's mesh order, innermost → outermost, of TP, CP, PP, DP — TP=8, PP=16, DP=128 on 16k GPUs, with CP=16 switched on only for the 128k-context stages ([Llama 3, arXiv 2407.21783](https://arxiv.org/abs/2407.21783) §3.3 — verify against source) — and on TPUs, TP/FSDP inside the pod and DP over DCN

### Step 1: Fit a training step in memory

The minimum number of GPUs for a given microbatch we need is determined by memory constraints.

Per-GPU memory = bf16 params + fp32 master weights + fp32 grads + optimizer states + activations — the 16 bytes/param of [Basics §Training vs Inference](basics.md#training-vs-inference), plus activations. Each axis divides a different term:

| Axis | Weights + optimizer state | Activations |
|---|---|---|
| DP | replicated | ÷ dp — each replica sees fewer samples |
| ZeRO-1 / 2 / 3 | optimizer states ÷ dp; ZeRO-2 also grads; ZeRO-3 (= FSDP) also params | same as DP |
| TP (+SP) | ÷ tp | ÷ tp — along $`D`$ in the matmuls, along $`s`$ in the SP regions |
| PP | ÷ pp, by layers | ≈ unchanged — each stage keeps ~pp microbatches in flight |
| CP | unchanged | ÷ cp, along $`s`$ |
| EP | expert weights ÷ ep; attention and shared weights replicated | unchanged per token |

- **Fitting is a constraint.** Several combinations will fit. Identify what's too big, then pick the axis that shrinks it with the cheapest comms for the batch and links you have (Steps 2–3):
  - Parameters + optimizer state too big → **ZeRO-3**, if tokens per GPU clear $`\alpha`$ (model-agnostic, off the critical path). If the batch can't feed that many shards → **TP** (batch-free, ≤ ~8-way), then **PP**
  - One layer's gathered weights too big for HBM → **TP** — ZeRO-3 still materialises a full layer at a time
  - Activations too big → smaller microbatch + gradient accumulation, recompute, **TP** (÷ tp), or **CP** when sequence length is the cause
  - MoE → **EP**: the experts are where the parameters are, and DP's threshold inflates by $`E/k`$ ([Data parallelism](#data-parallelism))
- Worked example: 70B with Adam is ≈ 1.1 TB of state, on 80 GB GPUs. TP=8 → 140 GB per GPU, doesn't fit. TP=8 × PP=2 → 70 GB, fits tightly. TP=8 + ZeRO-3 over ≥ 2 replicas → fits. **Three configurations fit; what separates them is whether tokens per GPU clear α for the ZeRO-3 gathers, and whether you can afford PP's microbatching, which forbids ZeRO-3.** 
- The playbook's breakpoints, GPU-rich case: < 10B → a single technique, TP or ZeRO-3 across 8 GPUs; 10B+ → TP=8 + PP, TP=8 + ZeRO-3, or pure ZeRO-3; 512+ GPUs → combine DP with TP or PP; 1024+ → TP=8 + DP (ZeRO-2) + PP. Long context → add CP; MoE → add EP
- GPU-poor case: full activation recompute (~⅓ more FLOPs) and gradient accumulation — trade compute and step time for memory

### Step 2: Hit the target global batch size

- GBS is chosen based on optimization dynamics.
- $`\text{GBS (tokens)} = \text{mbs} \times \text{gas} \times \text{dp} \times \text{seq}`$. Step 1 pinned tp, pp, cp, ep and mbs, so one replica occupies tp·pp·cp·ep GPUs. Two knobs remain: **dp** (more replicas — more GPUs, same step time) and **gas** (gradient accumulation — same GPUs, longer step)
- Scale up: more dp or gas. Scale down: less dp, in favour of other axes. **Scaling up via dp means adding GPUs; via gas doesn't**
- **Why CP is the knob for long sequences**: DP's unit is a _sample_, and each replica needs ≥ 1 per microbatch. At a 1M-token sequence, a 4M-token batch is four samples, so dp ≤ 4 and DP can't absorb more GPUs. CP splits a single sample, so it's the only axis that adds GPUs when the batch arrives as a few long samples — and it's cheap exactly there, because the KV ring overlaps with attention, which dominates compute at long context
- What this step really guards against — naive scaling fails three ways:
  1. **Comms-bound**: GBS is fixed, so more GPUs → fewer tokens per DP shard → below $`\alpha`$. The whole reason the batch-free axes (TP, EP, PP) exist
  2. **Memory-bound**: raising mbs to reach the batch blows activations → use gas instead. Gas is free for DP and ZeRO-1/2 (the reduce happens once per optimizer step, so every accumulated microbatch counts toward α) but _not_ for ZeRO-3, which re-gathers weights every microbatch — the same mechanism that makes PP + ZeRO-3 a bad pair
  3. **Idle-bound**: PP bubbles when gas is small relative to pp; EP load imbalance

### Step 3: Optimize throughput

- The conventional path (playbook), in order: **scale TP up to the node size** (fast intra-node links; it reduces how much of the other axes you need) → **increase DP with ZeRO-3 while holding the target GBS** → **switch to PP when DP's communication becomes the bottleneck** → then **tune mbs** to balance memory, matmul size and comms. This is the three questions in procedure form: TP first because it's batch-free and on the fast link, DP because it's the cheapest to hide, PP when DP hits α
- **Compute-bound is necessary, not sufficient.** Two configurations can both satisfy $`T_{comms} < T_{math}`$ and still differ in MFU:
  - _Exposed vs hidden comms_: the ratio says comms are hideable, not hidden. TP's collectives sit on the critical path and are only hidden with async TP; DP/FSDP overlap by construction. At equal ratios, prefer the axis that overlaps
  - _Bubbles_: PP loses ≈ $`(pp-1)/\text{gas}`$ of the step whatever the comms; zero-bubble schedules shrink it, at a code cost
  - _Local kernel efficiency_: compute-bound against the network ≠ high utilisation on the SMs. TP slices $`F`$ into thin matmuls, and a small mbs gives small tiles — hence "play with mbs"
  - _Memory headroom_: near the HBM limit you pay recompute (~⅓ more FLOPs for full checkpointing) or allocator thrash — the playbook's 80B-on-4-nodes example fits but runs badly
  - _Tails that can't overlap_: the last layer's gradient reduce, PP waiting for its final microbatch, ZeRO-3's first gather
  - _Simplicity_: ZeRO and PP are model-agnostic; TP needs model-specific sharding (+SP). The playbook's benchmarks saw the TP-vs-PP ranking flip with implementation quality
- **So the recipe below is the conventional, lowest-risk path to compute-bound; the residual is settled by benchmarking** — the playbook's own words are "no general recipe, experiment"

### The recipe

- **Relatively small dense model**: aggressive FSDP if you have the batch size (~2500 tokens per device), plus some PP or TP if it doesn't fit
  - Why FSDP (questions 1 + 2): 16 bytes/param of Adam state doesn't fit one GPU replicated, but sharded over a few dozen it does; FSDP is the only axis whose comms are fully off the critical path, so it's the cheapest to hide; and TP has little to give a small model — its cap $`F/\alpha`$ is ~4-way at $`F = 11k`$ (7B) and ~2-way at $`F = 5.5k`$ (1B). "If you have the batch size" is the one condition. "Plus PP or TP if it doesn't fit" is mostly about activations, which FSDP doesn't shrink
- **Larger dense model**: 1–2-node TP + many-node PP + pure DP. On a TPU: TP/FSDP within the ICI slice, pure DP across pods over DCN
  - Why TP, then PP, then DP — the order _is_ the preference (Megatron's classic 3D layout). TP first: batch-free, shrinks weights _and_ activations, up to its ~8-way cap — which happens to be one node. PP second: the only axis that crosses nodes for ~free, eats devices without needing batch, and divides the weight bytes FSDP must gather per replica. DP last, in the ZeRO-1 flavour since PP breaks ZeRO-3's amortization. **CP is orthogonal** — add it when sequence length, not parameter count, is the memory problem
  - Why pure DP across pods, TP/FSDP inside (the [case study](#tpu-v5p)): question 3. The slow link gets the traffic that's smallest, least frequent and most overlappable — pure DP reduces gradients once per step, async behind the backward pass, and since the pod already shards the weights internally, each chip only exchanges its own shard with its twin in the other pod. FSDP over DCN would gather weights every layer on the slow link; TP would put critical-path activations on it. And you never _need_ model parallelism across pods — a pod already holds the model
- **MoE**: the same, plus EP, which is generally preferred to TP. $`F > 8\,C/W_{node}`$ → lots of multi-node EP; otherwise roughly 2-node EP
  - Why EP over TP: experts have small $`F`$ — DeepSeek-V3's is 2048, below $`\alpha`$ before you split it at all, so TP on an expert is comms-bound at $`Y = 1`$ and slices the matmul into thin, low-utilization tiles. EP keeps each expert whole and splits along $`E`$ instead; its comms are the sparse AllToAll (each token's activation moves only to its $`k`$ experts) vs TP's AllGather + ReduceScatter of every token every layer; it's batch-free; and it shards exactly where the parameters live, fixing the $`E/k`$ inflation DP suffers
- **PP** works well if you can handle the code complexity of zero-bubble schedules and keep batch sizes large enough to avoid DP bottlenecks. It usually rules out ZeRO-3 (an AllGather per pipeline stage) but ZeRO-1 is fine
- **Any model parallelism that spans nodes reduces the FSDP cost** — the reason the mixes above exist
- **Would we ever skip TP? Yes.** DeepSeek-V3 trained with 16-way PP, 64-way EP spanning 8 nodes and ZeRO-1 DP — no TP at all, called out in the report as avoiding "costly" tensor parallelism ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437) §3.2 — verify against source). Small dense models skip it for the opposite reason: $`F`$ is too small for TP to help. **TP is the axis for when you must consume devices without batch and cut per-device weight bytes at the lowest latency; if EP + PP + FSDP already fit and clear α with the batch you have, TP buys nothing and charges per-layer critical-path comms**
  - Why the skip was forced by α, not taste: for a MoE, EP does TP's job — shard the FFN weights — along $`E`$ with whole experts instead of along $`F`$ with per-layer dense collectives. V3's experts have $`F = 2048`$, so TP=8 would leave 256-wide slices against an α in the thousands. Their H800s make it worse: NVLink is 160 GB/s per GPU (V3 report §3.2, vs the H100's 450 in the [α table](#arithmetic-intensity)), so in-node α ≈ 6200 in bf16 and ≈ 12,400 in FP8 — even the dense $`d_{model} = 7168`$ projections sit 7–14× below the line at TP=8, and DualPipe can't hide per-layer critical-path collectives
  - Memory didn't need it either: 16-way PP × 64-way EP is already 1024-way weight sharding (~650M params per GPU), and the report frames its activation tricks — recompute RMSNorm + MLA up-projections, FP8 activations, EMA on CPU — as what let them train "without costly TP". The dense DeepSeek LLM 67B _did_ use TP (HAI-LLM: DP/TP/SP/1F1B — verify against source); the switch coincides with going MoE
- Scale-out, serving edition ([vLLM](https://www.aleksagordic.com/blog/vllm)): TP within the node → PP across nodes if it still doesn't fit → DP replicas behind a lightweight coordination layer, load balancing across replicas, one or more API servers in front

## Strategies

_From [training](https://jax-ml.github.io/scaling-book/training/) and [GPUs](https://jax-ml.github.io/scaling-book/gpus/). For each strategy the same three questions: **what is sharded**, **what is communicated per step**, **at what per-device batch size does it go bandwidth-bound**. Worth keeping the answers in that parallel shape so the summary table below writes itself._

For simplicity’s sake, we’ll approximate a Transformer as a stack of MLP blocks. 

### Reference: a single 2-matmul layer

The strategies below all shard this same toy layer — $`\text{In}[B,D] \to \text{Tmp}[B,F] \to \text{Out}[B,D]`$ — so its forward/backward can be checked against each strategy's sharded version.

- **Forward pass**: need to compute $`\text{Loss}[B]`$
  1. $`\text{Tmp}[B,F] = \text{In}[B,D] \cdot_D W_{in}[D,F]`$
  2. $`\text{Out}[B,D] = \text{Tmp}[B,F] \cdot_F W_{out}[F,D]`$
  3. $`\text{Loss}[B] = \dots`$
- **Backward pass**: need to compute $`dW_{out}[F,D], dW_{in}[D,F]`$ — each line below is one of the two general backward formulas from [FLOPs](basics.md#basics) ($`\partial L/\partial A = (\partial L/\partial C)B^\top`$, $`\partial L/\partial B = A^\top(\partial L/\partial C)`$), applied to the two forward matmuls in reverse order
  1. $`d\text{Out}[B,D] = \dots`$
  2. $`dW_{out}[F,D] = \text{Tmp}[B,F]^\top \cdot d\text{Out}[B,D]`$
  3. $`d\text{Tmp}[B,F] = d\text{Out}[B,D] \cdot W_{out}[F,D]^\top`$
  4. $`dW_{in}[D,F] = \text{In}[B,D]^\top \cdot d\text{Tmp}[B,F]`$
  5. $`d\text{In}[B,D] = d\text{Tmp}[B,F] \cdot W_{in}[D,F]^\top`$ (_needed for previous layers_)

### Summary

([scaling book](https://jax-ml.github.io/scaling-book/training/); CP/SP/EP per the [Megatron parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html))

1. **Data parallelism**: _activations sharded along batch, parameters and optimizer state are replicated on each device. Communication only occurs during the backwards pass._ <div align="center">
   $`\displaystyle \text{In}[B_X, D] \cdot_D W_{in}[D, F] \cdot_F W_{out}[F, D] \to \text{Out}[B_X, D]`$ </div>
2. **Fully-sharded data parallelism (FSDP or ZeRO-3)**: _activations sharded along batch (like pure data parallelism), parameters sharded along the same mesh axis and AllGathered just-in-time before use in the forward pass. Optimizer state also sharded along batch. Reduces duplicated memory._ <div align="center">
   $`\displaystyle \text{In}[B_X, D] \cdot_D W_{in}[D_X, F] \cdot_F W_{out}[F, D_X] \to \text{Out}[B_X, D]`$ </div>
3. **Tensor parallelism (also called Megatron sharding or model parallelism)**: _activations sharded along $`D`$ ($`d_{model}`$), parameters sharded along $`F`$ ($`d_{ff}`$). AllGather and ReduceScatter activations before and after each block. Compatible with FSDP._ <div align="center">
   $`\displaystyle \text{In}[B, D_Y] \cdot_D W_{in}[D, F_Y] \cdot_F W_{out}[F_Y, D] \to \text{Out}[B, D_Y]`$ </div>
   - **Important intuition**: FSDP moves **weights**, TP moves **activations**.
4. **Sequence parallelism (SP)**: _extends tensor parallelism to shard the non-matmul ops (LayerNorm, dropout) along the sequence dimension too — only active when TP is, and covers exactly what plain TP leaves redundantly replicated._
5. **Context parallelism (CP)**: _shards activations along the sequence dimension across all layers (not just the ops SP covers) — the lever for long-context training, targeting attention's KV memory specifically._
6. **Pipeline parallelism**: _weights sharded along the layer dimension, activations microbatched and rolled along the layer dimension. Communication between pipeline stages is minimal (just moving activations over a single hop)._
7. **Expert parallelism (EP)**: _shards a MoE model's experts (not activations or a dense weight) across devices; routing tokens to their expert requires an AllToAll rather than an AllGather/ReduceScatter._

The thresholds, up front — each row is derived in its own section below, all in terms of the [α table](#arithmetic-intensity):

| Strategy | What is communicated | Compute-bound when | TPU (v5p) | GPU (H100) |
|---|---|---|---|---|
| DP / FSDP | weights: grads (DP) or params + grads (FSDP); off the critical path | $`B/X > \alpha`$ | ~2550 tokens/chip, ÷ $`M_X`$ ICI axes | ~2200 in-node, ~2475 across nodes; ÷2 for a 2-node ring; × $`E/k`$ for MoE |
| TP | activations, every block, on the critical path | $`Y < M_Y \cdot F/\alpha`$ | ~11-way per axis at $`F{\approx}30k`$ | $`F/2475`$ → 8-way (one node), 16-way over exactly 2 nodes |
| FSDP + TP | both — and each shrinks the other's bytes | $`B/N > \alpha^2 / (M_X M_Y F)`$ | ~100 tokens/chip at $`F{=}32k`$ | same idea: any multi-node model parallelism cuts FSDP's cost |
| PP | one microbatch of activations per stage boundary | ≈ free: $`1.5 \cdot 2BD / (W N_{layers})`$ | — | the cheap way across nodes; but ZeRO-3 stops amortizing |
| EP | routed tokens, sparse AllToAll | few nodes: $`F > \alpha (Z-8)/k`$; wide: $`F > 8\alpha`$ | AllToAll is $`B/4W`$, no node term | ~2-node EP unless $`F > 8\alpha \approx 20k`$ |
| CP | K/V chunks around a ring | cheap at long context (overlaps attention) | | |

### Data parallelism

Source: [scaling book](https://jax-ml.github.io/scaling-book/training/)

- Syntax: <div align="center">
  $`\displaystyle \text{In}[B_X, D] \cdot_D W_{in}[D, F] \cdot_F W_{out}[F, D] \to \text{Out}[B_X, D]`$ </div>
- Activations sharded along batch dimension, weights fully replicated
- Forward pass is normal — weights are replicated, so each device just runs the reference layer's forward pass on its own batch shard, no communication needed
- Backward pass: each device computes a **local, unreduced** gradient from its own batch shard (the $`\{U_X\}`$ tag), then AllReduces it across the batch axis to get the true (summed) gradient
  1. $`d\text{Out}[B_X,D] = \dots`$
  2. $`dW_{out}[F,D]\{U_X\} = \text{Tmp}[B_X,F]^\top \cdot d\text{Out}[B_X,D]`$
  3. $`dW_{out}[F,D] = \textbf{AllReduce}_X\left(dW_{out}[F,D]\{U_X\}\right)`$ (_not on the critical path — can be done async_)
  4. $`d\text{Tmp}[B_X,F] = d\text{Out}[B_X,D] \cdot W_{out}[F,D]^\top`$
  5. $`dW_{in}[D,F]\{U_X\} = \text{In}[B_X,D]^\top \cdot d\text{Tmp}[B_X,F]`$
  6. $`dW_{in}[D,F] = \textbf{AllReduce}_X\left(dW_{in}[D,F]\{U_X\}\right)`$ (_not on the critical path — can be done async_)
  7. $`d\text{In}[B_X,D] = d\text{Tmp}[B_X,F] \cdot W_{in}[D,F]^\top`$ (_needed for previous layers_)
- **Not on the critical path**: each AllReduce can be performed whenever convenient and doesn't block subsequent ops — in particular it can overlap with the backward pass of earlier layers, which is why DP communication is so cheap to hide
- Communication cost, per layer: $`T_{comms} = \dfrac{2 \cdot 2 \cdot 2 \cdot DF}{W}`$ — AllReduce is $`2\times`$ an AllGather, bf16 is 2 bytes/param, and there are 2 weight matrices ($`W_{in}, W_{out}`$), each $`DF`$ params. $`W`$ is $`W_{ici}`$ on a TPU; on a GPU it's $`W_{GPU}`$ while the DP axis stays inside the node and $`W_{node}`$ once it crosses (tree reduce, node first)
- Compute cost, per layer: $`T_{math} = \dfrac{4 \cdot 2(B/X)DF}{C}`$ — 4 matmuls in the backward pass ($`dW_{out}, d\text{Tmp}, dW_{in}, d\text{In}`$), each $`2(B/X)DF`$ FLOPs at per-chip batch size $`B/X`$
- **Compute-bound** when $`T_{math} > T_{comms}`$, i.e. per-chip batch size $`B/X > C/W = \alpha`$ — ≈2550 tokens on TPU v5p (one ICI axis; ÷ $`M_X`$ if FSDP gets more), ≈2200 tokens per GPU inside an H100 node, ≈2475 across nodes
- **MoE** ($`E`$ experts, $`k`$ active per token): $`E\times`$ the weights to reduce for only $`k\times`$ the FLOPs: <div align="center">
  $`\displaystyle T_{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}, \qquad T_{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W} \quad\Rightarrow\quad \frac{B}{X} > \frac{E}{k} \cdot \alpha`$ </div>
  - The critical per-device batch inflates by $`E/k`$, the ratio of total to activated parameters — this is what makes DP "significantly harder" for MoEs. Mitigation: shard the weights along the expert dimension instead ([EP](#expert-parallelism-ep))
- **Small rings**: the ring cost is really $`\text{bytes} \cdot (n-1)/n`$ over $`n`$ participants, which we drop for large $`n`$. When the DP axis spans only $`n`$ nodes of a GPU cluster ($`N = 8n`$ GPUs): <div align="center">
  $`\displaystyle T_{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (n-1)}{n \cdot W_{node}} \quad\Rightarrow\quad \frac{B}{N} > \alpha \cdot \frac{n-1}{n}`$ </div>
  - 2-node DP → $`B/N > 1237`$ on H100, half the threshold. **This is why you see 2-way data parallelism so often.** In-network reductions (SHARP) + pure DP shave another ~30%
- **Takeaway: DP / ZeRO needs ~2500 tokens per device to be compute-bound — on v5p and on H100/B200 alike** (perfect overlap and full FLOPs utilization assumed)

### Fully-sharded data parallelism (FSDP / ZeRO-3)

- Syntax: <div align="center">
  $`\displaystyle \text{In}[B_X, D] \cdot_D W_{in}[D_X, F] \cdot_F W_{out}[F, D_X] \to \text{Out}[B_X, D]`$ </div>
- Practically, vanilla DP is rarely useful because our parameters + optimizer state don't fit in a single chip
- FSDP splits the model params and optimizer states across the data parallel shards and efficiently gathers and scatters them as needed
  - Recall: FSDP moves **weights**! 
- Forward pass, we AllGather the weights before matrix multiplication (not on critical path)
- Backward pass: need to compute $`dW_{out}[F,D_X], dW_{in}[D_X,F]`$ 
- Interesting bit, the process is the **same** as DP! Just split the AllReduce into a ReduceScatter + AllGather, and notice that the output of the ReduceScatter is what we're looking for.
- **Not on the critical path**: Similar to DP, comms are non-blocking.
- FLOPs : Comms ratio is the same as DP, and so ratios earlier apply. 
- If we're doing DP, there's no reason not to do FSDP! 
  - One exception: under pipeline parallelism ZeRO-3's per-microbatch weight AllGather stops amortizing — see [PP](#pipeline-parallelism). ZeRO-1 (shard only the optimizer state) still works there

### Tensor parallelism

- Syntax: <div align="center">
  $`\displaystyle \text{In}[B, D_Y] \cdot_D W_{in}[D, F_Y] \cdot_F W_{out}[F_Y, D] \to \text{Out}[B, D_Y]`$ </div>
- Let's reconcile something: Most people also say that TP shards model weights, so why are we sharding activations here? 
- The equivalence is that by sharding weights in the contradicting dimension, we get partial sums post matrix multiplication, and we need to AllReduce = ReduceScatter + AllGather. If we _snapshot_ at the ReduceScatter step, then we get the syntax above.
- Forward pass. We have to AllGather activations first. Post matrix multiplication, we have a partial sum and have to ReduceScatter. These are both **on the critical path**. 
  - Recall: TP moves **activations**!
- Backward pass. We have to AllGather activations first. Post matrix multiplication, our derivative wrt the input activations is a partial sum, so we have to ReduceScatter again. These are both **on the critical path**. (Note that we also have to AllGather input activations too, but "this can be skipped by sharing with the Forward pass")
- Backward pass: need to compute $`dW_{out}[F_Y, D], dW_{in}[D, F_Y]`$ 
  1. $`d\text{Out}[B, D_Y] = \dots`$
  2. $`d\text{Out}[B, D] = \textbf{AllGather}_Y\left(d\text{Out}[B, D_Y]\right)`$ (_on critical path_)
  3. $`dW_{out}[F_Y, D] = \text{Tmp}[B, F_Y]^\top \cdot d\text{Out}[B, D]`$
  4. $`d\text{Tmp}[B, F_Y] = d\text{Out}[B, D] \cdot W_{out}[F_Y, D]^\top`$ (_can throw away $`d\text{Out}[B,D]`$ here_)
  5. $`\text{In}[B, D] = \textbf{AllGather}_Y\left(\text{In}[B, D_Y]\right)`$ (_this can be skipped by sharing with (1) from the forward pass_)
  6. $`dW_{in}[D, F_Y] = \text{In}[B, D]^\top \cdot d\text{Tmp}[B, F_Y]`$
  7. $`d\text{In}[B, D]\{U_Y\} = d\text{Tmp}[B, F_Y] \cdot W_{in}[D, F_Y]^\top`$ (_needed for previous layers_)
  8. $`d\text{In}[B, D_Y] = \textbf{ReduceScatter}_Y\left(d\text{In}[B, D]\{U_Y\}\right)`$ (_on critical path_)
- **Compute-bound** when $`F/Y > C/W = \alpha`$. Note that compared to DP, we only swap out $`B`$ for $`F`$, because for comms, we are moving **activations**! 
- $`F/Y`$ isn't the most interpretable. Rather, we should say that $`Y < M_Y \cdot F/\alpha`$ to remain compute bound, where $`M_Y`$ is the number of ICI axes (on GPUs, $`M_Y = 1`$ always)
- **How many ways, in practice**
  - TPU v5p: $`Y < M_Y \cdot F/2550`$ — at $`F \approx 30k`$ that's ~11-way per ICI axis (the book's rule of thumb is 4–8-way per axis), more if TP is given a second axis
  - GPU: $`Y < F/2200`$ inside a node, $`F/2475`$ across nodes. LLaMA-3 ($`F = 28{,}000`$): ~11-way, i.e. rounding down to the node, **8-way TP**. Spanning exactly 2 nodes gets the extra 2× from the small-ring factor, so 16-way TP generally works ($`F > 2475 \cdot (Y - 8)`$), up to ~19-way in theory
  - **Takeaway: on GPUs, TP goes comms-bound past $`Y > F/2475`$ → intra-node TP, or at most 2-node TP.** The torus has no such cliff (ICI is uniform), but lands at the same order of magnitude; on GPUs the only way to keep scaling TP is a bigger NVLink domain (GB200 NVL72: 72 GPUs)
  - Serving systems follow the same rule ([vLLM](https://www.aleksagordic.com/blog/vllm)): if the model doesn't fit one GPU, TP across the node first (TP=8). TP's fine-grained, on-critical-path AllGather/ReduceScatter every block is exactly what needs NVLink's bandwidth and latency

### Combining FSDP and TP

- Syntax: <div align="center">
  $`\displaystyle \text{In}[B_X, D_Y] \cdot_D W_{in}[D_X, F_Y] \cdot_F W_{out}[F_Y, D_X] \to \text{Out}[B_X, D_Y]`$ </div>
- Because FSDP shards activations in the X axis, we reduce the size of activations needed to move in TP. 
  - Tensor parallelism performs $`\textbf{AllGather}_Y([B_X, D_Y])`$ which shrinks as $`X`$ grows
- Similarly, because TP shards weights in the Y axis, we reduce the size of weights needed to move in FSDP.
  - FSDP performs $`\textbf{AllGather}_X([D_X, F_Y])`$ which shrinks as $`Y`$ grows
- So by combining both we push the minimum batch size per replica down further. Let $`X`$ = chips on FSDP, $`Y`$ = chips on TP, $`N = XY`$, and $`M_X, M_Y`$ = the number of mesh axes given to each (these should roughly sum to 3)
  - FSDP comms grow with $`X`$ while TP comms shrink with $`X`$, so total comms $`\max(T_{\text{FSDP}}, T_{\text{TP}})`$ is minimised where the two are equal: $`X_{opt} = \sqrt{\dfrac{B}{F}\dfrac{M_X}{M_Y}N}`$
  - e.g. $`N{=}64`$ (a 4×4×4 array), $`B{=}48{,}000`$, $`F{=}32{,}768`$ → $`X \approx 13.9`$, so pick $`X{=}16, Y{=}4`$
- Compute-bound condition, with $`\alpha \equiv C/W_{ici}`$ (the ICI arithmetic intensity): $`\dfrac{B}{N} > \dfrac{\alpha^2}{M_X M_Y F}`$
  - Plugging in $`F{=}32{,}768`$, $`\alpha{=}2550`$, $`M_X M_Y = 2`$ (as it must be for a 3D mesh) gives $`B/N > 99`$, vs ~850 for the pure FSDP case — a factor of ~8
  - I.e. because the parallelism schemes are complementary, $`T_{comms}`$ decreases and we saturate the chips more easily
  - Note the scaling asymmetry driving this: compute time scales linearly with batch size, but comms time only as $`\sqrt{B}`$ — so $`T_{math}/T_{comms} = \sqrt{BF}\sqrt{M_X M_Y} / (\alpha\sqrt{N})`$ grows as $`\sqrt{B}`$
- The same logic holds on GPUs, and it generalises: **any model parallelism that spans nodes — TP, PP or EP — shrinks the weight bytes FSDP has to gather per replica.** That's why the GPU [recipe](#recipe) mixes PP + EP + TP across many nodes to bring the FSDP cost down, where the TPU version spends extra ICI axes instead

### Pipeline parallelism

- Note that we can _also_ combine pipeline parallelism! In the pseudocode above, TPU 0 is almost always idle.
- Pipeline parallelism splits the model across devices, whilst introducing minimal communication across devices, although also facing the pipeline bubble issue.
  - ![pipeline1.png](images/pipeline1.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
- _(existing)_ Micro-Batching
  - Micro-Batching mitigates the pipeline bubble issue.
  - ![pipeline2.png](images/pipeline2.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
- _(existing)_ Looping Pipelines
  - Looping mitigates the pipeline bubble issue further.
  - ![pipeline3.png](images/pipeline3.png)![pipeline4.png](images/pipeline4.png)[Source](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_looping.html)
- We can also, per DeepSeek v3, carefully overlap the forward and backward matmuls.
- Communication cost (source: [scaling book — GPUs](https://jax-ml.github.io/scaling-book/gpus/), but hardware-agnostic): with $`N_{MB}`$ microbatches and $`N_{stages}`$ stages there are $`N_{MB} + N_{stages} - 2`$ hops of $`2BD / N_{MB}`$ bytes each: <div align="center">
  $`\displaystyle T_{total\ PP\ comms} = \frac{2BD}{W \cdot N_{MB}} \cdot (N_{MB} + N_{stages} - 2), \qquad T_{per\text{-}layer} \approx 1.5 \cdot \frac{2BD}{W \cdot N_{layers}}`$ </div>
  - We're dividing by $`N_{layers}`$, so this is vastly smaller than any other cost. **From a communication standpoint, pipelining is basically free** — and zero-bubble schedules mostly remove the bubbles too
- So why not just pipeline everything?
  1. **Code complexity**: doesn't fit automatic-parallelism frameworks like XLA's GSPMD — microbatching changes the structure of the program, and custom zero-bubble schedules make it worse by interleaving forward and backward in complicated ways
  2. **It plays badly with DP and FSDP** — probably the biggest reason. ZeRO-3 in particular: it has to AllGather the weights on _every microbatch_, with only $`B / N_{MB}`$ tokens to amortize each gather. And in the backward pass we can't AllReduce or ReduceScatter a stage's gradients until the _last_ microbatch has passed that stage → significant non-overlapped communication. ZeRO-1 still works
  3. **Bubbles and step imbalance**: careful scheduling reduces bubbles but some usually remain, and the stage-to-stage activation hand-off sits on the critical path
- **Where the hardware matters**: on a GPU cluster PP is _the_ cheap way to cross node boundaries — one small activation hand-off per stage tolerates the InfiniBand drop that TP can't ([vLLM](https://www.aleksagordic.com/blog/vllm): TP=8 inside the node, then PP across nodes when the model still doesn't fit). On a TPU the torus is uniform, so PP earns its keep only through the FSDP-cost reduction above, not through topology

### Sequence parallelism (SP)

Source: [Megatron parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

- Only active when TP is active (`tensor_model_parallel_size > 1`) — it's **TP extended to shard the non-matmul ops too** (LayerNorm, dropout) along the sequence dimension, which plain TP otherwise leaves redundantly replicated on every device
- Fills the gap TP leaves open: TP shards $`D`$ inside the matmuls, but the LayerNorm/dropout between blocks were still computed redundantly everywhere — SP shards those along sequence instead, removing the redundant compute at ~no extra communication (reuses the AllGather/ReduceScatter boundary TP already pays for at each block)

### Context parallelism (CP)

Source: [Megatron parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

- Shards activations along the **sequence** dimension, across *all* layers (unlike SP, which only covers the specific ops TP leaves unsharded) — the lever for **long-context training**, where activation memory (the $`5as^2b`$ term — see [Basics](basics.md#training-vs-inference)) would otherwise dominate
- Targets attention specifically: each device holds only its sequence chunk's queries, plus only the K/V it needs; during the backward pass, missing KV pairs are exchanged via point-to-point sends around a ring (an AllGather/ReduceScatter reshaped into ring hops) rather than a single AllGather
- Composes with TP/PP/DP as another mesh axis: `data_parallel_size = world_size / (tensor_model_parallel_size × pipeline_model_parallel_size × context_parallel_size)`
- Two main implementations, differing in _how_ the missing KV reaches each device:
  - **Ring attention** (what the point-to-point description above is): each device keeps only its own KV chunk and passes it to the next device in a ring while computing partial attention on the chunk it currently holds (online-softmax accumulation, P2P — not a collective). Memory stays flat, and comms overlap with compute by construction, at the cost of $`P-1`$ sequential hops
  - **DeepSpeed-Ulysses**: reshard from "sequence-sharded, heads-complete" to "sequence-complete, heads-sharded" via an **AllToAll**, run ordinary (e.g. FlashAttention) attention on the now-complete sequence with a shard of the heads, then AllToAll back. Only 2 communication rounds (vs ring's $`P-1`$) and plays nicely with existing attention kernels, but the AllToAll sits on the critical path rather than overlapping with compute
  - (A third, simpler option — plain **AllGather-CP**: AllGather K/V across all CP devices, then compute locally — is the easiest to implement but briefly holds the *full* sequence's KV per device, undoing much of CP's memory win)

### Expert parallelism (EP)

Source: [Megatron parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html); cost model from [scaling book — GPUs](https://jax-ml.github.io/scaling-book/gpus/)

- Shards a MoE model's **experts** (not activations, nor a single dense weight) across devices — `expert_model_parallel_size` must divide the total expert count (e.g. 8 experts ÷ EP=4 → 2 experts/device)
- In practice, EP is typically used in conjunction with other forms of parallelism, such as data parallelism. This is because EP only affects the MoE layers and doesn't shard activations. If used in isolation, our GPUs would be doing redundant computation for all the non-MoE blocks.
- Routing needs an **AllToAll**, not an AllGather/ReduceScatter: tokens are dispatched to whichever device holds their selected expert, computed there, then AllToAll'd back to their originating device — the "dispatch/combine" step, and literally the AllToAll primitive from [§The collectives and their costs](#the-collectives-and-their-costs) above
- Unlike DP/TP/PP/CP — which shard a fixed, static computation graph — EP's communication pattern is **data-dependent** (which experts a batch of tokens picks), which is why load balancing (token dropping via a capacity factor, aux-loss balancing) is a live concern here and isn't for the others
- Cost model. Shard the weights along the expert dimension, $`W_{in}[E_Z, D, F]`$; the MLP block then needs the 2 AllToAlls above. On a GPU cluster with 8 GPUs per node, only the $`(Z-8)/Z`$ fraction of shards living on other nodes has to leave the node, and each token's $`k`$ experts touch at most $`\min(8k/Z, 1)`$ of the nodes: <div align="center">
  $`\displaystyle T_{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}, \qquad T_{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W_{node} \cdot Z} \cdot \min\left(\frac{8k}{Z}, 1\right)`$ </div>
  - Two regimes where EP is compute-bound ($`\alpha = C/W_{node}`$):
    - $`k > Z/8`$ (few nodes, every node gets touched): need $`F > \alpha (Z-8)/k`$ → **a small amount of EP, ~2 nodes**, workable even at small $`F`$
    - $`Z \gg k`$: need $`F > 8\alpha`$ → **$`Z`$ arbitrarily large, up to $`E`$-way EP**, if $`F`$ is large enough
    - Both show up in practice: DeepSeek-V3 (very small per-expert $`F`$) does a small, restricted cross-node EP; large-$`F`$ models do significant cross-node EP alongside TP
  - **Takeaway: if $`F < 8\,C/W_{node}`$ (≈ 19,800 on H100), EP spans 1–2 nodes at similar (slightly lower) cost to TP; if $`F > 8\,C/W_{node}`$, EP can span up to $`E`$ nodes at relatively low cost**
  - The $`(Z-8)/Z`$ and $`\min(8k/Z, 1)`$ terms are node-structure facts. On a TPU torus there's no node boundary: the AllToAll is just $`B/4W`$, times $`k/N`$ when sparse

### 5D parallelism (Megatron)

_To cover: how DP/TP/PP/CP/EP combine into one 5D device mesh in practice — [Megatron parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) — mirroring [Combining FSDP and TP](#combining-fsdp-and-tp) above but for the full stack. Don't forget to come back to this._

## Case study: LLaMA-3 70B on TPU pods vs H100 clusters

Same model, same problem — the per-chip batch is too thin for FSDP alone.

### TPU v5p

Source: [scaling book](https://jax-ml.github.io/scaling-book/training/)

0. A TPU v5p pod has ~8,960 chips ("8k"). Suppose we want to train LLaMA-3 70B ($`F \approx 30{,}000`$) at $`BS = 2M`$ tokens
1. Typically, when scaling beyond a single pod, we do TP or FSDP within the ICI domain, and pure DP across multiple pods (over DCN)
   - To be **compute-bound** over DCN: $`B/\text{slice} > C/W_{dcn} \approx 73{,}440`$ for TPU v5p
2. If we did pure FSDP: $`B/N > 2550/M_X`$ (the ICI threshold from [Data parallelism](#data-parallelism) above). At $`BS{=}2M`$ and 3 axes of FSDP, we'd be able to use at most $`\approx 2{,}400`$ chips — past that we're **comms-bound**, and adding more chips won't help
3. If we combine FSDP + TP: $`B/N > 108`$ (the [mixed threshold](#combining-fsdp-and-tp) above, using LLaMA-3 70B's $`F \approx 30{,}000`$), which lets us scale to $`\approx 18{,}000`$ chips
4. But $`18{,}000 > 8{,}960`$, so a single pod isn't enough — we need to cross DCN, using 2 pods ($`N \approx 17{,}920`$). This is fine — still **compute-bound**, not DCN-bound — because the per-pod batch size $`= 1M > 73{,}440`$ ✓, giving a per-chip $`B/N \approx 111`$: efficient, if cutting it a bit close
- **Takeaway**: scaling across multiple TPU pods with pure data parallelism is fairly straightforward, so long as the per-pod batch size clears the DCN threshold (~73k tokens for v5p)

### H100 cluster

Source: [scaling book — GPUs](https://jax-ml.github.io/scaling-book/gpus/)

0. 4,096 H100s, $`F = 28{,}672`$, batch ≈ 4M tokens
1. Model parallelism: 8-way TP is the most before going comms-bound ($`F/2475 \approx 11`$, rounded down to the node)
2. Memory: ~700 GB of training state ÷ 8 = 87.5 GB per GPU doesn't fit in 80 GB → ZeRO-3 across the DP axis as well
3. But 8-way TP × 512-way DP leaves $`4M / 4096 = 976`$ tokens per GPU — well under the ~2500 FSDP threshold → comms-bound
4. The fix is pipelining across nodes: PP divides the per-layer weight AllGather/ReduceScatter by the pipeline depth (fewer weight bytes per replica), pulling FSDP back under threshold. ~40 days at 45% MFU for 15T tokens
- **Takeaway, side by side**: on the torus, TP gets extra ICI axes and DP crosses pods over DCN, which the per-pod batch clears easily. On the GPU cluster TP is capped at the node, so PP across nodes is what buys back the FSDP cost
