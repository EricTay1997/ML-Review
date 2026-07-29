# Attention Variants

> Draft — seeded from reading notes, to expand. Mechanism foundations (attention math, RoPE, GQA, flash attention, KV cache) live in [Attention & Transformers](../../fundamentals/dl/08_attention_transformers/notes.md); this file tracks the modern variants models actually ship.

## Combined QKV projection

- 1 large GEMM (vs 3), fewer kernel launches
- Memory: reading $`X`$ once instead of 3 times
- Large matrix multiplies generally achieve better GPU occupancy and tensor core utilization
- Downsides: reshape/permute overhead, cache behavior
- See the line-by-line walkthrough of a merged-QKV implementation in [Basics](../../fundamentals/dl/01_basics/notes.md)

## Multi-head Latent Attention (MLA)

- DeepSeek V3's alternative to GQA: compress KV into a low-rank latent that is cached, then up-project
- Serves a similar purpose to GQA (shrink KV cache / bandwidth) — see [the big architecture comparison](llm_architectures.md)

## Sparse attention (DeepSeek)

- Lightning indexer + token selector: cheaply score candidate tokens, attend only to the selected subset
- Source: [Raschka's DeepSeek deep-dive](https://magazine.sebastianraschka.com/p/technical-deepseek)

## Linear attention: DeltaNet family

- DeltaNet (a Mamba2 upgrade)
  - $`S`$ (state) is a key-value association matrix ($`d_k \times d_v`$ — constant memory, $`O(Td^2)`$ compute)
  - Every step, decay by $`\alpha`$, update by $`\beta \cdot \delta`$, where $`\delta`$ is the difference between the new prediction and the old
  - Then multiply with $`q`$ to get the output
  - Qwen3-Next uses this (Gated DeltaNet, mixed with Gated Attention layers)
- Kimi Linear uses Kimi Delta Attention (KDA): channel-wise gating for the memory decay rate

## Positional-encoding variants

- M-RoPE / interleaved M-RoPE (Qwen-VL family) — written up in [Multimodal LLMs](multimodal_llms.md) (2D RoPE in the ViT, (t, h, w) triplets in the decoder, chunked→interleaved frequency allocation)

- Partial RoPE (MiniMax-M2)
  - Prevents "too much" rotation for long sequences, particularly those longer than the longest documents in the training dataset
  - I.e., the rationale could be that *no* rotation is better than a "bad" or "too extreme" rotation the model hasn't seen in training
- YaRN (OLMo 3) — context extension via frequency rescaling
- The RoPE wavelength/context-length analysis lives in [Attention & Transformers](../../fundamentals/dl/08_attention_transformers/notes.md)

## Attention bias and attention sinks

- GPT-OSS reintroduces attention bias and uses attention sinks (dedicated always-attended positions stabilizing long-context behavior)
