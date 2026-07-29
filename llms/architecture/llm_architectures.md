# LLM Architecture Comparison

> Draft — seeded from reading notes, to expand. Sources: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison), [Qwen3 from scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch), [DeepSeek technical deep-dive](https://magazine.sebastianraschka.com/p/technical-deepseek). Pre-2024 architecture deltas (GPT-2 → Llama → Mistral → Mixtral) are in [NLP Pre-Training §Model Trends](../../fundamentals/dl/17_nlp/pre_training.md).

## Model-by-model

- DeepSeek V3
  - MLA (similar role to GQA — see [Attention Variants](attention.md))
  - MoE
- DeepSeek V3.2
  - Sparse attention: lightning indexer + token selector
  - Post-training: rule-based outcome reward, length penalty, language-consistency reward (see [RL](../rl/grpo.md) and [Reasoning](../rl/reasoning.md))
- OLMo 2
  - Pre-LN, QK-Norm *(verify — Raschka describes OLMo 2's norm placement as a Post-Norm variant with QK-Norm)*
- Gemma 3
  - Sliding-window attention (SWA)
- Qwen 3
  - No dropout, RMSNorm, RoPE, QK-Norm, GQA, MoE
- GPT-OSS
  - vs Qwen3: GPT-OSS is the wider/shallower one, Qwen3 the deeper/narrower one (e.g. gpt-oss-20b: 24 layers at $`d=2880`$ vs Qwen3-30B-A3B: 48 layers at $`d=2048`$); depth buys flexibility but is harder to train due to instability issues, width buys inference speed at the cost of memory
  - Attention bias and attention sinks
- Qwen3-Next
  - More experts
  - Gated DeltaNet and Gated Attention
- MiniMax-M2
  - Partial RoPE — prevents "too much" rotation for sequences longer than the longest training documents; no rotation may be better than an extreme unseen rotation
- Kimi Linear
  - Kimi Delta Attention (KDA): channel-wise gating for memory decay rate
- OLMo 3
  - YaRN
- Qwen 3.5
  - Native tool-calling format is XML (the Qwen3-Coder style: `<function=name><parameter=key>value</parameter></function>` inside `<tool_call>` tags), not JSON — parameter values are raw text, so code / multi-line arguments need no JSON string-escaping; harnesses that assume Hermes-style JSON break until the parser targets the XML format

## mHC (manifold-constrained Hyper-Connections)

- HC (hyper-connections) generalize residual connections with a learned $`n \times n`$ mixing matrix across the residual streams
- mHC constrains that $`n \times n`$ mixing matrix to the Birkhoff polytope (doubly-stochastic matrices), implemented via the Sinkhorn-Knopp algorithm

## Multimodal architectures

- Unified Embedding-Decoder vs Cross-modality Attention — see [Multimodal LLMs](multimodal_llms.md)
