# Model Architecture

| File | Contents |
|---|---|
| [attention.md](attention.md) | Combined QKV, MLA, sparse attention, DeltaNet/KDA linear attention, partial RoPE, attention sinks |
| [llm_architectures.md](llm_architectures.md) | Model-by-model: DeepSeek V3/V3.2, Qwen3/Next, Gemma 3, OLMo 2/3, GPT-OSS, MiniMax-M2, Kimi; mHC |
| [multimodal_llms.md](multimodal_llms.md) | Unified embedding-decoder vs cross-modality attention |

Mechanism deep-dive (attention math, RoPE wavelengths, GQA, flash attention, KV cache, Titans): [fundamentals/dl/08_attention_transformers](../../fundamentals/dl/08_attention_transformers/notes.md). Pre-2024 model deltas: [fundamentals/dl/17_nlp/pre_training.md §Model Trends](../../fundamentals/dl/17_nlp/pre_training.md).

Write-up backlog: see [TODO.md](../TODO.md#architecture).
