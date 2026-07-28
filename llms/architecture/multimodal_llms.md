# Multimodal LLMs

> Draft — seeded from reading notes, to expand. Source: [Understanding Multimodal LLMs (Raschka)](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms). Earlier multimodal notes (NExT-GPT case study, encoder/decoder loss decomposition) are in [fundamentals/dl/21_multimodal](../../fundamentals/dl/21_multimodal/notes.md).

## Two architecture families

- Unified Embedding-Decoder architecture
  - Project image patches into the same embedding space as text tokens; a single decoder consumes the interleaved sequence
  - Examples: Qwen2-VL, Emu 3
- Cross-modality Attention architecture
  - Keep a separate vision encoder; inject visual features via cross-attention layers in the LLM
  - Example: Llama 3.2-V
