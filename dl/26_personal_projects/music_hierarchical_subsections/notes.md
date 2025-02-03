# Tackling Long Sequence Modelling Problems in Music Generation with Hierarchical Modelling

## Hypothesis

We can leverage commonly used musical forms to tackle long sequence modelling problems. 

## Problem

- The length of a typical music recording and its resolution means that we need architectures that support long context lengths. 
- Attention scales quadratically with respect to context length. 
- Various architectures like LDMs, or techniques like windowed sampling is used to address such issues.

## Key Idea

- Many songs are based off the form {Verse - Prechorus - Chorus - Verse - Prechorus - Chorus - Bridge - Chorus}, with optional omissions or repetitions. 
- The task of generating a long, coherent song is simplified with repetitions.
- We can generate each subsection separately, using cross-attention to ensure that the entire piece is coherent. 
- Assuming we generate our music in an auto-regressive way, 
  - It may also be possible to compress previously generated subsections to reduce the computational cost of these cross-attention layers.

## To Read

- [Motifs, Phrases, and Beyond: The Modelling of Structure in Symbolic Music Generation](https://arxiv.org/pdf/2403.07995)
- The [Harmony-Aware Hierarchical Music Transformer](https://arxiv.org/pdf/2109.06441) is able to improve the quality of generated music, especially in the form and texture, which indicates that the idea here has potential. 
  - I believe their method of tokenization to be quite manual, which restricts their training dataset. 
  - However, it may be sufficient since some other works have indicated that we don't need that much data to fine-tune music generation models (see [Audio Notes](../../19_audio/music))
- [MuseFormer](https://arxiv.org/pdf/2210.10349) uses fine and coarse grained attention to ultimately generate long music sequences with high quality and better structures. 
  - A limitation here is that they train to MIDI. 
  - While I have not focused much on works in symbolic music generation, this may be useful because other works (see [Audio Notes](../../19_audio/music)) have shown that it can be used as a conditioner for acoustic music generation. 
