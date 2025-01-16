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
