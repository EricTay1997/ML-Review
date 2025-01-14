# Audio (In-Progress)

- Most of the time, we decompose audio into a spectrogram and treat it like an image. 
  - We use a vocoder to convert spectrograms back into audio
- Music Generation
  - Most of the papers I've read here use a [diffusion](../10_diffusion/notes.md) model to generate music from noise.
  - [MusicGen](https://arxiv.org/pdf/2306.05284) does so with a "pure" transformer architecture
    - ![musicgen.png](musicgen.png)[Source](https://hackernoon.com/musicgen-from-meta-ai-understanding-model-architecture-vector-quantization-and-model-conditioning)

- Decompose audio into a spectrogram and treat it like an image
    - There are [many usecases](https://towardsdatascience.com/audio-diffusion-generative-musics-secret-sauce-f625d0aca800) here, one thing that I find cool is the humanization aspect of sound variability!
    - Conditional Generation
      - Contrastive Language-Image Pretraining (CLAP) embeddings are useful here.
      - AudioLDM
        - ![audio_ldm.png](audio_ldm.png)
        - Outside of the audio-specific STFT/MelFB/Vocoder components, this is very similar to LDMs 
        - Note, however, that an important difference is the additional condition of audio encoding $E^{\mathbf{x}}$
      - AudioLDM2
        - ![audioldm2.png](audioldm2.png)
        - We replace the audio and text encodings $E^{\mathbf{x}}$ and $E^{\mathbf{y}}$ with AudioMAE Features
        - Self-supervision?
      - One of the main issues with generating audio using diffusion models is that diffusion models are usually trained to generate a fixed-size output. 
        - We can condition on music start time and duration
        - ![stability_audio.png](stability_audio.png)[Source](https://stability.ai/research/stable-audio-efficient-timing-latent-diffusion)

https://www.assemblyai.com/blog/recent-developments-in-generative-ai-for-audio/
- CLAP

- AudioLM: https://arxiv.org/pdf/2209.03143
- MusicLM: https://arxiv.org/pdf/2301.11325
- Steerable https://arxiv.org/pdf/2402.09508
- https://deepmind.google/discover/blog/pushing-the-frontiers-of-audio-generation/
- Agents: https://arxiv.org/pdf/2410.03335
- Scale-invariant convolutions: https://arxiv.org/pdf/2102.02282
- DO MUSIC GENERATION MODELS ENCODE MUSIC THEORY?: https://arxiv.org/pdf/2410.00872
- Foundation Models for Music: A Survey - https://arxiv.org/pdf/2408.14340
