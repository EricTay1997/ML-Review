# Post-Training

- Post-Training  is a set of processes and techniques that refine and optimize a machine learning model after it's been trained.

## Model Optimization

- Quantization
- Pruning

## Fine-Tuning

- Re-train existing models on new data to change the type of output they produce.
- [Dreambooth](../10_diffusion/notes.md)
- LoRA

## Guidance

- Take an existing model and steer the generation process at inference time for additional control.
  - It's important to remember that we're _not_ changing the model internals. 
  - The model already has the ability to generate the conditioned outputs, we're instead e.g. restricting the inputs to guide this generation process. 
- (input) -> (output) -> (compute loss of output against objective), then backpropagate.

## Retrieval-Augmented Generation (RAG)