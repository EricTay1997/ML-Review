# Computational Performance

Overview

## Single Processor

- CPU vs GPU
- Multithreading vs Multiprocessing
  - The `threading` module uses threads, the `multiprocessing` module uses processes. 
  - The difference is that threads run in the same memory space, while processes have separate memory. 
  - This makes it a bit harder to share objects between processes with multiprocessing. 
  - Since threads use the same memory, precautions have to be taken or two threads will write to the same memory at the same time. 
  - This is what the global interpreter lock is for.
- Row-major vs Column-major
- Vectorization
- JAX
- Additional techniques
  - Mixed Precision Training
  - Gradient Checkpointing / Activation Recomputation
  - Gradient Accumulation

## Multiple Processors

### Data Parallelism

- Overview
- Parameter Sharding (Fully-sharded data parallelism)
- PyTorch
- JAX

### Pipeline Parallelism

- Overview
- Micro-Batching
- Looping Pipelines

### Tensor Parallelism

- Overview
  - Gather vs Scatter
- Asynchronous layers
- Transformers?

