# Performance Engineering

| File | Contents |
|---|---|
| [training.md](training.md) | Single-device efficiency: bounds, memory reduction, batch size, speed checklist |
| [parallelism.md](parallelism.md) | DP/FSDP, pipeline, tensor, 3D parallelism; TP-intra-node vs PP-inter-node placement |
| [inference.md](inference.md) | Prefill/decode, batching, speculative decoding, chunked prefill, vLLM internals |
| [gpus.md](gpus.md) | SMs, warps, memory hierarchy, occupancy limits |
| [tpus.md](tpus.md) | Rooflines (arithmetic intensity), TPU organization, ICI/DCN vs GPU networking |
| [python_concurrency.md](python_concurrency.md) | OS basics, threading vs multiprocessing vs asyncio |

Notebooks: `01_single_processor.ipynb`, `02_data_parallelism.ipynb`, `03_pipeline_parallelism.ipynb` (JAX, from Lippe's UvA tutorials), `bloom_tensorrt_llm.ipynb` (TensorRT-LLM serving), `python_concurrency.ipynb`. Supporting modules: `utils.py`, `single_gpu.py`, `data_parallel.py`, `pipeline_parallel.py`.

Write-up backlog: see [TODO.md](../TODO.md#performance).
