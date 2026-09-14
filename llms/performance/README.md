# Performance Engineering

| File | Contents |
|---|---|
| [basics.md](basics.md) | Single-device efficiency: bounds, memory reduction, batch size, speed checklist |
| [parallelism.md](parallelism.md) | Training parallelism, TPU and GPU side by side: collectives and their costs, the α = C/W threshold table, DP/FSDP, TP, PP, CP, EP; LLaMA-3 70B case study on both fabrics; recipe |
| [inference.md](inference.md) | Prefill/decode arithmetic intensity, inference memory, sharding prefill vs generation (scaling book ch. 7); serving engines, vLLM internals, speculative decoding (n-gram / Medusa / EAGLE) |
| [gpus.md](gpus.md) | SM anatomy (Tensor/CUDA cores, SIMT vs SIMD, warp scheduling), memory hierarchy with H100/B200 numbers, GPU↔TPU glossary, NVLink/InfiniBand network topology, occupancy limits |
| [tpus.md](tpus.md) | Rooflines (arithmetic intensity), TPU organization, ICI/DCN vs GPU networking |
| [python_concurrency.md](python_concurrency.md) | OS basics, threading vs multiprocessing vs asyncio |

Notebooks: `01_single_processor.ipynb`, `02_data_parallelism.ipynb`, `03_pipeline_parallelism.ipynb` (JAX, from Lippe's UvA tutorials), `bloom_tensorrt_llm.ipynb` (TensorRT-LLM serving), `python_concurrency.ipynb`. Supporting modules: `utils.py`, `single_gpu.py`, `data_parallel.py`, `pipeline_parallel.py`.

Write-up backlog: see [TODO.md](../TODO.md#performance).
