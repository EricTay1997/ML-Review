# GPUs

> Draft — seeded from reading notes, to expand. Primary source: [Aleksa Gordić's matmul deep-dive](https://www.aleksagordic.com/blog/matmul).

## CPU vs GPU

- CPUs and GPUs are processors.
- GPUs have many smaller, more specialized cores, which make them suited for parallel processing, e.g. tensor cores which compute matrix-matrix multiplications quickly.
- Terminology differences
  - A GPU is formed by multiple units named SMs (Streaming Multiprocessors); these function like CPU cores.
  - Each SM can execute many threads concurrently.
  - Threads are grouped into warps, a basic execution unit, where each warp contains 32 threads. These function like CPU threads.
  - While CPU threads can each execute different tasks at the same time, all GPU threads in a single warp can only execute one same task.
  - A threadblock is a collection of warps. The threads in the same thread block run on the same SM.
- PyTorch (amongst other libraries) allows us to use tensor cores for training DL models.

## Where things live

- HBM (GPU memory): model parameters, optimizer states, activations, gradients
- CPU (RAM): dataset / dataloader workers
- Disk: full dataset, checkpoints

## Memory hierarchy

- HBM (large, slow relative to on-chip)
- SRAM (on-chip): L2 cache, DSMEM, L1 cache/shared memory (SMEM), RMEM (registers)

## Compute: the Streaming Multiprocessor

- An SM contains tensor cores, CUDA cores and SFUs, load/store units, and warp schedulers.
- A warp scheduler can issue one warp instruction per cycle. A warp is a group of 32 threads.
- Each SM has 4 warp schedulers, where each scheduler can issue 1 instruction per cycle.
- Parallelism: an SM can *issue* instructions from at most four warps simultaneously (128 threads)
  - vs Concurrency: an SM can *host* up to 2048 concurrent threads.
- Peak throughput = maximum clock frequency × number of tensor cores × FLOPs per tensor core per cycle
  - But actual clock frequency can vary under power or thermal throttling.
- A thread is a single worker, but a warp is the execution unit. All threads in a warp execute the same instruction at the same cycle — this is called SIMT.
- Each thread gets a compiler-determined number of registers (up to 255), private to the thread until the block finishes; 32/thread is the budget at full occupancy (65,536 / 2,048).
- A thread block is a group of threads that can share data in fast on-chip memory. They are a group of up to 1024 threads that are guaranteed to be concurrently scheduled on a single SM. They not only share memory but synchronize well.
  - Blocks must all run on one SM since they share memory.

## Three resources limit concurrency (occupancy)

- Registers
  - Suppose we use thread blocks of 1024 threads, each thread has 32 registers; then since each SM has 65,536 registers, we can support 2 blocks per SM.
- Shared memory (SMEM)
  - System-level overhead of 1 KiB per block, on top of the kernel's own usage. (An A100 has up to 164 KB SMEM/SM: a kernel using $`S`$ bytes/block supports $`\lfloor 164\text{KB}/(S + 1\text{KiB}) \rfloor`$ blocks, further capped by the 32-blocks/SM architectural limit.)
- Threads/warps
  - Max number of threads per SM is 2048. With 1024 threads per block, we also have 2 blocks.
