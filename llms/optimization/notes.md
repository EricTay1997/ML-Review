# Optimization

> Draft — seeded from reading notes, to expand. Classic optimizer math (SGD/momentum/Adam derivations, second-order methods, LR-schedule theory) lives in [fundamentals/dl/04_optimization_and_regularization](../../fundamentals/dl/04_optimization_and_regularization/notes.md).

Width-scaling theory has its own files: [Kernels](kernels.md) — the notation and the three results (kernel trick, Mercer, representer theorem) the rest assumes — and [RKHS](rkhs.md) — the function-space view and the norm every kernel method regularises — then [NTK](ntk.md) — the infinite-width kernel / lazy-training regime, spectral view of convergence and early stopping — and [μP & μTransfer](muP.md) — the parametrization under which features still learn at infinite width, and the small-proxy hyperparameter-transfer recipe it enables.

## Batch size and learning dynamics

- Use the largest batch size that fits in memory — the only downside is learning dynamics; compensate by increasing the learning rate
- Hardware side of batch size (utilization, memory scaling): see [Performance / Training](../performance/basics.md#batch-size)

## To write up

- Muon, NTK, muP
