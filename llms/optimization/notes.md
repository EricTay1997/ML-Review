# Optimization

> Draft — seeded from reading notes, to expand. Classic optimizer math (SGD/momentum/Adam derivations, second-order methods, LR-schedule theory) lives in [fundamentals/dl/04_optimization_and_regularization](../../fundamentals/dl/04_optimization_and_regularization/notes.md).

## Batch size and learning dynamics

- Use the largest batch size that fits in memory — the only downside is learning dynamics; compensate by increasing the learning rate
- Hardware side of batch size (utilization, memory scaling): see [Performance / Training](../performance/training.md#batch-size)

## To write up

- Muon (second-order-flavored optimizer for LLM pretraining)
- muP (Maximal Update Parametrization) — hyperparameter transfer across model widths
