# Hyperparameter Optimization

- Popular HPO libraries include Syne Tune (`syne_tune`), Ray Tune and Optuna. 
- HPO libraries consist of a:
  - Searcher: Suggests new candidate configurations, randomly, based on Bayesian optimization, etc.
  - Scheduler: Decides when to run a trial and resources to allocate
  - Tuner: Runs the searcher and scheduler and does bookkepping
- Multi-Fidelity Hyperparameter Optimization
  - Multi-fidelity hyperparameter optimization allocates more resources to promising configurations and stop evaluations of poorly performing ones early. 
  - For example, successive halving involves discarding the lowest performing $\frac{\eta-1}{\eta}$ trials at each rung. 
- Asynchronous scheduling
  - We immediately schedule a new trial as soon as resources become available, reducing downtime.
  - Asynchronous successive halving is tricky, and the solution is to accept imperfect promotions.
  - In practice, suboptimal initial promotions only have a modest impact on performance
    - Ranking of hyperparameter configurations is often fairly consistent across rung levels.
    - Rungs grow over time and become more accurate.
    - My gut feel is that we probably don't want to overfit to hyperparameters either, but that may be domain specific. 