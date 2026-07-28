# Fundamentals

Knowledge accumulated in the past (originally for interview prep) — good reference, but dated in places. Current LLM-era notes live in [../llms/](../llms/).

## [classical/](classical/) — Classical (Non-DL) ML / Statistics

[Linear Algebra and Calculus](classical/01_linear_algebra_and_calculus/notes.md) ·
[Probability and Information Theory](classical/02_probability_and_info_theory/notes.md) ·
[Statistical Learning Theory](classical/03_statistical_learning_theory/notes.md) ·
[Testing and Metrics](classical/04_testing_and_metrics/notes.md) ·
[Bayesian Statistics](classical/05_bayesian_stats/notes.md) ·
[Linear Regression & Regularization](classical/06_linear_regression_and_regularization/notes.md) ·
[Naive Bayes, Logistic Regression & GLMs](classical/07_naive_bayes_and_logistic_regression_and_glms/notes.md) ·
[SVMs](classical/08_svms/notes.md) ·
[Decision Trees](classical/09_decision_trees/notes.md) ·
[Ensemble Learning](classical/10_ensemble_learning/notes.md) ·
[Dimensionality Reduction](classical/11_dimensionality_reduction/notes.md) ·
[Unsupervised Clustering](classical/12_unsupervised_clustering/notes.md) ·
[Gaussian Process](classical/13_gaussian_process/notes.md) ·
[Causal Inference](classical/14_causal_inference/notes.md) ·
[ARIMA](classical/15_arima/notes.md)

## [dl/](dl/) — Deep Learning

[Basics](dl/01_basics/notes.md) ·
[Activations](dl/02_activations/notes.md) ·
[Initialization](dl/03_initialization/notes.md) ·
[Optimization & Regularization](dl/04_optimization_and_regularization/notes.md) ·
[Coding Practices](dl/05_coding_practices/notes.md) ·
[CNNs](dl/06_cnns/notes.md) ·
[RNNs](dl/07_rnns/notes.md) ·
[Attention & Transformers](dl/08_attention_transformers/notes.md) ·
[Autoencoders](dl/09_autoencoders/notes.md) ·
[Diffusion](dl/10_diffusion/notes.md) ·
[Flows](dl/11_flows/notes.md) ·
[GANs & Adversarial Attacks](dl/12_gans/notes.md) ·
[GNNs](dl/13_gnns/notes.md) ·
[Meta-Learning](dl/14_meta_learning/notes.md) ·
[Contrastive Learning](dl/15_contrastive_learning/notes.md) ·
[Computer Vision](dl/16_computer_vision/notes.md) ·
[NLP](dl/17_nlp/) ·
[RL](dl/18_rl/notes.md) ·
[Audio](dl/19_audio/audio.md) / [Music](dl/19_audio/music.md) ·
[Video](dl/20_video/notes.md) ·
[Multimodal](dl/21_multimodal/notes.md) ·
[AI Safety](dl/23_safety/01_overview.md) ·
[Hyperparameter Optimization](dl/24_hyperparameter_optimization/notes.md) ·
[Personal Projects](dl/26_personal_projects/) ·
[Misc](dl/27_misc/notes.md)

Numbering gaps are intentional — those topics moved to `llms/`: 22_post_training → [llms/post_training](../llms/post_training/), 25_computational_performance → [llms/performance](../llms/performance/), 23_safety/05_evals → [llms/evals](../llms/evals/), 17_nlp's scalability was retired along with its old agents note (superseded by [llms/agents/harnesses.md](../llms/agents/harnesses.md)). 18_rl keeps the classic-RL notes; the LLM-era RL work lives in [llms/rl](../llms/rl/).

## [interview_prep/](interview_prep/)

ML system design case studies (recommenders, search, ETA, harmful-content classification, object detection) and a written/coding question bank.

## Code

Code implementations mostly come from online resources/tutorials; the first priority is learning goals. Highlights:
* From-scratch implementations, including BERT, GPT-2, Llama 2-3.2, DDPM, Real-NVP.
* Post-training experiments, including (LoRA) fine-tuning and DPO ([dl/17_nlp/post_training.ipynb](dl/17_nlp/post_training.ipynb)).
* Data and model parallelism, with and without JAX (+FLAX) ([../llms/performance/](../llms/performance/)).
* Experiments with TensorRT-LLM for model serving.
* Basic PPO ([../llms/rl/code.ipynb](../llms/rl/code.ipynb)).
