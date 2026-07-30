# Write-Up Backlog

Things already read that still need to be written up as notes in `llms/`, ordered by folder. Check items off (`- [x]`) as the notes actually land in the target file — the `todo-check` skill cross-references this list on every push.

## architecture/

- [ ] [The Big LLM Architecture Comparison (Raschka)](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) → expand [llm_architectures.md](architecture/llm_architectures.md) and [attention.md](architecture/attention.md)
- [ ] [Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention (Raschka)](https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures) — Gemma 4, DeepSeek V4, cross-layer KV sharing, compressed attention → [llm_architectures.md](architecture/llm_architectures.md); also covers the mHC item below
- [ ] [A Visual Guide to Attention Variants in Modern LLMs (Raschka)](https://magazine.sebastianraschka.com/p/visual-attention-variants) — MHA → GQA/MLA, sparse attention, hybrid linear-attention stacks → [attention.md](architecture/attention.md)
- [ ] Gated DeltaNet deep-dive (Qwen3-Next) → expand the linear-attention section of [attention.md](architecture/attention.md)
- [ ] [Scaling Latent Reasoning via Looped Language Models (Ouro, arXiv 2510.25741)](https://arxiv.org/pdf/2510.25741) — looped transformers / latent recurrence → [llm_architectures.md](architecture/llm_architectures.md)
- [ ] [Qwen3 from scratch (Raschka)](https://magazine.sebastianraschka.com/p/qwen3-from-scratch) → [llm_architectures.md](architecture/llm_architectures.md)
- [ ] [The technical DeepSeek deep-dive (Raschka)](https://magazine.sebastianraschka.com/p/technical-deepseek) → MLA/sparse-attention detail in [attention.md](architecture/attention.md)
- [ ] [Understanding Multimodal LLMs (Raschka)](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) → expand [multimodal_llms.md](architecture/multimodal_llms.md)
- [ ] mHC paper (DeepSeek — manifold-constrained hyper-connections; find/link the paper) → expand the mHC section in [llm_architectures.md](architecture/llm_architectures.md)
- [ ] *(to read)* MoE canon — no MoE note exists despite MoE appearing in every model entry: [Switch Transformer (arXiv 2101.03961)](https://arxiv.org/abs/2101.03961), [DeepSeekMoE (arXiv 2401.06066)](https://arxiv.org/abs/2401.06066), [aux-loss-free load balancing (arXiv 2408.15664)](https://arxiv.org/abs/2408.15664) → new architecture/moe.md
- [ ] *(to read)* SSM lineage behind DeltaNet: [Mamba (arXiv 2312.00752)](https://arxiv.org/abs/2312.00752), [Mamba-2 / SSD (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060), + the hybrid attention+SSM interleaving pattern → [attention.md](architecture/attention.md)
- [ ] *(to read)* [YaRN (arXiv 2309.00071)](https://arxiv.org/abs/2309.00071) — the paper behind the queued OLMo 3 mention → [attention.md](architecture/attention.md)

## optimization/

*Scope: what happens in weight space — optimizers, training dynamics, forgetting, scaling laws.*

- [ ] Muon optimizer: [Keller Jordan's post](https://kellerjordan.github.io/posts/muon/), [willccbb thread](https://x.com/willccbb/status/2050038277454143918?lang=en) → [notes.md](optimization/notes.md)
- [ ] [Can Muon Fine-tune Adam-Pretrained Models? (arXiv 2605.10468)](https://arxiv.org/pdf/2605.10468) — optimizer-mismatch problem
- [ ] [Optimizer-Model Consistency: full finetuning with the pretraining optimizer forgets less (arXiv 2605.06654)](https://arxiv.org/pdf/2605.06654)
- [ ] [RL's Razor: Why Online RL Forgets Less (arXiv 2509.04259)](https://arxiv.org/pdf/2509.04259) — RL implicitly favors KL-minimal solutions (bridges to rl/ and post_training/)
- [ ] [Scaling Laws, Carefully (Lilian Weng, 2026-06)](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/) — pretraining compute/data power laws *(you listed this under post-training; it's classic scaling laws, filed here)*
- [ ] NTK (neural tangent kernel) → [notes.md](optimization/notes.md)
- [ ] muP / hyperparameter transfer → [notes.md](optimization/notes.md)

## data/

- [ ] Pretraining data curation, mixtures, deduplication → [recipe.md](data/recipe.md). (Checked 2026-07: nothing on this exists in the repo yet — the links you remember are not in here; re-locate sources.)
- [ ] [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388) → [recipe.md](data/recipe.md)
- [ ] [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631) → [recipe.md](data/recipe.md) *(VLM side also feeds [multimodal_llms.md](architecture/multimodal_llms.md); source for the M-RoPE item above)*
- [ ] [Kimi K3 Technical Report (MoonshotAI GitHub, PDF)](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) → [recipe.md](data/recipe.md) *(eval sections separately queued in evals/)*
- [ ] [GLM-5: from Vibe Coding to Agentic Engineering (arXiv 2602.15763)](https://arxiv.org/abs/2602.15763) → [recipe.md](data/recipe.md)
- [ ] [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv 2606.19348)](https://arxiv.org/abs/2606.19348) → [recipe.md](data/recipe.md) *(architecture side — CSA/HCA, mHC — also feeds architecture/)*
- [ ] [DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) → [recipe.md](data/recipe.md) *(also queued in rl/ as the GRPO/RLVR origin)*
- [ ] [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) → [recipe.md](data/recipe.md) *(infra §also queued in performance/)*
- [ ] *(to read)* Curation canon: [FineWeb / FineWeb-Edu (arXiv 2406.17557)](https://arxiv.org/abs/2406.17557), [DCLM (arXiv 2406.11794)](https://arxiv.org/abs/2406.11794), [Dolma (arXiv 2402.00159)](https://arxiv.org/abs/2402.00159)
- [ ] *(to read)* [Deduplicating Training Data Makes LMs Better (arXiv 2107.06499)](https://arxiv.org/abs/2107.06499); mixtures: [DoReMi (arXiv 2305.10429)](https://arxiv.org/abs/2305.10429)
- [ ] *(to read)* Synthetic pretraining data: [Textbooks Are All You Need / Phi (arXiv 2306.11644)](https://arxiv.org/abs/2306.11644), [Cosmopedia (HF)](https://huggingface.co/blog/cosmopedia); [data-constrained scaling (arXiv 2305.16264)](https://arxiv.org/abs/2305.16264) (double-files with optimization scaling laws)

## post_training/

*Scope: what loss on what data — SFT, distillation family, preference optimization, reward modeling. KL estimators (k1/k2/k3) stay canonical in [rl/kl_divergence.md](rl/kl_divergence.md).*

- [ ] [LoRA Without Regret (Thinking Machines)](https://thinkingmachines.ai/blog/lora/) → PEFT section of [notes.md](post_training/notes.md)
- [ ] [On-Policy Distillation (Thinking Machines)](https://thinkingmachines.ai/blog/on-policy-distillation/) → write the OPD stub in [notes.md](post_training/notes.md)
- [ ] OPD variants: OPSD, g-OPD, v-OPD → same section
- [ ] [SFT, RL, and On-Policy Distillation Through a Distributional Lens (nrehiew)](https://nrehiew.github.io/blog/sft_rl_opd/) — how each objective reshapes the distribution / forgetting (cross-ref optimization/ RL's Razor)
- [ ] Deepen RLHF/DPO sections; add k2 = ½(log r)² alongside k1/k3 in [rl/kl_divergence.md](rl/kl_divergence.md)
- [ ] Check DeepSeek V3's reward pipeline for general (non-reasoning) data → §Reward Modeling in [notes.md](post_training/notes.md)
- [ ] *(to read)* [LIMA (arXiv 2305.11206)](https://arxiv.org/abs/2305.11206) — SFT data quality over quantity → SFT section
- [ ] *(to read)* [Tülu 3 (arXiv 2411.15124)](https://arxiv.org/abs/2411.15124) — the open end-to-end post-training recipe; origin of "RLVR" as a named stage → recipe overview + cross-ref rl/

## rl/

*Scope: learning from your own samples — policy-gradient machinery, KL estimation, async/staleness infrastructure, verifiers.*

- [ ] [Predicting and Controlling Staleness in Fully Asynchronous RL (Applied Compute)](https://www.appliedcompute.com/research/staleness-in-fully-async-rl) → new async-RL section
- [ ] [Is Frontier Asynchronous RL Solved? (Luk Huang)](https://luk-huang.github.io/personal-website/blog/is-frontier-asynchronous-rl-solved.html) → same section
- [ ] [Single-Rollout Asynchronous Optimization for Agentic RL (arXiv 2607.07508)](https://arxiv.org/abs/2607.07508) *(you listed this under post-training; it's async agentic RL, filed here)*
- [ ] [CompactionRL: RL with Context Compaction for Long-Horizon Agents (arXiv 2607.05378)](https://arxiv.org/abs/2607.05378) — cross-ref [agents/harnesses.md](agents/harnesses.md) context management
- [ ] [Spinning Up (OpenAI)](https://spinningup.openai.com/en/latest/) — policy gradient theorem, REINFORCE, VPG, TRPO, PPO → [policy_gradients.md](rl/policy_gradients.md)
- [ ] [Coding PPO from scratch, parts 1–4 (Medium)](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) → [policy_gradients.md](rl/policy_gradients.md) / `code.ipynb`
- [ ] [The N Implementation Details of RLHF with PPO (HF)](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) — resolve the "(verify against source)" items in [rlhf_ppo.md](rl/rlhf_ppo.md)
- [ ] [Approximating KL Divergence (Schulman)](http://joschu.net/blog/kl-approx.html) — flesh out estimator derivations in [kl_divergence.md](rl/kl_divergence.md)
- [ ] *(to read)* [Let's Verify Step by Step (arXiv 2305.20050)](https://arxiv.org/abs/2305.20050) — PRMs, outcome vs process supervision → [reasoning.md](rl/reasoning.md)
- [ ] *(to read)* RL rollout infrastructure (bridge to performance/): [verl/HybridFlow (arXiv 2409.19256)](https://arxiv.org/abs/2409.19256), [OpenRLHF (arXiv 2405.11143)](https://arxiv.org/abs/2405.11143) — vLLM-in-the-loop, weight sync, train/inference numerical mismatch → new async/infra section
- [ ] *(to read)* [GAE paper (arXiv 1506.02438)](https://arxiv.org/abs/1506.02438) — the source for the λ notes in [rlhf_ppo.md](rl/rlhf_ppo.md)
- [ ] *(to read)* [DeepSeekMath (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300) — where GRPO comes from (grpo.md currently cites nothing) + [DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)
- [ ] *(to read)* [Scaling LLM Test-Time Compute Optimally (arXiv 2408.03314)](https://arxiv.org/abs/2408.03314) → [post_training/notes.md](post_training/notes.md) §Inference-time scaling

## evals/

- [ ] [LLM evaluation: 4 approaches (Raschka)](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches) → expand [llm_evals.md](evals/llm_evals.md)
- [ ] DeepSeekMath V2 meta-verifiers, from the evals angle → [llm_evals.md](evals/llm_evals.md) (section moved from rl/reasoning.md 2026-07-28)
- [ ] Model-report eval sections: Kimi K3 report, Cursor Composer 2, GLM 5.2 → [llm_evals.md](evals/llm_evals.md)

## computer_use/

- [ ] [CUA-Gym (arXiv 2605.25624)](https://arxiv.org/pdf/2605.25624) → data section of [notes.md](computer_use/notes.md)
- [ ] Benchmarks: Online-Mind2Web, Weave Bench, OSWorld, OSWorld-V2 → [notes.md](computer_use/notes.md)
- [ ] External harnesses + action spaces: Anthropic, OpenAI, Gemini, Muse Spark, Yutori, Compaction → [notes.md](computer_use/notes.md)
- [ ] *(to read)* UI grounding: [UGround (arXiv 2410.05243)](https://arxiv.org/abs/2410.05243), [OS-Atlas (arXiv 2410.23218)](https://arxiv.org/abs/2410.23218); [Set-of-Marks prompting (arXiv 2310.11441)](https://arxiv.org/abs/2310.11441)
- [ ] *(to read)* [UI-TARS (arXiv 2501.12326)](https://arxiv.org/abs/2501.12326); [WebArena (arXiv 2307.13854)](https://arxiv.org/abs/2307.13854) — the pre-OSWorld canon

## performance/

- [ ] [How To Scale Your Model (JAX scaling book)](https://jax-ml.github.io/scaling-book/index) — remaining chapters (ch. 1–2 seeded in [tpus.md](performance/tpus.md))
- [ ] [vLLM deep-dive (Gordić)](https://www.aleksagordic.com/blog/vllm) → expand vLLM internals in [inference.md](performance/inference.md), re-add benchmark figure
- [ ] [matmul deep-dive (Gordić)](https://www.aleksagordic.com/blog/matmul) → expand [gpus.md](performance/gpus.md)
- [ ] Write a tensor-parallelism notebook (the old empty one was removed)
- [ ] *(to read)* [The Ultra-Scale Playbook (HuggingFace)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — GPU/PyTorch complement to the JAX scaling book; covers Megatron-style parallelism (subsumes the earlier "Megatron notes" item) → [parallelism.md](performance/parallelism.md) + [basics.md](performance/basics.md)
- [ ] *(to read)* [ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054) — FSDP's ancestor, currently uncited in [parallelism.md](performance/parallelism.md)
- [ ] *(to read)* [Ring Attention (arXiv 2310.01889)](https://arxiv.org/abs/2310.01889) — sequence/context parallelism, an entirely missing parallelism axis
- [ ] *(to read)* Expert parallelism for MoE (DeepSeek-V3 report, [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) §infra; incl. its FP8 training recipe) → [parallelism.md](performance/parallelism.md) / [basics.md](performance/basics.md)
- [ ] *(to read)* Inference quantization: [AWQ (arXiv 2306.00978)](https://arxiv.org/abs/2306.00978), [GPTQ (arXiv 2210.17323)](https://arxiv.org/abs/2210.17323) → [inference.md](performance/inference.md)
