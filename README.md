# nanoGPT-optimizer-benchmark
Benchmarking AdamW, SophiaG and AdamSNSM optimizers for training NanoGPT under resource‑constrained conditions (single GPU with 8GB VRAM). Includes reproducible training pipeline, experimental results and engineering practices for memory‑efficient LLM training.
# NanoGPT Optimizer Benchmark

We present a systematic comparison of three optimization algorithms—AdamW, SophiaG, and AdamSNSM—for training a medium‑sized GPT model (NanoGPT) under resource‑constrained conditions. All experiments are conducted on a single NVIDIA RTX 4060 (8GB VRAM) using the OpenWebText dataset. Our goal is to evaluate the convergence, generalization, memory footprint, and practical usability of each optimizer in a realistic setting where hardware resources are limited.

This repository contains the complete training pipeline, experiment configurations, raw results, and a detailed analysis report. The code is built upon Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) and extends it with implementations of two recent optimizers:

- **SophiaG** – A scalable stochastic second‑order optimizer that uses diagonal Hessian approximations (Liu et al., ICLR 2024).  
- **AdamSNSM** – A memory‑efficient variant that combines Subset‑Norm and Subspace‑Momentum to reduce optimizer state memory (Amanova et al., ICML 2024).

## Key Findings

- **AdamW** provides the most stable and reliable baseline, achieving a final validation loss of 3.42 with consistent training dynamics.
- **AdamSNSM** delivers almost identical performance (validation loss 3.46) while using only 10–20% of the optimizer memory required by AdamW. It also shows the best generalization (lowest overfitting degree).
- **SophiaG** did not outperform AdamW under our settings, revealing its sensitivity to hyperparameters and model scale. This highlights that theoretical advantages do not always transfer directly to moderate‑sized models with limited data.

In addition, we share a collection of engineering practices that proved essential for training on a single 8GB GPU, including:

- Multi‑threaded resume‑able dataset download (hfd + aria2)
- Gradient accumulation and memory‑efficient configurations
- Mixed precision training (AMP) and periodic CUDA cache clearing
- Two‑stage hyperparameter tuning (small‑subset screening followed by full validation)
