# Full-resolution CUDA Darcy generator validation

## Protocol

The experiment uses the `full` configuration's `IndependentDarcy2D` at 256 × 256, batch size 64, tolerance 1e-7 and 30,000 maximum Jacobi iterations per level. Four fresh generators use NumPy seeds 1000 through 1003. The first batch includes Warp kernel compilation and warm-up. A fresh generator repeats seed 1000 for exact same-device replay. The replay is excluded from reference error statistics.

The upstream finest-grid check takes the maximum absolute difference between the last two Jacobi iterates over the entire batch. The verifier measures that same quantity separately for every sample and checks that its maximum equals the upstream value. The CPU sparse solves use both the pinned NVIDIA stencil and the conservative harmonic-face discretization at the original 257 × 257 solver grid. Boundary error is relative L2 on the perimeter of that grid, adjacent to the zero exterior nodes. Error summaries use population standard deviation and linear-interpolated empirical percentiles.

Generation timing includes the upstream solver's host checks and synchronization. The script synchronizes CUDA before and after measurements. It records CPU sparse-solve time separately. GPU memory is sampled with `nvidia-smi` every approximately 100 ms, covers Torch and Warp allocations, and provides a lower bound on the device peak. GPU context is initialized before the memory baseline.

## Reproduce

Import [the launcher](../kaggle/verify_cuda_batch64.ipynb) into Kaggle, select an NVIDIA Tesla T4 accelerator, turn on internet access, and run all cells. The launcher uses commit `4fcab71e8e253d39b6dd0728e472d2f719424882` on top of the prior CUDA verification branch. It installs the pinned PhysicsNeMo source and the same PyTorch, Warp, NumPy and SciPy versions as the previous T4 run. The Kaggle submission timeout is 25 minutes. CPU reference solves use one BLAS thread.

The underlying command is:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_cuda_batch64.py --out /kaggle/working/batch64-measurements
```

The script saves each full batch of fields before sparse analysis. If the analysis is interrupted, copy the partial output to a CPU machine and run `python scripts/verify_cuda_batch64.py --out PATH --analyze-saved` to analyze the saved batches. The successful Kaggle archive retains the measurements, figure and logs. Raw fields are scratch data and are only included in the archive when the verification exits incomplete.

## Results

**PASS.** The Tesla T4 generated 256 distinct permeability fields in four batches. A fifth batch replayed seed 1000 exactly for pressure and permeability. The other seeds each produced 64 new permeability fields. All outputs were finite, with shape `[64, 1, 256, 256]` and dtype `float32`. Permeability took only 0.5 and 2.0. Across the full solver grids, their observed proportions were approximately 50.1% and 49.9%. Generation and synchronization completed without errors. [Kaggle run](https://www.kaggle.com/code/adebanjiadelowo/fno-darcy-cuda-batch-64-verification)

| Relative L2 error | Mean | Median | Std. dev. | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA stencil | 0.1643% | 0.1407% | 0.0853% | 0.3451% | 0.4156% | 0.4253% |
| Boundary-adjacent NVIDIA stencil | 0.1898% | 0.1752% | 0.0788% | 0.3547% | 0.4219% | 0.4842% |
| Conservative Darcy | 28.5722% | 27.8041% | 9.5299% | 46.8495% | 51.1765% | 57.3113% |

All four analyzed batches satisfied the upstream stopping criterion. Zero of 256 samples had a final maximum update at or above 1e-7. Per-sample final updates had mean 7.07e-8, median 6.71e-8, population standard deviation 9.91e-9, P95 8.94e-8, P99 9.69e-8 and maximum 9.69e-8. The replay also met the batch tolerance. Sparse direct comparisons independently quantify the remaining pressure error.

The first constructor took 15.39 ms, and the next three took 1.48–1.50 ms. The first generation, including warm-up, took 3.756 s. Subsequent independent batches took 1.994–2.021 s, or 31.67–32.10 samples/s. The sampled GPU memory peak was 201 MiB, 198 MiB above baseline. This whole-device measurement can miss short-lived peaks and does not include model training. CPU reference solves took 221.62 s. The verifier took 235.21 s, and the notebook took 423.79 s including setup.

The run used NVIDIA driver 580.178.04, CUDA 13.0, PyTorch 2.14.1+cu130, Warp 1.18.0, PhysicsNeMo 2.3.0a0 at commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`, Python 3.13.16, NumPy 2.5.3 and SciPy 1.18.1.

**CONDITIONAL GO** for a short CUDA FNO training and memory probe targeting the pinned NVIDIA stencil. Use the observed label error when interpreting sub-percent FNO errors, and measure training memory directly. The conservative discrepancy persists across the larger sample, with a 28.57% mean versus 34.03% in the earlier four-field run. The 256 samples come from four batches on one GPU, so the empirical tail and zero observed misses cannot establish rates for rare failures. This experiment used one batch per seed and performed no FNO training.

## Evidence

- [JSON report](../results/cuda_batch64_validation/measurements/report.json), [per-sample CSV](../results/cuda_batch64_validation/measurements/per_sample.csv), [memory trace](../results/cuda_batch64_validation/measurements/gpu_memory.csv), and [error distributions](../results/cuda_batch64_validation/measurements/error_distributions.png)
- [Original archive](../results/cuda_batch64_validation/cuda_batch64_validation.zip), SHA-256 `cda85fd3dc8b7692c81e661a0259271d27b9e039a8485f5f8027769b944d934a`
- Local tests passed (65 total). The earlier CPU and CUDA result files, figures, and archive are byte-identical to their versions on the base branch.
