# CUDA FNO training feasibility probe

The probe targets the pinned NVIDIA PhysicsNeMo Darcy discrete operator. The prior 256 × 256 generator audit measured a mean stencil-relative error of 0.1643%, P99 of 0.4156%, and mean conservative Darcy discrepancy of 28.57%. These reference-label differences bound how the training results should be interpreted. [Generator validation](CUDA_BATCH64_VALIDATION.md)

## Protocol

The probe uses `configs/full.yaml` with 16 Fourier modes, seed 0, one pseudo-epoch and validation after that epoch. The resolution is 256 × 256 and both training and generation batch sizes are 64. There are 4,096 fixed training fields, 256 fixed validation fields, and up to 64 optimizer updates. The validated independent sampler uses tolerance 1e-7. The model, Adam settings, loss, scheduler, data ordering and validation metric come from the existing training engine. The probe does not generate a test split because this experiment ends at validation.

The verifier loads or generates splits through the existing cache API in fresh scratch space, checks field shape, dtype and finiteness, and compares permeability hashes across splits. It records host high-water RSS, dataset sizes and preparation timing. Training uses the existing PhysicsNeMo capture wrapper with AMP and CUDA graphs disabled. The loss backward hook and Adam pre/post hooks mark synchronized phase boundaries. The gradient norm is observed between backward and Adam without clipping. Synchronization, memory queries and gradient diagnostics add overhead, so measured times represent the instrumented run. Whole-device memory is sampled via NVIDIA SMI and may miss short-lived peaks. PyTorch allocator peaks capture device allocations under its allocator, while the device trace also includes contexts and Warp.

If the epoch completes, the projection uses the observed update epoch after subtracting validation, then adds 30 measured-duration validations and seven checkpoint writes per 60-epoch run. It applies the 16-mode rate as a planning proxy for 12 runs, with a separate 25% margin. Dataset preparation is charged separately and can be amortized when reusing the same fixed splits. The projection does not benchmark the other mode counts.

## Reproduction

Import [the probe notebook](../kaggle/probe_cuda_fno_training.ipynb) into Kaggle, select a Tesla T4 and enable internet. It checks out `e2faadffb561009c3a897f88f9ce835ceb74ac31` based on `f799cd12d0a7d1e73aaf75aef5da41c44aa4ae18`, which includes PRs #1, #2 and #3. It installs PhysicsNeMo from the pinned source and matches the earlier CUDA audit's PyTorch, Warp, NumPy and SciPy versions. It uses a 1,200 s notebook budget with a 1,500 s Kaggle cap. The runner writes measurements outside the checkout and archives available evidence on return or timeout. A fresh session avoids cache provenance ambiguity.

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/probe_cuda_fno_training.py --out /kaggle/working/cuda_training_probe
```

## Measured results

The Tesla T4 completed all 64 optimizer updates without CUDA OOM. The run used NVIDIA driver 580.178.04, CUDA 13.0, PyTorch 2.14.1+cu130, Warp 1.18.0, PhysicsNeMo 2.3.0a0 at source commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`, Python 3.13.16, NumPy 2.5.3 and SciPy 1.18.1. PyTorch reported 14,911.69 MiB of usable device capacity. [Kaggle run](https://www.kaggle.com/code/adebanjiadelowo/fno-darcy-t4-training-probe)

| Dataset | Samples | Generation and reference diagnostics | Preparation through cache write | Cache size |
|---|---:|---:|---:|---:|
| Training, seed 1000 | 4,096 | 224.47 s | 230.61 s | 2.000 GiB |
| Validation, seed 2000 | 256 | 14.16 s | 14.49 s | 0.125 GiB |

All fields had the expected shape and dtype, were finite, and were distinct within each split. No permeability hash occurred in both splits. Host process high-water RSS was 6,641 MiB during preparation. The PyTorch allocator peaked at 16 MiB allocated and reserved per split. Warp allocations appear in the whole-device trace rather than the PyTorch peak. The training loop spent 0.979 s in batch preparation, including device transfers.

| Phase | Mean duration | PyTorch peak allocated | PyTorch peak reserved |
|---|---:|---:|---:|
| Forward and loss | 0.626 s/update | 9,420 MiB | 12,740 MiB |
| Backward | 0.870 s/update | 9,941 MiB | 12,740 MiB |
| Adam step | 0.00406 s/update | 1,152 MiB | 12,740 MiB |
| Validation | 3.320 s total | 5,598 MiB | 12,740 MiB |

The whole-device sampled peak was 12,911 MiB (12.61 GiB). The memory driver reported about 2,001 MiB free after validation. The legacy `training.peak_train_memory_mb` field in the raw report reflects allocator activity after a validation-phase reset in this instrumented run. Use the probe's `train_peak_allocated_mib` and phase-level CSV for training peaks.

The epoch took 100.524 s including validation, or 97.203 s excluding it. Checkpoint writing took 0.117 s. The first update took 2.239 s and updates 2–64 averaged 1.490 s. Across all 64 updates, mean update time was 1.502 s and epoch-average normalized MSE was 0.34856. The last update's loss was 0.01636. Every recorded gradient norm was finite, ranging from 0.0547 to 1.5134. Validation predictions and targets were finite. Validation normalized MSE was 0.01702 and mean physical-unit relative L2 was 6.989%. These are one-epoch stability observations. The script took 353.30 s and the notebook took 661.98 s including setup.

## Full-study estimate and decision

The nominal schedule contains 46,080 updates. Scaling the measured 16-mode epoch yields **1.648 h per 60-epoch run** and **19.78 h across 12 runs**, plus dataset preparation. Each projected run includes roughly 1.66 min of validation and 0.82 s of checkpoint writes. With a 25% margin the training projection is **24.72 h**. The measured dataset generation overhead was 238.64 s for the training and validation splits, plus cache writing and checks. Full-study test-set generation and final evaluation are additional costs outside this probe.

**CONDITIONAL GO.** The tested full-resolution batch fits a T4 and completes training and validation. Split the study into resumable sessions and budget using the remaining quota at launch. The projection uses one instrumented epoch, one seed and the largest mode count, so it does not establish runtime or peak memory for every run. Keep the conservative Darcy discrepancy and the measured generator label error in mind when interpreting scientific results.

## Evidence and checks

- [Report](../results/cuda_training_probe/report.json), [per-update metrics](../results/cuda_training_probe/training_metrics.csv), [GPU trace](../results/cuda_training_probe/gpu_memory.csv), [validation](../results/cuda_training_probe/validation.json), and [projection](../results/cuda_training_probe/runtime_projection.json)
- [Training loss figure](../results/cuda_training_probe/training_loss.png) and [original archive](../results/cuda_training_probe/cuda_training_probe.zip), SHA-256 `cf77b1e497862aae5e9e9fd9d781d42abb01cf858467a406eb4686c3346be59c`
- All 70 local tests passed. The archived CSV contains 64 consecutive updates, with finite loss and gradients and nonnegative times and memory values. Earlier generator evidence is byte-identical to main.
