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
