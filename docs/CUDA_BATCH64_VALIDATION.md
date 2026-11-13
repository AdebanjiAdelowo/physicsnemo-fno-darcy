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
