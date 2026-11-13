# Darcy generator verification on Kaggle CUDA

## Reproduce the run

Import [the verification notebook](../kaggle/verify_generators.ipynb) into Kaggle, select an NVIDIA Tesla T4 accelerator, and enable internet access. Run all cells. The notebook checks out commit `1f8a69e8a7c0772bf95bab4ed9022f1a7c8510d1`, based on GitHub main at `dce6ec2363a161a2c0b67817c86520c234c00b01`. The verifier adds diagnostics to the existing comparisons. Both generator implementations and all sparse reference definitions come from main.

The notebook installs `requirements.txt` in a fresh virtual environment with the audit's PyTorch, torchvision, Warp, NumPy and SciPy versions. The first Kaggle attempt failed during import because the image's optional RAPIDS stack loaded a Numba build incompatible with the resolved NumPy. The second stopped during environment creation because the image's `ensurepip` failed. The third attempt passed CUDA preflight but inherited the notebook's inline Matplotlib backend, which was unavailable in the isolated environment. The launcher bootstraps the virtual environment with the image's pip and selects the headless Agg backend for the verifier. The PhysicsNeMo pin is `b45a5c810c741e6b41f8515be24c51121f8fc21f`. The verifier checks CUDA availability and runs:

```sh
python scripts/verify_generator.py --device cuda --out-root /kaggle/working/generator-check-cuda
```

The notebook ends after packaging the report, figure, execution log, driver information and package list in `/kaggle/working/physicsnemo-fno-darcy-generator-check-cuda.zip`. The Kaggle submission used a 30-minute time limit.

## Local regression checks

The full local test suite passed (62 tests). A fresh CPU verification at 64, 128 and 256 matched every scientific value in the tracked CPU JSON exactly. The tracked CPU report and figure were left intact. The uploaded notebook contains only setup and generator verification.

## Measured results

**PASS** for the CUDA verification procedure. The unmodified NVIDIA sampler reproduced the CPU diversity defect. The independent sampler produced distinct fields on the same solver.

The verification ran on a Tesla T4 (GPU 0) for **22.07 s**, including the CPU sparse reference solves and figure generation. The notebook reported **342.9 s** including setup. The environment was Linux x86-64, Python 3.13.16, NVIDIA driver 580.178.04, CUDA 13.0, PyTorch 2.14.1+cu130, Warp 1.18.0, PhysicsNeMo 2.3.0a0, NumPy 2.5.3 and SciPy 1.18.1. [Kaggle run, version 4](https://www.kaggle.com/code/adebanjiadelowo/fno-darcy-cuda-generator-verification?version=4)

| Generator | Distinct fields in each batch of 16 | Distinct coefficient values in each batch | Distinct fields across both batches |
|---|---:|---:|---:|
| `Darcy2D` | 1, 1 | 4 of 1600 | 2 |
| `IndependentDarcy2D` | 16, 16 | 1600 of 1600 | 32 |

At resolution 32, fresh instances of both generators regenerated pressure and permeability exactly with NumPy seed 0. Seed 1 changed the permeability. Both outputs had shape `[16, 1, 32, 32]`, dtype `float32`, and zero NaNs and infinities. Permeability took values 0.5 and 2.0.

The table reports the independent sampler's mean relative L2 difference across four samples per resolution and tolerance. The final update is the upstream maximum over the finest grid and batch.

| Grid | Tolerance | NVIDIA stencil | Conservative Darcy | kΔu = −1 | Final max update |
|---:|---:|---:|---:|---:|---:|
| 64 | 1e-6 | 0.4163% | 31.0479% | 1.9861% | 8.64e-7 |
| 64 | 1e-7 | 0.04545% | 31.4907% | 1.6331% | 9.69e-8 |
| 128 | 1e-6 | 0.2700% | 33.3539% | 0.7452% | 2.46e-7 |
| 128 | 1e-7 | 0.1629% | 32.9907% | 1.0160% | 9.69e-8 |
| 256 | 1e-6 | 1.3292% | 35.3383% | 1.0541% | 2.42e-7 |
| 256 | 1e-7 | 0.1679% | 34.0302% | 0.5025% | 6.71e-8 |

All full-grid fields were finite and all six finest-grid updates met their tolerances. On nodes adjacent to the zero exterior Dirichlet values, the pooled relative L2 differences from the NVIDIA stencil solution were 0.0332%, 0.1201% and 0.1825% at tolerance 1e-7. The CPU and CUDA diversity counts were identical. All reported relative L2 values and residual RMS values agreed within 3e-14 in absolute units. This comparison concerns the reported metrics. Exact replay was measured within a single device.

At 256 × 256, tightening the solver reduced the NVIDIA-stencil error from 1.33% to 0.168%. The conservative reference still differed by 34.0%. The generator is suitable for studying the pinned discrete operator within the measured scope. A larger generator-only sample at 256 × 256 using the study batch size would quantify the tail of the stencil error and the frequency of unmet stopping tolerances before training. Training feasibility and memory require a separate measurement.

## Evidence

- [Raw CUDA metrics](../results/generator_check_cuda.json) and [pressure comparison figure](../figures/generator_check_cuda.png)
- [Kaggle execution log](../results/cuda_verification/fno-darcy-cuda-generator-verification.log), [verifier log](../results/cuda_verification/verification.log), [driver and GPU inventory](../results/cuda_verification/nvidia-smi.txt), and [package versions](../results/cuda_verification/pip-freeze.txt)
- [Original Kaggle archive](../results/cuda_verification/physicsnemo-fno-darcy-generator-check-cuda.zip), SHA-256 `9dccca7a072ee9d6347028e1084e7d6b45afcb2c3a0239c6896eccd1cff82450`
