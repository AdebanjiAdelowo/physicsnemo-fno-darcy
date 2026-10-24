# Comparison with the NVIDIA example

Upstream: [`examples/cfd/darcy_fno`](https://github.com/NVIDIA/physicsnemo/tree/b45a5c810c741e6b41f8515be24c51121f8fc21f/examples/cfd/darcy_fno) at commit `b45a5c810c741e6b41f8515be24c51121f8fc21f` (`train_fno_darcy.py`, `config.yaml`, `validator.py`), read from a local clone at that commit. The installed package is the same commit (`study_metadata.json`, `physicsnemo.commit`).

**Same** means identical code or values. **Changed** means a deliberate difference of this repository.

## Model

| Item | Upstream | This repository | |
|---|---|---|---|
| Class | `physicsnemo.models.fno.FNO` | the same, from the installed package | Same |
| Arguments | in 1, out 1, decoder 1 layer of 32, dimension 2, latent 32, 4 FNO layers, modes 12, padding 9 | the same keys and values (`configs/official.yaml` above its marker); `num_fno_modes` varied over 4, 8, 12, 16 | Same, except the studied factor |
| Spectral layer | `SpectralConv2d`: `rfft2`, two complex blocks for rows `[:m]` and `[-m:]`, columns `[:m]` | the same | Same |
| Padding | constant zero, right and bottom, 9 cells | the same | Same |
| Coordinate features | on (two channels, `linspace(0, 1)`) | the same | Same |
| Activation | GELU | the same | Same |

Parameter counts, from the instantiated models: 268 017, 1 054 449, 2 365 169 and 4 200 177 for 4, 8, 12 and 16 modes. They equal $5873 + 4 \cdot 2 \cdot 32^2 \cdot m^2 \cdot 2$, and the spectral weights are 97.8% to 99.9% of the parameters. **VERIFIED** (`scripts/audit_checkpoints.py`).

**Mode compatibility.** The padded grid has 73 points at 64 × 64 and 265 at 256 × 256. The two row blocks are disjoint for $m \le \lfloor n/2 \rfloor$, which is 36 and 132. Modes 4 to 16 are far inside this bound on both grids. Up to commit `5c81978` the study validator accepted $m = \lfloor n/2 \rfloor + 1$, which silently passes the middle row through both blocks. This audit fixed the bound, with a regression test. No result was affected.

## Data

| Item | Upstream | This repository (`local`) | |
|---|---|---|---|
| Generator | `Darcy2D` | `IndependentDarcy2D`: `Darcy2D` with the coefficient draw replaced | Changed |
| Distinct fields per batch | 1 (CPU; see [GENERATOR_VALIDATION.md](GENERATOR_VALIDATION.md)) | all | Changed |
| Synthesis, threshold, solver kernels | upstream | the same upstream kernels | Same |
| Solver tolerance | $10^{-6}$ | $10^{-7}$ | Changed |
| Resolution | 256 | 64 | Changed (reduced study) |
| Training data | streamed, a new batch per step; 524 288 samples seen, 8192 distinct fields | finite and shared: 1024 fields, 30 passes | Changed |
| Validation | 4 fresh batches (256 samples, 4 distinct fields) per validation step, from the same stream | fixed set of 128 | Changed |
| Test set | none | fixed set of 256, seed 3000 | Added |
| Warp device | default `"cuda"` | `cpu` (Warp has no MPS backend) | Changed |
| Normalisation constants | permeability $1.25 \pm 0.75$, pressure $0.0452 \pm 0.0279$ | the same | Same |

## Training

| Item | Upstream | This repository (`local`) | |
|---|---|---|---|
| Loss | `MSELoss(reduction="mean")` of the normalised pressure | the same | Same |
| Optimiser | Adam, learning rate $10^{-3}$ | the same | Same |
| Schedule | `LambdaLR`, factor $0.85^{s}$, stepped every 8 pseudo-epochs | the same mechanism, stepped every 3 | Changed |
| Pseudo-epochs | 256 of 2048 samples | 30 of 1024 samples | Changed |
| Batch size | 64 | 32 | Changed |
| Optimiser steps | 8192 | 960 | Changed |
| Learning rate in the last pseudo-epoch | $1.0\times10^{-3}\cdot 0.85^{31} = 6.5\times10^{-6}$ | $0.85^{9}\cdot10^{-3} = 2.3\times10^{-4}$ | Changed |
| Forward wrappers | `StaticCaptureTraining`, `StaticCaptureEvaluateNoGrad`, AMP and CUDA graphs off | the same | Same |
| Checkpoints | `save_checkpoint` every 8 pseudo-epochs and at the end, resumption | every 10 and at the end, no resumption | Changed |
| Seeding | none | NumPy and PyTorch seeded per run; model initialised on CPU | Added |
| Logging | `LaunchLogger`, MLFlow, validation figure | CSV and JSON | Changed |

The learning rate in the last pseudo-epoch is about 35 times higher in the reduced study than upstream. Upstream, the final model is taken from a nearly annealed optimiser. In the reduced study it is taken while the validation error still fluctuates by about 0.04 percentage points from one pseudo-epoch to the next (see [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md)).

## Evaluation

| Item | Upstream | This repository | |
|---|---|---|---|
| Metric | validation MSE in normalised units (`GridValidator.compare`) | per-sample relative $L^2$ error in physical units on the fixed test set, and the same MSE | Added |
| Normalisation inversion | in the validator plot only | `denormalise` of prediction and target before the error; checked independently in NumPy (`scripts/audit_checkpoints.py`) | Added, VERIFIED |
| Timings | none | synchronised optimisation, data and inference times | Added |

## Fairness of the comparison between Fourier-mode settings

Within the study, every setting uses the same data, the same splits, the same seeds 0 to 2 (initialisation and batch order), the same schedule and the same number of optimiser steps. The number of modes is the only difference. **VERIFIED** (configuration, code, and `optimiser_steps = 960` in every record).

What this does not equalise:

- **Parameters.** 16 modes has 15.7 times the parameters of 4 modes. Each setting is a different capacity, as intended.
- **Tuning.** One learning rate and one schedule for all settings. A larger model may need a different learning rate or more steps. The comparison is at equal training budget, not at each setting's best.
- **Convergence.** In 12 of 12 runs the validation error is still falling, on average, over the last five pseudo-epochs.

## Upstream results

The upstream example publishes no accuracy number for this configuration that this repository could compare with. `configs/official.yaml` reproduces the upstream settings for a future CUDA run (pending).
