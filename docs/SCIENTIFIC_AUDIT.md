# Scientific audit of the Fourier-mode study

Audit of commit `5c81978` (the state in which the reduced study was published), carried out on 24 October 2026 on an Apple M3 Pro (MPS for training, CPU for everything else). It establishes what the reduced 64 × 64 study demonstrates, checks every number it reports, and records two experiments added to test the robustness of its conclusions.

Companion documents: [UPSTREAM_COMPARISON.md](UPSTREAM_COMPARISON.md) (code and protocol against NVIDIA's example), [GENERATOR_VALIDATION.md](GENERATOR_VALIDATION.md) (what the data generator computes), [EXPERIMENT_GAPS.md](EXPERIMENT_GAPS.md) (what is still missing, with commands and resource estimates).

Classifications:

| Label | Meaning |
|---|---|
| **VERIFIED** | Reproduced by execution in this audit, or confirmed by a second, independent implementation |
| **SUPPORTED BUT LIMITED** | Consistent with the evidence, but the evidence is narrower than the statement |
| **UNVERIFIED** | Not tested; no evidence for or against |
| **INCORRECT** | Contradicted by the evidence |

## 1. Summary

1. **The saved numbers are correct.** The four mean test errors (1.458%, 0.920%, 0.979%, 0.970%) follow from the per-run records. They were recomputed from the checkpoints in NumPy on CPU, independently of the repository's metric code, with a largest relative difference of $2.8\times10^{-5}$ over the 12 runs. The parameter counts match the analytic count. **VERIFIED.**
2. **The data are sound for the comparison.** The datasets regenerate bitwise from their seeds, the splits contain no duplicate or near-duplicate fields (every test field differs from every training field in at least 30.8% of pixels), and the solver error (0.045%) is 20 times below the best model error. **VERIFIED.**
3. **The learned operator is NVIDIA's discrete generator, not the Darcy equation.** The generator's stencil is consistent with $-\Delta u = 1/k$, not with $-\nabla\cdot(k\nabla u) = 1$, and its output differs from a conservative Darcy solve by about 32%. **VERIFIED** with two independent assemblies and a refinement study.
4. **The repeated-permeability defect of `Darcy2D` is real on CPU and has a source-level cause that applies to CUDA.** As shipped, every batch holds one field from a three-parameter family. **VERIFIED on CPU; on CUDA SUPPORTED BUT LIMITED** (static evidence only).
5. **`IndependentDarcy2D` produces independent fields.** **VERIFIED.**
6. **4 modes are worse than 8, 12 and 16.** This holds in all 6 seeds of each setting (permutation $p = 0.002$). **VERIFIED** for this protocol.
7. **8 modes being best is not robust.** In three further seeds the order reverses (16 < 12 < 8). Over six seeds, 8, 12 and 16 modes are statistically indistinguishable (0.918%, 0.937%, 0.908%; all pairwise $p > 0.5$). **The statement "8 modes is optimal" would be INCORRECT. "Accuracy saturates from 8 modes at this budget" is SUPPORTED BUT LIMITED.**
8. **The reduced study is training-limited.** With three times the optimiser steps (same data, same final learning rate) the error of 8, 12 and 16 modes falls from about 0.97% to about 0.34%, with the three settings still within 0.02 percentage points (one seed each). The absolute error level of the reduced study is a property of its budget. **VERIFIED** (one seed per setting).
9. **One minor bug was found and fixed.** The validator's largest admissible number of modes was one too high: it admitted a setting where one Fourier row is counted twice. No configuration used it.

## 2. Scope of what the reduced study demonstrates

The reduced study is one experiment:

- one grid (64 × 64), one sampler (`IndependentDarcy2D`), one forcing, one permeability contrast (0.5/2);
- 1024 training samples, 960 optimiser steps, a learning rate still at $2.3\times10^{-4}$ in the last pseudo-epoch;
- one architecture (width 32, depth 4, padding 9) with only `num_fno_modes` varied;
- one device (Apple MPS), with Warp generating on CPU.

Within that scope it shows that 4 retained modes limit accuracy, and that 8, 12 and 16 modes reach the same error to within seed-to-seed variation. It does not show an optimal number of modes, a resolution-independent result, a property of the continuous Darcy operator, or anything about CUDA hardware.

A possible explanation, which this audit does not test: the permeability is the sign of a Fourier series with frequencies 0 to 4 per period, which correspond to indices up to about 4.6 on the padded 73-point FFT grid. 4 retained modes (indices 0 to 3) cannot represent the highest generating frequency in a single spectral layer, while 8 can. If this is the mechanism, the threshold depends on `nr_permeability_freq` and not on the grid.

## 3. Claim-by-claim verification

### 3.1 Numerical results (task 8)

Source: `results/local/modes*_seed*/eval_metrics.json`, recomputed by `scripts/audit_results.py` and `scripts/audit_checkpoints.py`.

| Modes | Claim | Recomputed from records | Recomputed from checkpoints (CPU, NumPy) | Status |
|---|---|---|---|---|
| 4 | 0.01458 | 0.014582 | 0.014582 | VERIFIED |
| 8 | 0.00920 | 0.009195 | 0.009195 | VERIFIED |
| 12 | 0.00979 | 0.009790 | 0.009790 | VERIFIED |
| 16 | 0.00970 | 0.009696 | 0.009696 | VERIFIED |

Per run, the largest relative difference between the stored error (computed on MPS) and the CPU recomputation is $2.8\times10^{-5}$ (`results/audit/checkpoint_recomputation.json`). Every checkpoint is the final one (pseudo-epoch 30).

### 3.2 Metric, normalisation, parameters, timing (task 9)

| Item | Finding | Status |
|---|---|---|
| Relative $L^2$ | per sample, $\lVert\hat u - u\rVert_2/\lVert u\rVert_2$ over the 64 × 64 grid, then averaged over the 256 test samples; matches an independent NumPy float64 computation | VERIFIED |
| Normalisation inversion | prediction and target are both denormalised with the configuration constants before the error | VERIFIED |
| Parameter counts | 268 017 / 1 054 449 / 2 365 169 / 4 200 177 $= 5873 + 16\,384\,m^2$; the spectral weights are 97.8% to 99.9% | VERIFIED |
| Training time | wall-clock, the device synchronised before every reading; optimisation and data time separated | VERIFIED (code) |
| "Time does not depend on the number of modes" | optimisation times 301 to 315 s for all settings on MPS at 64 × 64. The audit's repeat runs took 271 to 425 s per run for the same work, depending on machine load and temperature, so differences of less than about 30% are not resolvable on this laptop | SUPPORTED BUT LIMITED (one device, one size) |
| Inference timing | median of 30 synchronised forward passes of a batch of 32 after 5 warm-up passes, MPS | VERIFIED (code); values device-specific |

### 3.3 Fairness and mode compatibility (task 10)

All settings share data, splits, seeds, schedule and the number of steps (960 in every record). **VERIFIED.** No setting is tuned separately (SUPPORTED BUT LIMITED as a fair comparison: fair at equal budget, not at each setting's optimum).

On the padded grid of $n = 73$ points the spectral layer keeps rows `[:m]` and `[-m:]`, so $m \le \lfloor n/2\rfloor = 36$ retains every mode once. Modes 4 to 16 are well inside this, and also inside the Nyquist limit of the unpadded 64-point grid (32). **VERIFIED.** At 256 × 256 the bound is 132.

**Bug (fixed in `7fdb421`).** `max_fno_modes` returned $\lfloor n/2\rfloor + 1$. At that value the layer runs, but the middle row is multiplied by both weight blocks (gain 2 with identity weights). The README called this "the largest admissible number of modes". **INCORRECT as stated; fixed.** Regression test: `tests/test_model.py::test_max_fno_modes_counts_every_mode_once` fails on the old bound and passes on the new one. No result is affected.

### 3.4 Seeds, splits and leakage (task 6)

| Item | Finding | Status |
|---|---|---|
| Dataset seeds | NumPy global generator seeded with 1000 / 2000 / 3000 before each split | VERIFIED |
| Regeneration | test split regenerated from seed 3000 is bitwise identical to the cached file | VERIFIED (CPU) |
| Shared datasets | all runs, including the audit's new runs, load the same three files (cache keys `cba61bd0e4`, `8ba1b915bb`, `1c531c8f19`) | VERIFIED |
| Run seeds | `torch.manual_seed`, model built on CPU, batch order from a CPU generator seeded by the run seed | VERIFIED (code, and `tests/test_pipeline.py::test_training_is_reproducible` on CPU) |
| Leakage | 0 exact duplicates; nearest test-to-train Hamming distance at least 30.8% (median 36%); the same within the training set (31.2%) | VERIFIED |
| Bitwise reproducibility of MPS training | not tested; the new seeds were run, the old ones were not re-run | UNVERIFIED |

### 3.5 Provenance

`results/local/study_metadata.json` records commit `cfb157c` with `git_dirty: true`. `git diff 37a50d4 cfb157c -- src configs` is empty, as the README states. **VERIFIED.** Every later change to `src/` and `configs/` up to `5c81978` concerns metadata, cache selection for other devices and plotting. None changes the training or evaluation arithmetic, and the cached data regenerate bitwise. **VERIFIED.** The audit's new study (`results/audit_seeds345/`) records the clean commit `5c81978`.

### 3.6 Discretisation and generator (tasks 3 to 5, 7)

See [GENERATOR_VALIDATION.md](GENERATOR_VALIDATION.md). Statements of the README:

| README statement | Status |
|---|---|
| The learning target is NVIDIA's discrete operator, not the conservative discretisation | VERIFIED |
| The kernel scales $\nabla k\cdot\nabla u$ by a factor $2\Delta x$ relative to a consistent discretisation | VERIFIED |
| Generated fields differ from the conservative solution by about a third, at every resolution | VERIFIED (31.5% to 34%) |
| They approach the solution of $k\Delta u = -1$ under refinement | VERIFIED (first order, also for smooth $k$) |
| Solver tolerance table (0.42% / 0.045% at 64, etc.) | VERIFIED at 64 by an independent assembly; 128 and 256 SUPPORTED BUT LIMITED (one implementation) |
| `Darcy2D` as shipped: 1 distinct field per batch, 4 of 1600 coefficient values (CPU) | VERIFIED |
| "It has not been tested on CUDA" | correct. The audit adds static evidence that the cause is backend-independent in Warp 1.18.0 |
| The benchmark is "posed on the unit square" | SUPPORTED BUT LIMITED: the generator's domain is $(0, (N+2)/(N+1))^2$, and the crop drops the last row and column |

New finding: the shipped generator's fields form a three-parameter family, and under the upstream protocol 98.8% of fresh validation fields lie within 5% Hamming distance of a training field.

## 4. Robustness of the eight-mode result (task 11)

### 4.1 Uncertainty in the original three seeds

| Modes | mean | sd over seeds | t-based 95% interval of the mean | test-set sampling SE per run | final-checkpoint jitter |
|---|---|---|---|---|---|
| 4 | 1.458% | 0.152 pp | 1.08% to 1.84% | 0.027 pp | 0.045 pp |
| 8 | 0.920% | 0.118 pp | 0.63% to 1.21% | 0.019 pp | 0.040 pp |
| 12 | 0.979% | 0.023 pp | 0.92% to 1.04% | 0.023 pp | 0.036 pp |
| 16 | 0.970% | 0.012 pp | 0.94% to 1.00% | 0.025 pp | 0.039 pp |

"Final-checkpoint jitter" is the standard deviation of the validation error over the last five pseudo-epochs of a run. It shows how much the reported error depends on stopping exactly at pseudo-epoch 30. It is about as large as the differences between 8, 12 and 16 modes (0.05 to 0.06 pp). Seed variation dominates test-set sampling by a factor of 5 or more, so the seed is the unit of replication.

With three seeds per setting: 8 vs 12, Welch $p = 0.48$; 8 vs 16, $p = 0.54$. An exact permutation test on 3 + 3 seeds cannot go below $p = 0.10$ even for complete separation, so three seeds cannot establish any ordering at the 5% level.

### 4.2 Three further seeds (run in this audit)

Seeds 3, 4, 5, identical configuration and data, commit `5c81978`, Apple MPS (`results/audit_seeds345/`):

| Modes | seeds 0 to 2 (original) | seeds 3 to 5 (new) | all six seeds | 95% interval (six seeds) |
|---|---|---|---|---|
| 4 | 1.458% | 1.388% | **1.423% ± 0.116** | 1.30% to 1.55% |
| 8 | 0.920% | 0.916% | **0.918% ± 0.117** | 0.80% to 1.04% |
| 12 | 0.979% | 0.895% | **0.937% ± 0.080** | 0.85% to 1.02% |
| 16 | 0.970% | 0.847% | **0.908% ± 0.092** | 0.81% to 1.00% |

Six-seed comparisons (`results/audit/reduced_study_statistics_6seeds.json`):

| Comparison | difference | Welch $p$ | permutation $p$ | seed ranges overlap |
|---|---|---|---|---|
| 4 vs 8 | +0.51 pp | < 0.001 | 0.002 | no |
| 4 vs 12 | +0.49 pp | < 0.001 | 0.002 | no |
| 4 vs 16 | +0.52 pp | < 0.001 | 0.002 | no |
| 8 vs 12 | −0.02 pp | 0.75 | 0.74 | yes |
| 8 vs 16 | +0.01 pp | 0.88 | 0.89 | yes |
| 12 vs 16 | +0.03 pp | 0.58 | 0.56 | yes |

The new seeds reverse the order of 8, 12 and 16, and the small seed spread of 12 and 16 modes in the original study (0.01 to 0.02 pp) does not recur (0.10 pp in the new seeds). The README's statement that "seed sensitivity is larger for the small models" was an artefact of three seeds. **INCORRECT** in the light of six seeds.

Per-sample paired tests over the 256 shared test samples give very small $p$-values for 8 vs 12 or 16, but their sign changes between the two groups of seeds. They treat test samples as independent replicates of a fixed pair of trained models and say nothing about new training seeds, so they are not evidence of a ranking.

The timings of these runs are not used: concurrent audit work on the CPU slowed some of them (271 to 425 s for the same work).

### 4.3 Three times the training budget (run in this audit)

Modes 8, 12 and 16, seed 0, the same data. 90 pseudo-epochs (2880 steps) instead of 30, with the learning rate decayed every 9 pseudo-epochs instead of 3, so that the schedule has the same shape and the same final learning rate ($2.3\times10^{-4}$). Commit `c7cb072`, Apple MPS (`results/audit_long90/`):

| Modes | test error, 960 steps (seed 0) | test error, 2880 steps (seed 0) | final training loss, 960 / 2880 steps | test MSE / training loss, 2880 steps |
|---|---|---|---|---|
| 8 | 1.03% | **0.348%** | $3.5\times10^{-4}$ / $4.0\times10^{-5}$ | 1.17 |
| 12 | 0.97% | **0.327%** | $3.4\times10^{-4}$ / $2.9\times10^{-5}$ | 1.47 |
| 16 | 0.96% | **0.340%** | $3.0\times10^{-4}$ / $2.9\times10^{-5}$ | 1.58 |

Three times the optimiser steps lower the error by a factor of about 3 for every setting. The three settings again lie within 0.02 percentage points of each other. That is one seed each, so no ordering is claimed. The final-checkpoint jitter falls to 0.003 to 0.008 pp.

Consequences:

- **The reduced study is limited by training, not by capacity.** The error level of about 0.9% that 8, 12 and 16 modes share at 960 steps is a property of the budget. The README's own caveat ("the comparison is at a fixed training budget and not at convergence") is confirmed and turns out to be a large effect.
- **The plateau above 8 modes persists at 3× the budget** (one seed). Whether it persists at convergence, and whether the gap between 4 and 8 modes changes with training length, is not known: 4 modes was not run at 3× the budget.
- With longer training the ratio of test MSE to final training loss stays at 1.17 for 8 modes (1.18 at 960 steps) but rises to 1.47 for 12 modes (from 1.10) and to 1.58 for 16 modes (from 1.21). Overfitting to the 1024 training fields starts to appear for the larger models, so the data budget, not only the step budget, will matter for a converged comparison.
- The error is still 7 times the solver error (0.045%), so solver accuracy is not yet a limit.

### 4.4 Conclusion on the eight-mode result

At three times the budget the error is three times lower and the settings above 8 modes still coincide (one seed), so the plateau is not an artefact of the short schedule alone, but its level is. At 64 × 64, with 1024 training samples and 960 steps, retaining 4 modes per dimension gives a clearly higher error than retaining 8 or more. Beyond 8 modes no difference is detectable with six seeds, and the parameter count grows fourfold from 8 to 16 modes without measurable benefit. The location of the plateau is a property of this protocol. It may move with resolution, data, training length, the permeability's frequency content and the architecture's width.

## 5. Changes made by this audit

| Commit | Change | Justification |
|---|---|---|
| `7fdb421` | `max_fno_modes` returns $\lfloor (N+p)/2\rfloor$; regression test; README sentence | demonstrated double counting at the old bound (§3.3) |
| `c7cb072` | `scripts/audit_*.py`, `tests/test_generator_audit.py`, `requirements-lock.txt`, `docs/GENERATOR_VALIDATION.md`, `docs/UPSTREAM_COMPARISON.md`, `results/audit/` | reproducible evidence for this audit |
| later commits | `results/audit_seeds345/`, the longer-training study, this document, `docs/EXPERIMENT_GAPS.md`, README corrections | see git log |

The original records in `results/local/` are unchanged. The README keeps the original three-seed table as the published study and adds the six-seed evidence next to it.

## 6. How to reproduce this audit

```bash
uv venv --python 3.12 && uv pip install -r requirements-lock.txt   # or: pip install -r requirements.txt
python -m pytest -q                                    # 62 tests, CPU, no network
python scripts/audit_results.py                        # three-seed statistics from results/local
python scripts/audit_results.py --study results/local --study results/audit_seeds345 \
       --out results/audit/reduced_study_statistics_6seeds.json
python scripts/train.py --config local                 # needed once for the checkpoints and data (~65 min, MPS)
python scripts/audit_checkpoints.py                    # recompute errors from runs/local checkpoints (CPU)
python scripts/audit_data.py                           # regeneration, leakage, generator family, discretisation (CPU, ~1 min)
python scripts/train.py --config local name=audit_seeds345 'study.seeds=[3,4,5]'
```
