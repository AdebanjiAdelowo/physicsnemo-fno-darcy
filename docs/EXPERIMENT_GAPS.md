# Experiment gaps

What has been run, what has not, and what each missing experiment would answer. Every pending item has an exact command and a resource estimate. Estimates marked *measured* come from probes on the Apple M3 Pro (18 GB unified memory) on 24 October 2026. Estimates marked *assumed* rest on hardware ratios that have not been measured.

## Status

| Experiment | Status | Where |
|---|---|---|
| Reduced study, 64 × 64, modes 4/8/12/16, seeds 0 to 2 | run | `results/local/` |
| Same, seeds 3 to 5 | run (audit) | `results/audit_seeds345/` |
| Same, three times the steps, modes 8/12/16, seed 0 | run (audit) | `results/audit_long90/` |
| Generator checks on CPU | run | `results/generator_check.json`, `results/audit/data_audit.json` |
| Generator checks on CUDA | Run | [Tesla T4 verification](CUDA_GENERATOR_VALIDATION.md) |
| Main study, 256 × 256, `configs/full.yaml` | **PENDING** | Kaggle step B |
| Upstream control, `configs/official.yaml` | **PENDING** | Kaggle step C |

## Gaps, in order of importance for the Fourier-mode question

### G1. Convergence in the training budget (local, affordable)

**Why.** Tripling the steps lowered the error threefold ([SCIENTIFIC_AUDIT.md §4.3](SCIENTIFIC_AUDIT.md)). The mode comparison at convergence is unknown, and 4 modes was not run at the longer budget, so it is not known whether the gap between 4 and 8 modes closes, stays or grows.

**What.** All four mode settings, at least three seeds, at 3× and 9× the steps, with the same final learning rate.

```bash
python scripts/train.py --config local name=budget3x training.max_pseudo_epochs=90 scheduler.decay_pseudo_epochs=9 'study.seeds=[0,1,2]'
python scripts/train.py --config local name=budget9x training.max_pseudo_epochs=270 scheduler.decay_pseudo_epochs=27 'study.seeds=[0,1,2]'
python scripts/audit_results.py --study runs/budget3x --out runs/budget3x/statistics.json
```

**Cost (measured).** 13 to 14 min per run at 3× on MPS: 12 runs ≈ 2.8 h. 9× ≈ 40 min per run: 12 runs ≈ 8 h. Disk under 100 MB per study (checkpoints included). Run when no other session uses the GPU.

### G2. Training-data budget

**Why.** At 3× the steps the test MSE exceeds the training loss by 47% to 58% for 12 and 16 modes (17% for 8). The larger models begin to fit the 1024 training fields. A converged comparison needs more data, or the comparison measures memorisation.

**What.** 4096 training fields at 64 × 64 with the 3× schedule.

```bash
python scripts/train.py --config local name=data4096 training.pseudo_epoch_sample_size=4096 training.max_pseudo_epochs=90 scheduler.decay_pseudo_epochs=9 'study.seeds=[0,1,2]'
```

**Cost (measured rates).** Generation about 16 to 20 min on CPU (the 1024-sample set took about 4 to 5 min), 134 MB. Training 4× the steps of G1-3×, about 55 min per run: 12 runs ≈ 11 h on MPS. A CUDA GPU is preferable.

### G3. CUDA generator check

The Tesla T4 run reproduced the CPU diversity counts for both generators and matched the CPU reference errors. See the [CUDA verification](CUDA_GENERATOR_VALIDATION.md).

```bash
python scripts/verify_generator.py --device cuda
```

**Measured cost.** 22.07 s for verification on a Tesla T4, including the CPU reference solves.

### G4. The 256 × 256 study (`configs/full.yaml`)

**Why.** The upstream resolution. The reduced study does not stand in for it.

**Resource estimate.**

| Item | Estimate | Basis |
|---|---|---|
| Datasets | 4864 samples, 2.5 GB on disk, about 5 GB host RAM while loaded | measured size per sample (0.52 MB) |
| Generation | about 2 h on CPU (1.5 s per sample at tolerance $10^{-7}$); faster on CUDA, not measured | measured on CPU |
| GPU memory, batch 64, fp32 | **about 20 GB** of activations (0.3 GB per sample) | measured on MPS at batch 2, 4, 8; agrees with an analytic count of the saved activations. A batch-64 step on the 18 GB M3 Pro exhausted memory and swapped |
| Compute | 3840 steps of batch 64 per run; 0.12 s per sample-step on M3 Pro MPS ≈ 8 h per run on the laptop, about 100 h for 12 runs | measured per-sample step time |
| On an A100 | 0.5 to 1.5 h per run, 6 to 20 GPU-hours for 12 runs | *assumed* 5 to 15 times the M3 Pro throughput |

**Consequence for Kaggle.** The T4 and P100 GPUs that Kaggle has offered have 16 GB (check the current hardware), below the estimated 20 GB for batch 64. Step B may fail with out-of-memory on those GPUs. The notebook's probe would show this before the main run. Two options keep the configuration valid: a GPU with at least 24 GB (A10G, L4, A100), or gradient accumulation of two batches of 32. The second needs a code change and gives the same mean-reduced gradient up to round-off. Halving the batch without accumulation would change the protocol and must be reported as a different study.

```bash
python scripts/train.py --config full            # on a >= 24 GB CUDA GPU
python scripts/evaluate.py --config full --verify device=mps data.generation_device=cuda   # locally, after unpacking the eval bundle
```

### G5. Upstream control (`configs/official.yaml`), and its test set

**Why.** It reproduces what NVIDIA's example trains on.

**Design issue to resolve first.** With `independent_samples: false` the control's validation and *test* sets are also generated by the shipped `Darcy2D`. A test set of 256 samples then holds 4 distinct fields (one per batch of 64). Its error would be an average over 4 fields drawn from a three-parameter family, and would not be comparable with the study. The control should also be evaluated on the `IndependentDarcy2D` test set of the same resolution and tolerance. That needs a small change to `evaluate.py`: an option to choose the evaluation generator independently of the training generator. **Not implemented.**

**Cost.** 8192 steps of batch 64, with streaming generation at every step. Training compute is about twice that of one `full` run (*assumed* 1 to 3.5 h on an A100). The generation cost per streamed batch on CUDA is the main unknown, and the notebook's probe measures it.

### G6. A conservative Darcy target

**Why.** All results are errors against the generator's operator ($-\Delta u = 1/k$ in the limit), whose solutions are $C^1$ across interfaces. Solutions of $-\nabla\cdot(k\nabla u) = 1$ have kinks at the interfaces and more high-frequency content, so the number of modes needed may differ.

**What.** A generator option that replaces the Jacobi output by `fno_darcy.reference.direct_solve(k, dx, "flux")`, then the reduced study on it. **Not implemented.**

**Cost (measured rate).** A direct solve at 64 × 64 takes milliseconds, so 1408 samples take under a minute. Training as in the reduced study (about 1 h for 12 runs on MPS).

### G7. Frequency content of the permeability

**Why.** A hypothesis from the audit: the shipped permeability is the sign of a series with frequencies up to 4, which correspond to FFT indices up to about 4.6 on the padded grid. That would explain why 4 modes is too few and 8 is enough. If so, the threshold moves with `nr_permeability_freq` and not with the grid.

**What.** The reduced study with `nr_permeability_freq` 3 and 8. This needs the parameter passed through `make_datapipe` (not exposed yet).

### G8. Other untested factors

- Width, depth and padding are fixed at the upstream values; they interact with the number of modes.
- One permeability contrast (4:1) and one forcing; no out-of-distribution test.
- No zero-shot resolution transfer (train at 64, test at 128).
- Bitwise reproducibility of MPS training: the original runs were not re-run with their own seeds. Cost: 5 min for one run.

```bash
python scripts/train.py --config local name=repro_seed0 'study.fno_modes=[8]' 'study.seeds=[0]'
```

- CUDA timings, and the comparison of FNO inference time with solver time on the same device.

## Not to be claimed until the gaps are closed

- An optimal number of Fourier modes, for this or any resolution.
- Any statement about the 256 × 256 benchmark, the upstream example's accuracy, or CUDA training behaviour.
- Any statement about the Darcy equation in divergence form.
- That the errors of the reduced study are converged.
