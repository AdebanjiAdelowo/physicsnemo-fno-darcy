# Generator validation

What the PhysicsNeMo `Darcy2D` generator (commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`, Warp 1.18.0) produces, and what the `IndependentDarcy2D` modification of this repository changes. Every number below comes from a script in this repository and a tracked JSON file. All executions are on CPU (macOS arm64). Nothing was executed on CUDA.

Classifications: **VERIFIED** (reproduced by execution or by an independent second implementation), **SUPPORTED BUT LIMITED** (evidence is sound but incomplete in scope), **UNVERIFIED** (not tested), **INCORRECT** (contradicted by evidence).

| Evidence file | Produced by |
|---|---|
| [`results/generator_check.json`](../results/generator_check.json) | `python scripts/verify_generator.py` (commit `5a68284`) |
| [`results/audit/data_audit.json`](../results/audit/data_audit.json) | `python scripts/audit_data.py` (this audit) |
| [`tests/test_generator_audit.py`](../tests/test_generator_audit.py), [`tests/test_data.py`](../tests/test_data.py) | `python -m pytest -q` |

## 1. The grid

From `darcy.py` and the kernels in `physicsnemo/datapipes/benchmarks/kernels/`:

- The solver array has $N + 1$ nodes per dimension for `resolution` $N$, spacing $h = 1/(N+1)$.
- Nodes outside the array read as $u = 0$ (`index_zero_edges_batched_2d`) and the permeability outside is extended by its edge value (`index_clamped_edges_batched_2d`).
- With array node $i$ at $x_i = (i+1)h$, the zero ghost nodes are at $x = 0$ and $x = (N+2)h$. The domain is therefore $(0, (N+2)/(N+1))^2$, about 1.6% longer than the unit square at $N = 64$.
- `__iter__` returns `[:N, :N]` of the array. The crop drops the last row and column (the node next to the far boundary) and keeps the first, so the returned grid is not symmetric about the domain centre.

These are small geometric details. They matter only when the data are compared with a solver on the exact unit square. **VERIFIED** (source reading).

## 2. The discrete equation that is solved

The Jacobi kernel `darcy_mgrid_jacobi_iterative_batched_2d` sets

$$u^{\ast}_{ij} = \frac{k_{ij}\,(u_W + u_E + u_S + u_N)/h^2 + (k_E - k_W)(u_E - u_W)/(2h) + (k_N - k_S)(u_N - u_S)/(2h) + f}{4k_{ij}/h^2},$$

with $f = 1$. Its fixed point is

$$k\,\Delta_h u + \frac{\delta_x k\,\delta_x u + \delta_y k\,\delta_y u}{2h} + 1 = 0,$$

where $\delta$ is the difference of the two neighbours. A consistent central discretisation of $k\Delta u + \nabla k\cdot\nabla u + 1 = 0$ (the non-conservative form of $\nabla\cdot(k\nabla u) = -1$) divides the cross term by $4h^2$. The kernel's cross term is the consistent one multiplied by $2h$, so it vanishes as $h \to 0$.

**Consequence.** The stencil is consistent with

$$-\Delta u = 1/k \quad \text{in } \Omega, \qquad u = 0 \text{ on } \partial\Omega,$$

a Poisson problem with a piecewise-constant source. It is not consistent with the Darcy equation $-\nabla\cdot(k\nabla u) = 1$. For piecewise-constant $k$ the cross term is $O(1)$ on the single layer of nodes next to an interface. That perturbation is supported on a set of width $O(h)$, so its effect on $u$ also vanishes at first order. The solutions of $-\Delta u = 1/k$ are $C^1$ across interfaces, which is why the generated fields have no kinks where the permeability jumps.

### Evidence

**(a) Direct solves of three discrete problems** for the same permeability fields as the generator (`IndependentDarcy2D`, seed 0, 4 samples, tolerance $10^{-7}$). Relative $L^2$ difference from the generated pressure:

| Grid | own stencil | conservative, harmonic faces | $k\,\Delta_h u = -1$ |
|---|---|---|---|
| 64 | 0.045% | 31.5% | 1.63% |
| 128 | 0.16% | 33.0% | 1.02% |
| 256 | 0.17% | 34.0% | 0.50% |

The direct solves were assembled twice, independently. `fno_darcy.reference.direct_solve` builds the matrix entry by entry; `scripts/audit_data.py` builds it from Kronecker products of shift operators. At 64 × 64 the two agree with each other to 11 significant digits (0.045445162783% and 0.045445162784%; 31.4907485645% in both). **VERIFIED.**

**(b) Refinement with a smooth permeability**, $k = \exp(\sin 2\pi x \cos 2\pi y)$. Direct solves only, no Jacobi iteration:

| Grid | kernel vs $k\Delta u = -1$ | kernel vs conservative | consistent non-conservative vs conservative |
|---|---|---|---|
| 32 | 1.43% | 19.3% | 0.109% |
| 64 | 0.77% | 21.3% | 0.028% |
| 128 | 0.40% | 22.4% | 0.0075% |
| 256 | 0.20% | 23.0% | 0.0020% |

The generator stencil approaches $k\Delta u = -1$ at first order (the difference halves per refinement) and stays about 23% away from the conservative solution. Replacing only the divisor $2h$ by $4h^2$ gives a stencil that converges to the conservative solution at second order (a factor of about 4 per refinement). The mis-scaled cross term is therefore the whole cause of the discrepancy. `tests/test_generator_audit.py` keeps the first two columns as tests. **VERIFIED.**

### What is learned

The FNO in this repository learns the discrete map

$$\mathcal{G}_h : k \mapsto \text{crop}\bigl(\text{Jacobi}_{\text{tol}}(A_h^{\text{kernel}}(k),\, f)\bigr),$$

that is, the generator's stencil, solved to its stopping tolerance and cropped. It is not a discretisation of the continuous Darcy solution operator $k \mapsto u$ with $-\nabla\cdot(k\nabla u) = 1$. As $h \to 0$, $\mathcal{G}_h$ approaches the solution operator of $-\Delta u = 1/k$. All test errors in this repository are errors against $\mathcal{G}_h$.

## 3. Stopping tolerance

The iteration stops when the largest change over one Jacobi step, checked every 1000 steps, is below `convergence_threshold` (times the grid reduction factor on coarse levels). This is a criterion on the update size, not on the residual, and on fine grids Jacobi updates are small long before the error is small:

| Grid | error vs own stencil, tolerance $10^{-6}$ (default) | tolerance $10^{-7}$ |
|---|---|---|
| 64 | 0.42% | 0.045% |
| 128 | 0.27% | 0.16% |
| 256 | 1.33% | 0.17% |

At 64 × 64 with $10^{-7}$ the solver error (0.045%) is about 20 times smaller than the best FNO error in the reduced study (0.92%), so it does not limit the comparison. At 256 × 256 with the upstream default tolerance the solver error (1.3%) exceeds that FNO error. **VERIFIED** at 64 by the independent assembly; 128 and 256 come from `results/generator_check.json` and rest on the first assembly only (**SUPPORTED BUT LIMITED**).

## 4. Sample diversity of `Darcy2D` as shipped

`initialize_batch` fills the coefficient array of shape `[4, batch, 5, 5]` with `init_uniform_random_4d`, which calls `wp.rand_init(seed, wp.tid())`.

**Mechanism.** In Warp 1.18.0, `wp.tid()` used as a single value returns the leading launch index. The documentation of the builtin says so (`i = wp.tid()` "Returns the first thread index"), and the generated code calls `wp::tid(index, bounds)`, which returns `coord.i`. Every coefficient is therefore seeded by its trigonometric component index $c \in \{0,1,2,3\}$ alone.

**Consequences, measured on CPU:**

| Property | Measured | Status |
|---|---|---|
| Distinct fields in a batch of 16 | 1 (two batches) | VERIFIED |
| Distinct coefficient values per batch | 4 of 1600 | VERIFIED |
| Each value depends only on the leading index | yes, 1 value per component | VERIFIED (`test_shipped_coefficients_depend_only_on_the_leading_launch_index`) |
| Field unchanged when the four amplitudes are scaled by a positive constant | yes | VERIFIED |
| Different batches give different fields | yes (new NumPy-drawn seed per batch) | VERIFIED |

A shipped field is $k = \text{threshold}_0\bigl(\sum_{c=0}^{3} a_c S_c\bigr)$ with four fixed patterns $S_c$, and thresholding at zero ignores the length of $a$. **The shipped generator draws from a three-parameter family** (a direction in $\mathbb{R}^4$), not from a random field with 100 independent coefficients.

**Effect on the upstream protocol.** The upstream example trains for 256 pseudo-epochs of 32 batches, which is 8192 distinct fields, each repeated 64 times, and validates on fresh draws from the same family. Simulated at 64 × 64 with the real Warp coefficient kernel and a NumPy copy of the synthesis (pixel-identical to the Warp kernel), here is how far 256 fresh validation fields are from their nearest training field:

| Sampler | median nearest-neighbour Hamming distance | within 1% of pixels | within 5% |
|---|---|---|---|
| `Darcy2D` as shipped | 2.0% | 18% | 98.8% |
| `IndependentDarcy2D` | 34% | 0% | 0% |

The upstream validation error therefore measures interpolation within a densely sampled three-parameter family. Errors obtained that way are not comparable with errors on independent fields. **VERIFIED** (CPU, simulation of the protocol, not a training run).

**On CUDA.** The CPU and CUDA code paths of Warp 1.18.0 call the same templated `wp::tid` function, which is compiled for both targets, so the same behaviour is expected on CUDA. This is static evidence only: **UNVERIFIED by execution**. `python scripts/verify_generator.py --device cuda` (Kaggle step A) is the test. Older Warp versions may differ; the README reports the same pattern in a standalone kernel on Warp 1.5.1 to 1.18.0 on CPU, which this audit did not repeat.

## 5. `IndependentDarcy2D`

`IndependentDarcy2D.initialize_batch` replaces only the coefficient draw with `numpy.random.uniform(-1, 1, size=(4, batch, 5, 5))` and then calls the upstream synthesis, threshold and solver kernels unchanged.

| Check | Result | Status |
|---|---|---|
| Distinct fields | 16 of 16 per batch; 1024 of 1024 in the training set | VERIFIED |
| Coefficients actually used | 81 of 100 per sample (the sine factor vanishes for frequency 0), all independent | VERIFIED (source reading) |
| Exact duplicates train/validation/test | 0 | VERIFIED |
| Smallest Hamming distance test to train | 30.8% of pixels (median 36%) | VERIFIED |
| Smallest Hamming distance within the training set | 31.2% | VERIFIED |
| Regeneration of the cached test split from seed 3000 | bitwise identical | VERIFIED (CPU) |
| Solver | upstream, unchanged | VERIFIED (source reading) |

The splits are seeded with 1000 (train), 2000 (validation) and 3000 (test) through NumPy's global generator. No test field is a duplicate or a near-duplicate of a training field. **There is no data leakage.**

`IndependentDarcy2D` fixes the diversity defect, but its fields do not come from the distribution the shipped generator was presumably meant to sample. They are a modification by this repository, and results on them describe this repository's benchmark, not NVIDIA's.

## 6. Normalisation constants

The upstream constants (permeability $1.25 \pm 0.75$, pressure $0.0452 \pm 0.0279$) are used unchanged. On the 64 × 64 `IndependentDarcy2D` data the measured pressure statistics are $0.0475 \pm 0.0271$, within 5% of the constants. Errors are computed after inverting the normalisation, in physical units, so the constants affect only the conditioning of training. **VERIFIED.**
