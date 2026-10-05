# Results

Tracked result files. Each study directory is a copy of a run directory without its checkpoints, written by `python scripts/results.py publish <run directory or archive>`.

| Path | Content |
|---|---|
| `generator_check.json` | output of `scripts/verify_generator.py`: sample diversity and solver checks of the data generator |
| `<study>/study_metadata.json` | timestamp, git commit, device, package versions, resolved configuration, dataset records |
| `<study>/summary.csv` | one row per run |
| `<study>/summary.json` | the same rows, and mean and standard deviation over seeds per number of Fourier modes |
| `<study>/fields.npz` | permeability, reference pressure and predictions of the plotted test samples |
| `<study>/modes<M>_seed<S>/history.csv` | learning rate, training loss and validation error per pseudo-epoch |
| `<study>/modes<M>_seed<S>/train_metrics.json` | parameter count, timings, memory |
| `<study>/modes<M>_seed<S>/eval_metrics.json` | validation and test errors, per-sample test errors, inference time |

Studies present:

| Study | Configuration | Status |
|---|---|---|
| `local` | `configs/local.yaml`, 64 × 64 grid | run, see the README of the repository |
| `full` | `configs/full.yaml`, 256 × 256 grid, CUDA | not run |
| `official` | `configs/official.yaml`, upstream settings, CUDA | not run |

Units: `*_mse_normalised` and the training loss are mean squared errors of the normalised pressure. `test_rel_l2_*` are relative $L^2$ errors of the pressure in physical units, as fractions. Times are wall-clock seconds or milliseconds with the device synchronised. Memory fields are CUDA peak allocations and are empty on other devices.
