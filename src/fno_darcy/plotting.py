"""Figures of a study directory (``summary.json``, ``fields.npz`` and the run histories)."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# one hue, light to dark, for the ordered quantity "number of Fourier modes"
MODE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
FIELD_CMAP, ERROR_CMAP, INPUT_CMAP = "viridis", "magma", "Greys"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
)


def mode_colours(modes: list[int]) -> dict:
    """Colour of every mode count, by its position in the sorted list."""
    modes = sorted(modes)
    if len(modes) > len(MODE_RAMP):
        raise ValueError(f"the ramp has {len(MODE_RAMP)} steps; got {len(modes)} mode counts")
    ramp = MODE_RAMP[len(MODE_RAMP) - len(modes) :]
    return dict(zip(modes, ramp))


def load_study(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text())
    histories = {}
    for run in summary["runs"]:
        with open(study_dir / run["run"] / "history.csv") as f:
            rows = list(csv.DictReader(f))
        histories[run["run"]] = {
            key: np.array([float(r[key]) if r[key] != "" else np.nan for r in rows]) for key in rows[0]
        }
    return {
        "summary": summary,
        "histories": histories,
        "fields": dict(np.load(study_dir / "fields.npz")),
        "metadata": json.loads((study_dir / "study_metadata.json").read_text()),
    }


def _describe(study: dict) -> str:
    cfg = study["metadata"]["config"]
    res = cfg["training"]["resolution"]
    return f"{res}×{res} grid, {study['metadata']['device']}"


def _image(ax, field, **kwargs):
    # arrays are indexed [x, y]; transpose so that x runs to the right and y upwards
    im = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", **kwargs)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(False)
    return im


def plot_fields(study: dict, path: Path, sample: int = 0) -> None:
    """Permeability, reference pressure, and prediction and absolute error for every mode count."""
    fields = study["fields"]
    modes = sorted(int(k[-2:]) for k in fields if k.startswith("prediction_modes"))
    truth = fields["truth"][sample]
    preds = {m: fields[f"prediction_modes{m:02d}"][sample] for m in modes}
    errors = {m: np.abs(preds[m] - truth) for m in modes}
    vmin = min(truth.min(), *(p.min() for p in preds.values()))
    vmax = max(truth.max(), *(p.max() for p in preds.values()))
    emax = max(e.max() for e in errors.values())

    ncols = len(modes) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(2.75 * ncols, 5.9), constrained_layout=True)
    k = fields["permeability"][sample]
    _image(axes[1, 0], k, cmap=INPUT_CMAP, vmin=0.0, vmax=2.5)
    axes[1, 0].set_title(f"Input $k$: light {k.min():g}, dark {k.max():g}")
    _image(axes[0, 0], truth, cmap=FIELD_CMAP, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Reference pressure $u$")
    for col, m in enumerate(modes, start=1):
        f_im = _image(axes[0, col], preds[m], cmap=FIELD_CMAP, vmin=vmin, vmax=vmax)
        rel = np.linalg.norm(preds[m] - truth) / np.linalg.norm(truth)
        axes[0, col].set_title(f"FNO, {m} modes")
        e_im = _image(axes[1, col], errors[m], cmap=ERROR_CMAP, vmin=0.0, vmax=emax)
        axes[1, col].set_title(f"|error|, rel. $L^2$ = {100 * rel:.2f}%")
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    for ax in axes[0]:
        ax.set_xticklabels([])
    for ax in axes[1]:
        ax.set_xlabel("$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$y$")
    fig.colorbar(f_im, ax=axes[0, :], shrink=0.85, pad=0.015, label="pressure $u$ (shared scale)")
    fig.colorbar(e_im, ax=axes[1, 1:], shrink=0.85, pad=0.02, label="absolute error (shared scale)")
    seed = int(fields["plot_seed"])
    fig.suptitle(f"Test sample {sample}, models of seed {seed} ({_describe(study)})", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_error_vs_modes(study: dict, path: Path) -> None:
    """Test relative L2 error against the number of modes and against the parameter count."""
    runs, agg = study["summary"]["runs"], study["summary"]["by_fno_modes"]
    modes = [a["fno_modes"] for a in agg]
    colours = mode_colours(modes)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True, sharey=True)
    for ax, key, label in ((axes[0], "fno_modes", "Fourier modes per dimension"), (axes[1], "parameters", "Trainable parameters")):
        ax.plot([a[key] for a in agg], [100 * a["test_rel_l2_mean_mean"] for a in agg], color=MUTED, lw=1.2, zorder=1)
        for r in runs:
            ax.scatter(r[key], 100 * r["test_rel_l2_mean"], s=34, color=colours[r["fno_modes"]], edgecolor="#fcfcfb", lw=1.2, zorder=3)
        ax.set_xlabel(label)
        ax.set_yscale("log")
    axes[0].set_xticks(modes)
    axes[1].set_xscale("log")
    for a in agg:
        axes[1].annotate(f"{a['fno_modes']} modes", (a["parameters"], 100 * a["test_rel_l2_mean_mean"]), textcoords="offset points", xytext=(6, 7), color=MUTED, fontsize=9)
    axes[0].set_ylabel("Test relative $L^2$ error (%)")
    n_seeds = agg[0]["n_seeds"]
    fig.suptitle(f"Error against Fourier modes: one point per seed ({n_seeds}), line through the seed means ({_describe(study)})", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(study: dict, path: Path) -> None:
    """Training loss and validation error against the pseudo-epoch for every run."""
    runs = study["summary"]["runs"]
    colours = mode_colours(sorted({r["fno_modes"] for r in runs}))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    labelled = set()
    for r in runs:
        h = study["histories"][r["run"]]
        label = f"{r['fno_modes']} modes" if r["fno_modes"] not in labelled else None
        labelled.add(r["fno_modes"])
        axes[0].plot(h["pseudo_epoch"], h["train_loss"], color=colours[r["fno_modes"]], lw=1.4, alpha=0.9, label=label)
        ok = ~np.isnan(h["validation_rel_l2"])
        axes[1].plot(h["pseudo_epoch"][ok], 100 * h["validation_rel_l2"][ok], color=colours[r["fno_modes"]], lw=1.4, alpha=0.9)
    axes[0].set_ylabel("Training loss (MSE, normalised units)")
    axes[1].set_ylabel("Validation relative $L^2$ error (%)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Pseudo-epoch")
    axes[0].legend(title="one line per seed")
    fig.suptitle(f"Training history ({_describe(study)})", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_cost(study: dict, path: Path) -> None:
    """Parameter count, training time and inference time against the number of modes."""
    agg = study["summary"]["by_fno_modes"]
    modes = [a["fno_modes"] for a in agg]
    colours = [mode_colours(modes)[m] for m in modes]
    panels = [
        ("parameters", None, 1e-6, "Trainable parameters (millions)"),
        ("optimisation_seconds_mean", "optimisation_seconds_std", 1.0, "Optimisation time per run (s)"),
        ("inference_ms_per_sample_mean", "inference_ms_per_sample_std", 1.0, "Inference time per sample (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
    x = np.arange(len(modes))
    for ax, (key, err_key, scale, label) in zip(axes, panels):
        values = [a[key] * scale for a in agg]
        errs = [(a[err_key] or 0.0) * scale for a in agg] if err_key else None
        ax.bar(x, values, width=0.62, color=colours, yerr=errs, ecolor=MUTED, capsize=3)
        ax.set_xticks(x, [str(m) for m in modes])
        ax.set_xlabel("Fourier modes per dimension")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.3g}", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", color=INK, fontsize=9)
    fig.suptitle(f"Cost against Fourier modes: mean over seeds, bars show the standard deviation ({_describe(study)})", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_all(study_dir: Path, out_dir: Path, prefix: str) -> list[Path]:
    study = load_study(study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "fields": plot_fields,
        "error_vs_modes": plot_error_vs_modes,
        "training_curves": plot_training_curves,
        "cost": plot_cost,
    }
    paths = []
    for name, fn in jobs.items():
        path = out_dir / f"{prefix}_{name}.png"
        fn(study, path)
        paths.append(path)
    return paths
