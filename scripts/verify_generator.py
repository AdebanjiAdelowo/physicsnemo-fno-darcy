"""Check what the PhysicsNeMo ``Darcy2D`` generator produces, independently of any model.

    python scripts/verify_generator.py
    python scripts/verify_generator.py --device cuda

Three checks, with the generator on the chosen Warp device (the direct solves always run on CPU):

1. Sample diversity of ``Darcy2D`` as shipped and of ``IndependentDarcy2D``.
2. Convergence of the multigrid Jacobi iteration: the generated pressure against a sparse
   direct solve of the stencil the iteration uses, for two stopping tolerances.
3. Which continuous problem the stencil represents: the generated pressure against direct
   solves of the conservative Darcy equation and of ``k * laplace(u) = -1``.

Writes ``results/generator_check.json`` and ``figures/generator_check.png`` for CPU, and
``results/generator_check_cuda.json`` and ``figures/generator_check_cuda.png`` for CUDA.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from _bootstrap import ROOT
from fno_darcy.data import IndependentDarcy2D, darcy_residual
from fno_darcy.plotting import ERROR_CMAP, FIELD_CMAP, INPUT_CMAP, _image, plt
from fno_darcy.provenance import collect_metadata, write_json
from fno_darcy.reference import FORMS, direct_solve
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

SEED = 0


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def diversity(cls, device: str, batch_size: int = 16, n_batches: int = 2) -> dict:
    np.random.seed(SEED)
    pipe = cls(resolution=32, batch_size=batch_size, device=device)
    per_batch, coefficients, fields = [], [], set()
    for _, batch in zip(range(n_batches), pipe):
        k = batch["permeability"].cpu().numpy()
        per_batch.append(len({f.tobytes() for f in k}))
        fields |= {f.tobytes() for f in k}
        coefficients.append(len(np.unique(pipe.rand_fourier.numpy())))
    return {
        "batch_size": batch_size,
        "batches": n_batches,
        "distinct_fields_per_batch": per_batch,
        "distinct_fields_total": len(fields),
        "coefficients_per_batch": int(np.prod(pipe.fourier_dim)),
        "distinct_coefficient_values_per_batch": coefficients,
    }


def solver_check(resolution: int, n_samples: int, thresholds: tuple, device: str) -> tuple[dict, dict]:
    out = {"resolution": resolution, "n_samples": n_samples, "by_threshold": []}
    for threshold in thresholds:
        np.random.seed(SEED)  # same permeability fields for every threshold
        pipe = IndependentDarcy2D(resolution=resolution, batch_size=n_samples, device=device, convergence_threshold=threshold)
        next(iter(pipe))
        u = pipe.darcy0.numpy().astype(np.float64)
        k = pipe.permeability.numpy().astype(np.float64)
        direct = {form: np.stack([direct_solve(k[b], pipe.dx, form) for b in range(n_samples)]) for form in FORMS}
        entry = {"convergence_threshold": threshold}
        for form in FORMS:
            entry[f"rel_l2_vs_{form}_solve"] = float(np.mean([rel_l2(u[b], direct[form][b]) for b in range(n_samples)]))
        for form in ("kernel", "flux"):
            entry[f"{form}_residual_rms"] = float(np.sqrt(np.mean(darcy_residual(u, k, pipe.dx, form=form) ** 2)))
        out["by_threshold"].append(entry)
    example = {"k": k[0], "generated": u[0], "flux": direct["flux"][0], "resolution": resolution, "threshold": thresholds[-1]}
    return out, example


def plot_example(example: dict, path) -> None:
    gen, flux = example["generated"], example["flux"]
    vmax = max(gen.max(), flux.max())
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.5), constrained_layout=True)
    _image(axes[0], example["k"], cmap=INPUT_CMAP, vmin=0.0, vmax=2.5)
    axes[0].set_title("Permeability $k$: light 0.5, dark 2")
    im = _image(axes[1], gen, cmap=FIELD_CMAP, vmin=0.0, vmax=vmax)
    axes[1].set_title("Pressure from Darcy2D")
    _image(axes[2], flux, cmap=FIELD_CMAP, vmin=0.0, vmax=vmax)
    axes[2].set_title(r"Direct solve of $-\nabla\cdot(k\nabla u)=1$")
    fig.colorbar(im, ax=axes[1:3], shrink=0.85, pad=0.02, label="pressure $u$ (shared scale)")
    err = _image(axes[3], np.abs(gen - flux), cmap=ERROR_CMAP, vmin=0.0)
    axes[3].set_title(f"|difference|, rel. $L^2$ = {100 * rel_l2(gen, flux):.0f}%")
    fig.colorbar(err, ax=axes[3], shrink=0.85, pad=0.03, label="absolute difference")
    for ax in axes:
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    n = example["resolution"]
    fig.suptitle(f"The generated reference field and the conservative Darcy solution for the same permeability ({n}×{n} grid)", color="#0b0b0b")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Warp device of the generator")
    parser.add_argument("--out-root", type=Path, default=ROOT, help="directory that receives results/ and figures/ (default: the repository)")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available on this machine.")
    suffix = "" if args.device == "cpu" else f"_{args.device}"

    report = {
        "environment": collect_metadata(torch.device(args.device)),
        "generator_device": args.device,
        "seed": SEED,
        "diversity": {
            "Darcy2D": diversity(Darcy2D, args.device),
            "IndependentDarcy2D": diversity(IndependentDarcy2D, args.device),
        },
        "solver": [],
    }
    print(f"generator device: {args.device}" + (f" ({report['environment']['gpu_name']})" if args.device == "cuda" else ""))
    for name, d in report["diversity"].items():
        print(f"{name}: distinct fields per batch of {d['batch_size']}: {d['distinct_fields_per_batch']}, "
              f"distinct coefficient values per batch: {d['distinct_coefficient_values_per_batch']} of {d['coefficients_per_batch']}")
    example = None
    for resolution in args.resolutions:
        result, ex = solver_check(resolution, args.samples, thresholds=(1e-6, 1e-7), device=args.device)
        report["solver"].append(result)
        example = ex if resolution == 128 or example is None else example
        for e in result["by_threshold"]:
            print(f"resolution {resolution}, tolerance {e['convergence_threshold']:g}: relative L2 against direct solves: "
                  f"own stencil {e['rel_l2_vs_kernel_solve']:.2e}, conservative Darcy {e['rel_l2_vs_flux_solve']:.2e}, "
                  f"k*laplace(u)=-1 {e['rel_l2_vs_laplace_solve']:.2e}")
    (args.out_root / "figures").mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "results" / f"generator_check{suffix}.json", report)
    plot_example(example, args.out_root / "figures" / f"generator_check{suffix}.png")
    print(f"written under {args.out_root}: results/generator_check{suffix}.json, figures/generator_check{suffix}.png")


if __name__ == "__main__":
    main()
