"""Reproduce Figure 3 using the pinned NeRF and NeRFStudio adapters."""

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Set CPU/runtime options before importing adapters and their ML frameworks.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

import nerf_adapter
import nerfstudio_adapter

ADAPTERS = {"nerf": nerf_adapter, "nerfstudio": nerfstudio_adapter}
EXACT_INTENSITY = .5 * -np.expm1(-2 / np.pi)
SAMPLE_COUNTS = np.unique(np.rint(np.geomspace(4, 256, 19)).astype(int))


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    backend: str
    reference_order: float | None = None


METHODS = (
    Method("nerf-infinite", r"NeRF ($\delta_K=10^{10}$)", "nerf"),
    Method("nerf-zero", r"NeRF ($\delta_K=0$)", "nerf", 1.),
    Method("nerfix", "NeRFix", "nerf", 1.5),
    Method("nerfstudio", "NeRFStudio", "nerfstudio"),
    Method("nerfstudio-fix", "NeRFStudio-Fix", "nerfstudio", 2.),
)


def measure_errors(adapter, render_batch, args):
    """Estimate MAE and RMS error from independent renders of the same ray."""
    maes, rmses = [], []
    for count in SAMPLE_COUNTS:
        adapter.set_seed(args.seed + int(count))
        absolute_sum, squared_sum = 0., 0.
        for start in range(0, args.trials, args.batch_size):
            num_rays = min(args.batch_size, args.trials - start)
            error = render_batch(num_rays, int(count)) - EXACT_INTENSITY
            absolute_sum += np.abs(error).sum()
            squared_sum += np.square(error).sum()
        maes.append(absolute_sum / args.trials)
        rmses.append(np.sqrt(squared_sum / args.trials))
    return np.array(maes), np.array(rmses)


def fit_errors(maes, rmses, fit_min, reference_order):
    """Fit E(K)=a*K^b in log space, plus an optional fixed-order reference."""
    selected = SAMPLE_COUNTS >= fit_min
    log_counts = np.log(SAMPLE_COUNTS[selected])
    log_maes = np.log(maes[selected])
    slope, intercept = np.polyfit(log_counts, log_maes, 1)
    rms_slope = np.polyfit(log_counts, np.log(rmses[selected]), 1)[0]
    fit = dict(mae_slope=float(slope), mae_coefficient=float(np.exp(intercept)),
               rmse_slope=float(rms_slope))
    if reference_order is not None:
        # Least-squares intercept with the theoretical slope held fixed.
        coefficient = np.exp(np.mean(log_maes + reference_order * log_counts))
        fit.update(reference_order=reference_order, reference_coefficient=float(coefficient))
    return fit


def plot_method(ax, method, maes, fit):
    """Draw measured points and a directly labeled theoretical reference rate."""
    coefficient, slope = fit["mae_coefficient"], fit["mae_slope"]
    ax.loglog(SAMPLE_COUNTS, maes, "o", linestyle="none",
              label=method.label + rf": $E(K)={coefficient:.3g}\,K^{{{slope:.3f}}}$")
    if method.reference_order is None:
        return
    order = method.reference_order
    reference_coefficient = fit["reference_coefficient"]
    reference = reference_coefficient * SAMPLE_COUNTS.astype(float) ** -order
    ax.loglog(SAMPLE_COUNTS, reference, "--", color="gray", alpha=.7, label="_nolegend_")
    label_k = 110.
    ax.annotate(rf"$\propto K^{{-{order:g}}}$",
                xy=(label_k, reference_coefficient * label_k ** -order),
                xytext=(0, 12), textcoords="offset points",
                ha="center", va="bottom", color="dimgray")


def save_results(args, rows, summaries):
    """Write measurements and the settings needed to reproduce them."""
    with (args.output / "errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "samples", "mae", "rmse"))
        writer.writerows(rows)
    metadata = dict(
        upstream_commit=nerf_adapter.COMMIT, nerfstudio_commit=nerfstudio_adapter.COMMIT,
        torch=nerfstudio_adapter.VERSION, nerfstudio_evaluations="K-1",
        seed=args.seed, trials=args.trials, batch_size=args.batch_size,
        fit_min=args.fit_min, dtype="float64", near=0., far=1., color=.5,
        exact=EXACT_INTENSITY, tensorflow=nerf_adapter.VERSION,
        numpy=np.__version__, slopes=summaries)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def reproduce(args, backends):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        adapter = ADAPTERS[method.backend]
        # Each adapter owns its shadow and restores the original on exit.
        with adapter.use_method(backends[method.backend], method.name) as render_batch:
            maes, rmses = measure_errors(adapter, render_batch, args)
        fit = fit_errors(maes, rmses, args.fit_min, method.reference_order)
        summaries[method.name] = fit
        rows.extend((method.name, int(k), mae, rmse)
                    for k, mae, rmse in zip(SAMPLE_COUNTS, maes, rmses))
        plot_method(ax, method, maes, fit)
        print(f"{method.name}: MAE slope {fit['mae_slope']:.4f}; "
              f"RMSE slope {fit['rmse_slope']:.4f}")
    ax.set(xlabel="Number of samples K", ylabel="Average absolute rendering error")
    ax.grid(True, which="both", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / ("figure3.png"), dpi=200)
    plt.close(fig)
    save_results(args, rows, summaries)
    print("Saved plots, errors.csv and metadata.json to", args.output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/figure3"))
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-min", type=int, default=16)
    parser.add_argument("--check", action="store_true", help="run numerical checks and exit")
    args = parser.parse_args()
    if args.trials < 1 or args.batch_size < 1 or not 4 <= args.fit_min <= 128:
        parser.error("positive trials/batch-size and 4 <= fit-min <= 128 required")
    return args


def main():
    args = parse_args()
    backends = {}
    for name, adapter in ADAPTERS.items():
        adapter.configure(args.seed)
        backends[name] = adapter.load()
    if args.check:
        for name, adapter in ADAPTERS.items():
            adapter.self_check(backends[name])
    else:
        reproduce(args, backends)


if __name__ == "__main__":
    main()
