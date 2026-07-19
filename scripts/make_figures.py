"""Regenerate every committed figure from the committed configs and results.

Run from the repository root after the benchmarks:

    python scripts/make_figures.py

Figures land in figures/. The hero scan is the first scan of the compare
benchmark (same seed derivation), so the hero figure shows exactly the data
the headline numbers were measured on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orient4d import cluster as cluster_mod  # noqa: E402
from orient4d import net as net_mod  # noqa: E402
from orient4d import template as template_mod  # noqa: E402
from orient4d.benchmark import _condition_seed, _detector, _scene_params, load_config  # noqa: E402
from orient4d.metrics import angular_error_deg, grain_agreement  # noqa: E402
from orient4d.plots import orientation_rgb, show_pattern  # noqa: E402
from orient4d.sim import (  # noqa: E402
    FOLDS,
    PHASES,
    DetectorGeometry,
    clean_pattern,
    simulate_pattern,
    simulate_scene_scan,
)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RES = ROOT / "results"

METHOD_STYLE = {
    "template": {"color": "#1f77b4", "marker": "o", "label": "template matching (0.5 deg)"},
    "cnn": {"color": "#d62728", "marker": "s", "label": "CNN"},
}


def _load(name: str) -> dict:
    with open(RES / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def _sweep_series(payload: dict, method: str, key: str) -> tuple[list, list, list]:
    xs, means, stds = [], [], []
    for cond in payload["conditions"]:
        agg = cond["methods"][method]["aggregate"].get(key)
        if agg is None:
            continue
        xs.append(cond["value"])
        means.append(agg["mean"])
        stds.append(agg["std"])
    return xs, means, stds


def _sweep_figure(
    name: str,
    xlabel: str,
    out: str,
    logx: bool = False,
    title: str | None = None,
) -> None:
    payload = _load(name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, style in METHOD_STYLE.items():
        if method not in payload["conditions"][0]["methods"]:
            continue
        for ax, key, ylabel in [
            (axes[0], "interior_orientation_mae_deg", "orientation MAE (deg)"),
            (axes[1], "interior_phase_accuracy", "phase accuracy"),
        ]:
            xs, means, stds = _sweep_series(payload, method, key)
            ax.errorbar(xs, means, yerr=stds, capsize=3, **style)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if logx:
                ax.set_xscale("log")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        title or f"{name.replace('_', ' ')} (grain-interior pixels, mean +/- std over 3 scans)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


def _colorwheel(ax, label: str) -> None:
    n = 256
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    r = np.hypot(xx, yy)
    ang = np.degrees(np.arctan2(yy, xx)) % 360.0
    hue = ang / 360.0
    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue) * 0.95, np.ones_like(hue)], axis=-1))
    alpha = ((r < 1.0) & (r > 0.55)).astype(float)
    ax.imshow(np.dstack([rgb, alpha]), origin="lower")
    ax.text(n / 2, n / 2, label, ha="center", va="center", fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis("off")


def hero_figure() -> None:
    cfg = load_config(ROOT / "configs" / "compare.yaml")
    params = _scene_params(cfg)
    det = _detector(cfg)
    scan = simulate_scene_scan(params, det, _condition_seed(cfg["seed"], 0, 0))

    lib = template_mod.build_library(det, step_deg=0.5)
    phase_t, theta_t, _ = template_mod.map_scan(scan.data, lib)
    model = net_mod.load_model(str(ROOT / "models" / "orientnet.pt"))
    phase_c, theta_c = net_mod.map_scan(model, scan.data)
    clus = cluster_mod.cluster_grains(scan.data, k=int(params.n_grains), seed=cfg["seed"])
    ari = grain_agreement(clus.labels, scan.grain_id)["ari"]

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.55])

    # Spot dark fields: apertures on a first-ring Bragg position of each
    # phase's unrotated lattice. Only grains oriented to diffract into that
    # exact spot light up (the per-pattern intensity budget is fixed in this
    # simulator, so integrating BF/ADF disks gives no grain contrast; spot
    # dark field is the virtual image that carries it).
    from orient4d.sim import lattice_reflections
    from orient4d.virtual import spot_dark_field

    g_hex, _ = lattice_reflections(PHASES["hexagonal"], det.k_max)
    g0 = g_hex[np.argmin(np.linalg.norm(g_hex, axis=1))]
    g_sq, _ = lattice_reflections(PHASES["square"], det.k_max)
    r_sq = float(np.linalg.norm(g_sq, axis=1).min())
    az = np.deg2rad(40.0)
    sdf_a = spot_dark_field(scan.data, det, (float(g0[1]), float(g0[0])), radius=0.07)
    sdf_b = spot_dark_field(
        scan.data, det, (r_sq * float(np.sin(az)), r_sq * float(np.cos(az))), radius=0.07
    )

    panels = [
        (0, 0, sdf_a, "gray", "spot dark field, aperture A"),
        (0, 1, sdf_b, "gray", "spot dark field, aperture B"),
        (0, 2, orientation_rgb(theta_t, phase_t), None, "orientation map: template matching"),
        (0, 3, orientation_rgb(scan.theta, scan.phase_id), None, "orientation map: ground truth"),
        (1, 2, orientation_rgb(theta_c, phase_c), None, "orientation map: CNN"),
    ]
    for row, col, img, cmap, title in panels:
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(scan.phase_id, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title("ground-truth phase\n(blue hexagonal, red square)", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(clus.labels, cmap="tab20", interpolation="nearest")
    ax.set_title(f"unsupervised grain clusters\n(k-means, ARI {ari:.2f})", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    err = angular_error_deg(theta_t, scan.theta, FOLDS[scan.phase_id.astype(int)])
    err = np.where(phase_t == scan.phase_id, err, np.nan)
    im = ax.imshow(err, cmap="viridis", vmin=0, vmax=2.0)
    ax.set_title("template orientation error (deg)", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 4])
    _colorwheel(ax, "hue =\norientation")
    ax = fig.add_subplot(gs[1, 4])
    ax.axis("off")
    ax.text(
        0.0,
        0.5,
        f"{params.scan_shape[0]}x{params.scan_shape[1]} probe positions\n"
        f"{params.n_grains} grains, 2 phases\n"
        f"dose {params.dose:g} e-/pattern\n"
        "hexagonal: full saturation\nsquare: desaturated",
        fontsize=8,
        va="center",
    )
    fig.suptitle(
        "4D-STEM orientation mapping: virtual images, orientation/phase maps, grain clusters",
        fontsize=12,
    )
    fig.tight_layout()
    # Higher dpi so the short side clears 1080 px for social posts; layout,
    # colormaps and fonts are unchanged.
    fig.savefig(FIG / "hero.png", dpi=170)
    plt.close(fig)


def pattern_gallery() -> None:
    det = DetectorGeometry()
    rng = np.random.default_rng(3)
    doses = [10, 100, 1000]
    rows = [("hexagonal", 10.0), ("square", 40.0)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for r, (name, theta) in enumerate(rows):
        phase = PHASES[name]
        for c, dose in enumerate(doses):
            pat = simulate_pattern(phase, theta, det, dose, rng)
            show_pattern(axes[r, c], pat, det, title=f"{name}, {dose} e-/pattern")
        clean = clean_pattern(phase, theta, det) * 1e4
        show_pattern(axes[r, 3], clean, det, title=f"{name}, clean")
    fig.suptitle(
        "simulated nanodiffraction patterns across dose (log intensity scale)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIG / "pattern_gallery.png", dpi=150)
    plt.close(fig)


def template_tuning_figure() -> None:
    payload = _load("template_tuning")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for refine, color in [(True, "#1f77b4"), (False, "#7f7f7f")]:
        rows = [s for s in payload["settings"] if s["refine"] == refine]
        steps = [s["step_deg"] for s in rows]
        mae = [s["aggregate"]["interior_orientation_mae_deg"]["mean"] for s in rows]
        ms = [s["aggregate"]["seconds_per_pattern"]["mean"] * 1e3 for s in rows]
        label = "with parabolic refinement" if refine else "grid snap only"
        axes[0].plot(steps, mae, "o-", color=color, label=label)
        axes[1].plot(steps, ms, "o-", color=color, label=label)
    cnn = payload["cnn_reference"]["aggregate"]
    axes[0].axhline(
        cnn["interior_orientation_mae_deg"]["mean"], color="#d62728", ls="--", label="CNN"
    )
    axes[1].axhline(cnn["seconds_per_pattern"]["mean"] * 1e3, color="#d62728", ls="--", label="CNN")
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.25, 0.35, 0.5, 0.7, 1.0])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for ax, ylabel in [(axes[0], "orientation MAE (deg)"), (axes[1], "cost (ms per pattern)")]:
        ax.set_xlabel("library step (deg)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 5.0])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("template-library tuning: accuracy versus per-pattern cost", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "template_tuning.png", dpi=150)
    plt.close(fig)


def clustering_figure() -> None:
    payload = _load("clustering")
    cfg = payload["config"]
    params = _scene_params(cfg)
    det = _detector(cfg)
    seed = _condition_seed(cfg["seed"], 0, 0)
    scan = simulate_scene_scan(params, det, seed)
    res = cluster_mod.cluster_grains(scan.data, k=int(params.n_grains), seed=seed)
    ari = grain_agreement(res.labels, scan.grain_id)["ari"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].imshow(res.labels, cmap="tab20", interpolation="nearest")
    axes[0].set_title(f"k-means clusters, k={int(params.n_grains)} (ARI {ari:.2f})", fontsize=9)
    axes[1].imshow(scan.grain_id, cmap="tab20", interpolation="nearest")
    axes[1].set_title("ground-truth grains", fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks([]), ax.set_yticks([])
    k_scores = payload["per_scan"][0]["k_scores"]
    ks = sorted(int(k) for k in k_scores)
    axes[2].plot(ks, [k_scores[str(k)] for k in ks], "o-", color="#1f77b4")
    axes[2].axvline(int(params.n_grains), color="k", ls=":", label="true grain count")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("silhouette score")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "unsupervised grain segmentation (scan 0 of the clustering benchmark)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIG / "clustering.png", dpi=150)
    plt.close(fig)


def training_figure() -> None:
    with open(RES / "training_history.json", encoding="utf-8") as fh:
        hist = json.load(fh)
    steps = [s["step"] for s in hist["steps"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    axes[0].plot(steps, [s["train_loss"] for s in hist["steps"]], color="#1f77b4")
    axes[0].set_ylabel("training loss")
    axes[1].plot(steps, [s["val_phase_accuracy"] for s in hist["steps"]], color="#2ca02c")
    axes[1].set_ylabel("val phase accuracy")
    axes[2].plot(steps, [s["val_angle_mae_deg"] for s in hist["steps"]], color="#d62728")
    axes[2].set_ylabel("val angle MAE (deg)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "CNN training curves; validation on the randomised stream (doses 10-3000, "
        "mixtures included)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "training.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    pattern_gallery()
    print("pattern_gallery.png")
    hero_figure()
    print("hero.png")
    _sweep_figure("dose_sweep", "dose (electrons per pattern)", "dose_sweep.png", logx=True)
    print("dose_sweep.png")
    _sweep_figure("grain_sweep", "number of grains (48x48 scan)", "grain_sweep.png")
    print("grain_sweep.png")
    _sweep_figure("mosaic_sweep", "intra-grain orientation spread sigma (deg)", "mosaic_sweep.png")
    print("mosaic_sweep.png")
    _sweep_figure("resolution_sweep", "pattern resolution (pixels)", "resolution_sweep.png")
    print("resolution_sweep.png")
    template_tuning_figure()
    print("template_tuning.png")
    clustering_figure()
    print("clustering.png")
    training_figure()
    print("training.png")


if __name__ == "__main__":
    main()
