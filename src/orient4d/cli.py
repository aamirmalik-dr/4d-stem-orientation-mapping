"""Command-line interface: simulate, map, virtual, cluster, benchmark, train, demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import benchmark as benchmark_mod
from . import cluster as cluster_mod
from . import io as io_mod
from . import net as net_mod
from . import plots
from . import template as template_mod
from . import train as train_mod
from .metrics import grain_agreement, orientation_phase_metrics
from .sim import DetectorGeometry, Scan4D, SceneParams, simulate_scene_scan


def _load_any(path: str) -> Scan4D:
    """Load a native scan, or wrap external 4D data with empty ground truth."""
    try:
        return io_mod.load_scan(path)
    except (KeyError, ValueError):
        data, det = io_mod.load_external(path)
        h, w = data.shape[:2]
        return Scan4D(
            data=data,
            grain_id=np.full((h, w), -1, dtype=np.int16),
            phase_id=np.full((h, w), -1, dtype=np.int8),
            theta=np.zeros((h, w), dtype=np.float32),
            purity=np.zeros((h, w), dtype=np.float32),
            det=det,
            params=SceneParams(scan_shape=(h, w)),
        )


def _has_truth(scan: Scan4D) -> bool:
    return bool(np.any(scan.phase_id >= 0))


def _print_metrics(scan: Scan4D, phase_map: np.ndarray, theta_map: np.ndarray) -> None:
    m = orientation_phase_metrics(theta_map, phase_map, scan.theta, scan.phase_id)
    mi = orientation_phase_metrics(
        theta_map, phase_map, scan.theta, scan.phase_id, mask=scan.purity >= 0.95
    )
    print(
        f"  phase accuracy        {m['phase_accuracy']:.3f} (interior {mi['phase_accuracy']:.3f})"
    )
    print(
        f"  orientation MAE       {m['orientation_mae_deg']:.3f} deg "
        f"(interior {mi['orientation_mae_deg']:.3f})"
    )
    print(
        f"  within 1 deg          {m['frac_within_1deg']:.3f} "
        f"(interior {mi['frac_within_1deg']:.3f})"
    )


def cmd_simulate(args: argparse.Namespace) -> int:
    params = SceneParams(
        scan_shape=(args.scan, args.scan),
        n_grains=args.grains,
        dose=args.dose,
        mosaic_sigma=args.mosaic,
    )
    det = DetectorGeometry(n_px=args.pattern_px)
    scan = simulate_scene_scan(params, det, args.seed)
    io_mod.save_scan(scan, args.out)
    print(f"wrote {args.out}: {scan.data.shape} counts, {params.n_grains} grains")
    if args.figure:
        plots.scene_figure(scan, args.figure)
        print(f"wrote {args.figure}")
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    scan = _load_any(args.scan)
    if args.method == "template":
        lib = template_mod.build_library(scan.det, step_deg=args.step)
        phase_map, theta_map, res = template_mod.map_scan(scan.data, lib)
        print(
            f"template matching: {lib.n_templates} templates, "
            f"{res.seconds_per_pattern * 1e3:.2f} ms/pattern"
        )
    else:
        model = net_mod.load_model(args.model)
        phase_map, theta_map = net_mod.map_scan(model, scan.data)
        print("cnn mapping done")
    if _has_truth(scan):
        _print_metrics(scan, phase_map, theta_map)
    if args.out:
        np.savez_compressed(args.out, phase_map=phase_map, theta_map=theta_map)
        print(f"wrote {args.out}")
    if args.figure:
        plots.map_figure(scan, phase_map, theta_map, args.figure, method_name=args.method)
        print(f"wrote {args.figure}")
    return 0


def cmd_virtual(args: argparse.Namespace) -> int:
    scan = _load_any(args.scan)
    plots.virtual_figure(scan, args.figure)
    print(f"wrote {args.figure}")
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    scan = _load_any(args.scan)
    k = None if args.k == "auto" else int(args.k)
    res = cluster_mod.cluster_grains(scan.data, k=k, seed=args.seed)
    print(f"clustered into k={res.k} grains")
    ari = float("nan")
    if _has_truth(scan):
        agree = grain_agreement(res.labels, scan.grain_id)
        ari = agree["ari"]
        print(f"  ARI {agree['ari']:.3f}  NMI {agree['nmi']:.3f}")
    if args.figure:
        plots.cluster_figure(scan, res.labels, ari, args.figure)
        print(f"wrote {args.figure}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    payload = benchmark_mod.run_config(args.config)
    out = benchmark_mod.load_config(args.config)["output"]
    print(f"benchmark '{payload['config'].get('name', args.config)}' done -> {out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    settings = train_mod.TrainSettings(steps=args.steps, seed=args.seed)
    model, history = train_mod.train(settings, width=args.width)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    net_mod.save_model(model, args.out)
    print(f"wrote {args.out}")
    if args.history:
        Path(args.history).parent.mkdir(parents=True, exist_ok=True)
        train_mod.save_history(history, args.history)
        print(f"wrote {args.history}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    sample = Path(args.sample)
    if not sample.exists():
        print(f"sample scan not found: {sample} (run from the repository root)", file=sys.stderr)
        return 1
    scan = io_mod.load_scan(sample)
    h, w = scan.data.shape[:2]
    print(f"loaded {sample}: {h}x{w} scan, {scan.det.n_px} px patterns")

    print("\ntemplate matching (step 0.5 deg):")
    lib = template_mod.build_library(scan.det, step_deg=0.5)
    phase_t, theta_t, _ = template_mod.map_scan(scan.data, lib)
    _print_metrics(scan, phase_t, theta_t)

    model_path = Path(args.model)
    if model_path.exists():
        print("\ncnn:")
        model = net_mod.load_model(str(model_path))
        phase_c, theta_c = net_mod.map_scan(model, scan.data)
        _print_metrics(scan, phase_c, theta_c)
    else:
        print(f"\ncnn skipped (no model at {model_path})")
        phase_c, theta_c = phase_t, theta_t

    print("\nclustering (k = true grain count):")
    res = cluster_mod.cluster_grains(scan.data, k=int(scan.params.n_grains), seed=0)
    agree = grain_agreement(res.labels, scan.grain_id)
    print(f"  ARI {agree['ari']:.3f}  NMI {agree['nmi']:.3f}")

    if args.figure:
        plots.map_figure(scan, phase_c, theta_c, args.figure, method_name="cnn")
        print(f"\nwrote {args.figure}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orient4d",
        description="4D-STEM orientation and phase mapping: simulate, map, cluster, benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate a polycrystalline 4D-STEM scan")
    p.add_argument("--scan", type=int, default=48, help="scan side length in probe positions")
    p.add_argument("--grains", type=int, default=8)
    p.add_argument("--dose", type=float, default=300.0, help="electrons per pattern")
    p.add_argument("--mosaic", type=float, default=0.3, help="intra-grain spread (deg)")
    p.add_argument("--pattern-px", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--figure", default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("map", help="orientation + phase map a scan")
    p.add_argument("scan")
    p.add_argument("--method", choices=["template", "cnn"], default="template")
    p.add_argument("--step", type=float, default=0.5, help="template library step (deg)")
    p.add_argument("--model", default="models/orientnet.pt")
    p.add_argument("--out", default=None, help="write maps to this .npz")
    p.add_argument("--figure", default=None)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("virtual", help="virtual bright/dark-field images")
    p.add_argument("scan")
    p.add_argument("--figure", required=True)
    p.set_defaults(func=cmd_virtual)

    p = sub.add_parser("cluster", help="unsupervised grain clustering")
    p.add_argument("scan")
    p.add_argument("--k", default="auto", help="cluster count or 'auto'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figure", default=None)
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("benchmark", help="run a YAML benchmark config")
    p.add_argument("config")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("train", help="train the orientation + phase CNN")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--out", default="models/orientnet.pt")
    p.add_argument("--history", default="results/training_history.json")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("demo", help="run all methods on the committed sample scan")
    p.add_argument("--sample", default="data/sample/scan_32.npz")
    p.add_argument("--model", default="models/orientnet.pt")
    p.add_argument("--figure", default="demo_map.png")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
