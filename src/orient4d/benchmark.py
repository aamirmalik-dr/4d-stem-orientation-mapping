"""Config-driven benchmark harness with fixed seeds.

Every committed number in this repository regenerates from a YAML config via
``orient4d benchmark <config>``. Modes:

- ``compare``: one operating point, all methods (template, cnn, clustering),
  several independent scans, aggregate mean and std plus per-pattern timing.
- ``sweep``: one scene parameter (dose, n_grains, mosaic_sigma, pattern_px)
  swept over a value list, template and CNN scored at each condition.
- ``template_tuning``: library step and refinement swept on fixed scans, so
  the classical baseline is scored at its best operating point (accuracy
  versus per-pattern cost), with a CNN reference row on the same scans.
- ``clustering``: unsupervised grain recovery, ARI/NMI at the true number of
  grains and at a silhouette-selected number.

Seeds derive deterministically from the config seed, the condition index and
the scan index, so re-running a config reproduces its JSON byte for byte
(up to wall-clock timing fields).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from . import cluster as cluster_mod
from . import net as net_mod
from . import template as template_mod
from .metrics import grain_agreement, orientation_phase_metrics
from .sim import DetectorGeometry, Scan4D, SceneParams, simulate_scene_scan

_AGG_KEYS = (
    "phase_accuracy",
    "orientation_mae_deg",
    "orientation_median_deg",
    "frac_within_1deg",
    "seconds_per_pattern",
    "interior_phase_accuracy",
    "interior_orientation_mae_deg",
    "interior_orientation_median_deg",
    "interior_frac_within_1deg",
    "ari",
    "nmi",
    "ari_selected",
    "selected_k",
)


def load_config(path: str | Path) -> dict:
    """Read a YAML benchmark config."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _scene_params(cfg: dict) -> SceneParams:
    scene = dict(cfg.get("scene", {}))
    if "scan_shape" in scene:
        scene["scan_shape"] = tuple(scene["scan_shape"])
    if "phase_names" in scene:
        scene["phase_names"] = tuple(scene["phase_names"])
    return SceneParams(**scene)


def _detector(cfg: dict) -> DetectorGeometry:
    return DetectorGeometry(**cfg.get("detector", {}))


def _condition_seed(base_seed: int, cond_idx: int, scan_idx: int) -> int:
    return int(base_seed) + 1000 * int(cond_idx) + int(scan_idx)


def _apply_condition(
    params: SceneParams, det: DetectorGeometry, parameter: str, value
) -> tuple[SceneParams, DetectorGeometry]:
    if parameter == "dose":
        return replace(params, dose=float(value)), det
    if parameter == "n_grains":
        return replace(params, n_grains=int(value)), det
    if parameter == "mosaic_sigma":
        return replace(params, mosaic_sigma=float(value)), det
    if parameter == "pattern_px":
        return params, replace(det, n_px=int(value))
    raise ValueError(f"unknown sweep parameter: {parameter}")


def _score_maps(
    scan: Scan4D,
    phase_map: np.ndarray,
    theta_map: np.ndarray,
    purity_threshold: float,
    seconds_per_pattern: float,
) -> dict:
    all_metrics = orientation_phase_metrics(theta_map, phase_map, scan.theta, scan.phase_id)
    interior = orientation_phase_metrics(
        theta_map, phase_map, scan.theta, scan.phase_id, mask=scan.purity >= purity_threshold
    )
    out = dict(all_metrics)
    out.update({f"interior_{k}": v for k, v in interior.items()})
    out["seconds_per_pattern"] = seconds_per_pattern
    return out


class _MethodRunner:
    """Caches template libraries and the CNN across conditions."""

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.base_dir = base_dir
        self._libraries: dict[tuple[DetectorGeometry, float], template_mod.TemplateLibrary] = {}
        self._model: net_mod.OrientPhaseNet | None = None

    def library(self, det: DetectorGeometry, step_deg: float) -> template_mod.TemplateLibrary:
        key = (det, float(step_deg))
        if key not in self._libraries:
            self._libraries[key] = template_mod.build_library(det, step_deg=step_deg)
        return self._libraries[key]

    def model(self) -> net_mod.OrientPhaseNet:
        if self._model is None:
            path = self.base_dir / self.cfg.get("cnn", {}).get("model", "models/orientnet.pt")
            self._model = net_mod.load_model(str(path))
        return self._model

    def run(self, method: str, scan: Scan4D, purity_threshold: float) -> dict:
        if method == "template":
            tcfg = self.cfg.get("template", {})
            # library_detector overrides model the practical case where the
            # template library's assumed beam/background budget is
            # miscalibrated relative to the data (forward-model mismatch).
            lib_det = replace(scan.det, **tcfg.get("library_detector", {}))
            lib = self.library(lib_det, float(tcfg.get("step_deg", 0.5)))
            phase_map, theta_map, res = template_mod.map_scan(
                scan.data, lib, refine=bool(tcfg.get("refine", True))
            )
            return _score_maps(
                scan, phase_map, theta_map, purity_threshold, res.seconds_per_pattern
            )
        if method == "cnn":
            t0 = time.perf_counter()
            phase_map, theta_map = net_mod.map_scan(self.model(), scan.data)
            spp = (time.perf_counter() - t0) / scan.data[..., 0, 0].size
            return _score_maps(scan, phase_map, theta_map, purity_threshold, spp)
        raise ValueError(f"unknown method: {method}")


def _aggregate(per_scan: list[dict]) -> dict:
    agg = {}
    for key in _AGG_KEYS:
        vals = [s[key] for s in per_scan if key in s and np.isfinite(s[key])]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def run_config(config_path: str | Path, base_dir: str | Path | None = None) -> dict:
    """Run one benchmark config and write its JSON output; returns the payload."""
    config_path = Path(config_path)
    base_dir = Path(base_dir) if base_dir else config_path.resolve().parent.parent
    cfg = load_config(config_path)
    mode = cfg["mode"]
    t0 = time.perf_counter()
    if mode == "compare":
        payload = _run_compare(cfg, base_dir)
    elif mode == "sweep":
        payload = _run_sweep(cfg, base_dir)
    elif mode == "template_tuning":
        payload = _run_template_tuning(cfg, base_dir)
    elif mode == "clustering":
        payload = _run_clustering(cfg, base_dir)
    else:
        raise ValueError(f"unknown benchmark mode: {mode}")
    payload["config"] = cfg
    payload["wall_seconds"] = time.perf_counter() - t0
    out_path = base_dir / cfg["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def _simulate(cfg: dict, params: SceneParams, det: DetectorGeometry, seed: int) -> Scan4D:
    return simulate_scene_scan(params, det, seed)


def _run_sweep(cfg: dict, base_dir: Path) -> dict:
    params0 = _scene_params(cfg)
    det0 = _detector(cfg)
    runner = _MethodRunner(cfg, base_dir)
    purity = float(cfg.get("purity_threshold", 0.95))
    n_scans = int(cfg.get("scans_per_condition", 3))
    methods = cfg.get("methods", ["template", "cnn"])
    sweep = cfg["sweep"]
    conditions = []
    for ci, value in enumerate(sweep["values"]):
        params, det = _apply_condition(params0, det0, sweep["parameter"], value)
        rows = {m: [] for m in methods}
        for si in range(n_scans):
            scan = _simulate(cfg, params, det, _condition_seed(cfg["seed"], ci, si))
            for m in methods:
                rows[m].append(runner.run(m, scan, purity))
        conditions.append(
            {
                "value": value,
                "methods": {
                    m: {"per_scan": rows[m], "aggregate": _aggregate(rows[m])} for m in methods
                },
            }
        )
    return {"mode": "sweep", "parameter": sweep["parameter"], "conditions": conditions}


def _run_compare(cfg: dict, base_dir: Path) -> dict:
    params = _scene_params(cfg)
    det = _detector(cfg)
    runner = _MethodRunner(cfg, base_dir)
    purity = float(cfg.get("purity_threshold", 0.95))
    n_scans = int(cfg.get("scans_per_condition", 3))
    methods = cfg.get("methods", ["template", "cnn"])
    rows = {m: [] for m in methods}
    cluster_rows = []
    for si in range(n_scans):
        scan = _simulate(cfg, params, det, _condition_seed(cfg["seed"], 0, si))
        for m in methods:
            rows[m].append(runner.run(m, scan, purity))
        if cfg.get("clustering", {}).get("enabled", True):
            res = cluster_mod.cluster_grains(
                scan.data, k=int(params.n_grains), seed=_condition_seed(cfg["seed"], 0, si)
            )
            cluster_rows.append(grain_agreement(res.labels, scan.grain_id))
    payload = {
        "mode": "compare",
        "methods": {m: {"per_scan": rows[m], "aggregate": _aggregate(rows[m])} for m in methods},
    }
    if cluster_rows:
        payload["clustering"] = {"per_scan": cluster_rows, "aggregate": _aggregate(cluster_rows)}
    return payload


def _run_template_tuning(cfg: dict, base_dir: Path) -> dict:
    params = _scene_params(cfg)
    det = _detector(cfg)
    runner = _MethodRunner(cfg, base_dir)
    purity = float(cfg.get("purity_threshold", 0.95))
    n_scans = int(cfg.get("scans_per_condition", 3))
    scans = [
        _simulate(cfg, params, det, _condition_seed(cfg["seed"], 0, si)) for si in range(n_scans)
    ]
    settings = []
    for step in cfg["tuning"]["steps_deg"]:
        for refine in cfg["tuning"].get("refine", [True, False]):
            lib = runner.library(det, float(step))
            per_scan = []
            for scan in scans:
                phase_map, theta_map, res = template_mod.map_scan(scan.data, lib, refine=refine)
                per_scan.append(
                    _score_maps(scan, phase_map, theta_map, purity, res.seconds_per_pattern)
                )
            settings.append(
                {
                    "step_deg": float(step),
                    "refine": bool(refine),
                    "n_templates": lib.n_templates,
                    "per_scan": per_scan,
                    "aggregate": _aggregate(per_scan),
                }
            )
    cnn_rows = [runner.run("cnn", scan, purity) for scan in scans]
    return {
        "mode": "template_tuning",
        "settings": settings,
        "cnn_reference": {"per_scan": cnn_rows, "aggregate": _aggregate(cnn_rows)},
    }


def _run_clustering(cfg: dict, base_dir: Path) -> dict:
    params = _scene_params(cfg)
    det = _detector(cfg)
    n_scans = int(cfg.get("scans_per_condition", 3))
    k_range = tuple(cfg.get("clustering", {}).get("k_range", (2, 16)))
    per_scan = []
    for si in range(n_scans):
        seed = _condition_seed(cfg["seed"], 0, si)
        scan = _simulate(cfg, params, det, seed)
        res_true = cluster_mod.cluster_grains(scan.data, k=int(params.n_grains), seed=seed)
        res_auto = cluster_mod.cluster_grains(scan.data, k=None, k_range=k_range, seed=seed)
        row = grain_agreement(res_true.labels, scan.grain_id)
        row["ari_selected"] = grain_agreement(res_auto.labels, scan.grain_id)["ari"]
        row["selected_k"] = res_auto.k
        row["k_scores"] = res_auto.k_scores
        per_scan.append(row)
    slim = [{k: v for k, v in row.items() if k != "k_scores"} for row in per_scan]
    return {"mode": "clustering", "per_scan": per_scan, "aggregate": _aggregate(slim)}
