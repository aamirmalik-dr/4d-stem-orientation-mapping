"""Condense the benchmark JSONs into results/metrics.json (headline summary)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _load(name: str) -> dict:
    with open(RES / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def _round(x: float, nd: int = 4) -> float:
    return float(round(x, nd))


def main() -> None:
    compare = _load("compare")
    dose = _load("dose_sweep")
    tuning = _load("template_tuning")
    clustering = _load("clustering")

    headline: dict = {"operating_point": compare["config"]["scene"]}
    for method in ("template", "cnn"):
        agg = compare["methods"][method]["aggregate"]
        headline[method] = {
            "orientation_mae_deg": _round(agg["orientation_mae_deg"]["mean"]),
            "orientation_mae_deg_std": _round(agg["orientation_mae_deg"]["std"]),
            "interior_orientation_mae_deg": _round(agg["interior_orientation_mae_deg"]["mean"]),
            "phase_accuracy": _round(agg["phase_accuracy"]["mean"]),
            "interior_phase_accuracy": _round(agg["interior_phase_accuracy"]["mean"]),
            "frac_within_1deg": _round(agg["frac_within_1deg"]["mean"]),
            "ms_per_pattern": _round(agg["seconds_per_pattern"]["mean"] * 1e3, 3),
        }
    cl = compare["clustering"]["aggregate"]
    headline["clustering"] = {
        "ari_at_true_k": _round(cl["ari"]["mean"]),
        "ari_std": _round(cl["ari"]["std"]),
        "nmi": _round(cl["nmi"]["mean"]),
    }
    cl2 = clustering["aggregate"]
    headline["clustering"]["ari_at_selected_k"] = _round(cl2["ari_selected"]["mean"])
    headline["clustering"]["selected_k_mean"] = _round(cl2["selected_k"]["mean"], 2)

    lowest = dose["conditions"][0]
    headline["lowest_dose"] = {"dose": lowest["value"]}
    for method in ("template", "cnn"):
        agg = lowest["methods"][method]["aggregate"]
        headline["lowest_dose"][method] = {
            "interior_orientation_mae_deg": _round(agg["interior_orientation_mae_deg"]["mean"]),
            "interior_phase_accuracy": _round(agg["interior_phase_accuracy"]["mean"]),
        }

    mismatch = _load("mismatch")
    headline["forward_model_mismatch"] = {}
    for method in ("template", "cnn"):
        agg = mismatch["methods"][method]["aggregate"]
        headline["forward_model_mismatch"][method] = {
            "interior_orientation_mae_deg": _round(agg["interior_orientation_mae_deg"]["mean"]),
            "interior_phase_accuracy": _round(agg["interior_phase_accuracy"]["mean"]),
        }

    best = min(
        tuning["settings"],
        key=lambda s: s["aggregate"]["interior_orientation_mae_deg"]["mean"],
    )
    headline["best_template_setting"] = {
        "step_deg": best["step_deg"],
        "refine": best["refine"],
        "n_templates": best["n_templates"],
        "interior_orientation_mae_deg": _round(
            best["aggregate"]["interior_orientation_mae_deg"]["mean"]
        ),
        "ms_per_pattern": _round(best["aggregate"]["seconds_per_pattern"]["mean"] * 1e3, 3),
    }

    with open(RES / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(headline, fh, indent=2)
    print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
