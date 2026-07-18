"""Re-run every committed benchmark config, then rebuild figures and metrics.

    python scripts/run_all.py [--skip-train]

Without --skip-train this retrains the CNN first (about 15 minutes on CPU);
with it, the committed models/orientnet.pt is reused. Total benchmark time is
a few minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    "compare.yaml",
    "dose_sweep.yaml",
    "grain_sweep.yaml",
    "mosaic_sweep.yaml",
    "resolution_sweep.yaml",
    "template_tuning.yaml",
    "mismatch.yaml",
    "clustering.yaml",
]


def main() -> int:
    skip_train = "--skip-train" in sys.argv
    if not skip_train:
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "orient4d.cli",
                "train",
                "--steps",
                "6000",
                "--out",
                "models/orientnet.pt",
                "--history",
                "results/training_history.json",
            ],
            cwd=ROOT,
        )
        if rc != 0:
            return rc
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        rc = subprocess.call(
            [sys.executable, "-m", "orient4d.cli", "benchmark", f"configs/{cfg}"], cwd=ROOT
        )
        if rc != 0:
            return rc
        print(f"{cfg}: {time.perf_counter() - t0:.0f} s")
    rc = subprocess.call([sys.executable, "scripts/make_metrics.py"], cwd=ROOT)
    if rc != 0:
        return rc
    rc = subprocess.call([sys.executable, "scripts/make_figures.py"], cwd=ROOT)
    if rc != 0:
        return rc
    with open(ROOT / "results" / "metrics.json", encoding="utf-8") as fh:
        print(json.dumps(json.load(fh), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
