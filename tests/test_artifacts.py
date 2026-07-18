"""Guards against drift between committed artifacts and the code."""

from pathlib import Path

import numpy as np
import pytest

from orient4d.io import load_scan
from orient4d.net import load_model

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "orientnet.pt"
SAMPLE = ROOT / "data" / "sample" / "scan_32.npz"


@pytest.mark.skipif(not MODEL.exists(), reason="committed model not present")
def test_committed_model_loads_and_matches_spec():
    model = load_model(str(MODEL))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 157014  # width 24, as documented in the model card


@pytest.mark.skipif(not SAMPLE.exists(), reason="committed sample not present")
def test_committed_sample_matches_documented_generation():
    scan = load_scan(SAMPLE)
    assert scan.data.shape == (32, 32, 64, 64)
    assert scan.params.n_grains == 6
    assert scan.params.dose == 300.0
    assert len(np.unique(scan.grain_id)) == 6


@pytest.mark.skipif(
    not (MODEL.exists() and SAMPLE.exists()), reason="committed artifacts not present"
)
def test_committed_model_performs_on_committed_sample():
    """The committed CNN must clearly beat chance on the committed sample."""
    from orient4d.metrics import orientation_phase_metrics
    from orient4d.net import map_scan

    scan = load_scan(SAMPLE)
    model = load_model(str(MODEL))
    phase_map, theta_map = map_scan(model, scan.data)
    m = orientation_phase_metrics(
        theta_map, phase_map, scan.theta, scan.phase_id, mask=scan.purity >= 0.95
    )
    assert m["phase_accuracy"] > 0.9
    # Random angles would give MAE ~ fold/4 (15 or 22.5 deg).
    assert m["orientation_mae_deg"] < 5.0
