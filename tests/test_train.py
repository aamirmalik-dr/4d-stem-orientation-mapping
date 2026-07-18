import numpy as np

from orient4d.sim import PHASES, DetectorGeometry
from orient4d.train import TrainSettings, sample_batch, train

DET = DetectorGeometry(n_px=32)


def test_sample_batch_shapes_and_ranges():
    settings = TrainSettings(dose_range=(50.0, 200.0))
    rng = np.random.default_rng(0)
    pats, pids, thetas = sample_batch(settings, DET, rng, 16)
    assert pats.shape == (16, 32, 32)
    assert np.all(pats >= 0)
    assert set(np.unique(pids)).issubset({0, 1})
    folds = np.array([PHASES["hexagonal"].fold, PHASES["square"].fold])
    assert np.all(thetas >= 0)
    assert np.all(thetas < folds[pids])


def test_tiny_training_run_executes():
    settings = TrainSettings(steps=4, batch_size=8, val_every=2, val_size=16, seed=1)
    model, history = train(settings, det=DET, width=8, progress=False)
    assert len(history["steps"]) >= 2
    last = history["steps"][-1]
    assert np.isfinite(last["train_loss"])
    assert 0.0 <= last["val_phase_accuracy"] <= 1.0
