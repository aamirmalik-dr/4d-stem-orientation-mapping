import numpy as np
import pytest
import torch

from orient4d.net import (
    OrientPhaseNet,
    angle_targets,
    decode_angle,
    load_model,
    net_input,
    save_model,
)


def test_forward_shapes():
    model = OrientPhaseNet(width=8)
    x = torch.zeros(5, 1, 64, 64)
    logits, angles = model(x)
    assert logits.shape == (5, 2)
    assert angles.shape == (5, 4)


def test_net_input_normalisation_and_resize():
    rng = np.random.default_rng(0)
    pats = rng.poisson(3.0, size=(2, 32, 32)).astype(np.float32)
    t = net_input(pats)
    assert t.shape == (2, 1, 64, 64)
    assert torch.isfinite(t).all()
    # Dose invariance: scaling counts by 100x changes the input only slightly
    # (Poisson realisations differ, but a pure rescale is exactly invariant).
    t2 = net_input(pats * 100.0)
    assert torch.allclose(t, t2, atol=1e-5)


def test_angle_targets_decode_roundtrip():
    theta = np.array([10.0, 55.0, 3.0, 88.0])
    phase = np.array([0, 0, 1, 1])
    targets = angle_targets(theta, phase)
    out = np.zeros((4, 4), dtype=np.float32)
    rows = np.arange(4)
    out[rows, 2 * phase] = targets[:, 0]
    out[rows, 2 * phase + 1] = targets[:, 1]
    decoded = decode_angle(out, phase)
    assert decoded == pytest.approx(theta, abs=1e-4)


def test_save_load_roundtrip(tmp_path):
    model = OrientPhaseNet(width=8)
    path = str(tmp_path / "m.pt")
    save_model(model, path)
    loaded = load_model(path)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        a1, b1 = model(x)
        a2, b2 = loaded(x)
    assert torch.allclose(a1, a2)
    assert torch.allclose(b1, b2)
