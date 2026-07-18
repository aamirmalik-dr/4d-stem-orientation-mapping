import numpy as np
import pytest

from orient4d.metrics import angular_error_deg, grain_agreement, orientation_phase_metrics


def test_angular_error_wraps_at_fold():
    assert angular_error_deg(0.5, 59.5, 60.0) == pytest.approx(1.0)
    assert angular_error_deg(89.5, 0.5, 90.0) == pytest.approx(1.0)
    assert angular_error_deg(30.0, 30.0, 60.0) == pytest.approx(0.0)
    assert angular_error_deg(0.0, 30.0, 60.0) == pytest.approx(30.0)


def test_angular_error_broadcasts_folds():
    err = angular_error_deg(np.array([59.0, 89.0]), np.array([1.0, 1.0]), np.array([60.0, 90.0]))
    assert err == pytest.approx([2.0, 2.0])


def test_orientation_metrics_scores_only_correct_phase():
    true_theta = np.array([10.0, 20.0, 30.0])
    true_phase = np.array([0, 0, 1])
    pred_theta = np.array([11.0, 20.0, 0.0])
    pred_phase = np.array([0, 1, 1])  # middle pixel is a phase error
    m = orientation_phase_metrics(pred_theta, pred_phase, true_theta, true_phase)
    assert m["phase_accuracy"] == pytest.approx(2.0 / 3.0)
    assert m["n_scored"] == 2
    assert m["orientation_mae_deg"] == pytest.approx((1.0 + 30.0) / 2.0)


def test_orientation_metrics_mask():
    m = orientation_phase_metrics(
        np.array([1.0, 50.0]),
        np.array([0, 0]),
        np.array([1.0, 10.0]),
        np.array([0, 0]),
        mask=np.array([True, False]),
    )
    assert m["n_selected"] == 1
    assert m["orientation_mae_deg"] == pytest.approx(0.0)


def test_grain_agreement_perfect_and_shuffled():
    true = np.array([0, 0, 1, 1, 2, 2])
    relabelled = np.array([5, 5, 3, 3, 9, 9])
    assert grain_agreement(relabelled, true)["ari"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    random_labels = rng.integers(0, 3, size=600)
    true_big = np.repeat(np.arange(3), 200)
    assert abs(grain_agreement(random_labels, true_big)["ari"]) < 0.05
