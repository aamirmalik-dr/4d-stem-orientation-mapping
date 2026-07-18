import json

import yaml

from orient4d.benchmark import run_config

BASE_SCENE = {
    "scan_shape": [10, 10],
    "n_grains": 3,
    "dose": 300,
    "mosaic_sigma": 0.3,
}


def _write_cfg(tmp_path, cfg):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    path = cfg_dir / f"{cfg['name']}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def test_sweep_mode(tmp_path):
    cfg = {
        "mode": "sweep",
        "name": "tiny_sweep",
        "seed": 5,
        "scans_per_condition": 1,
        "scene": dict(BASE_SCENE),
        "detector": {"n_px": 32},
        "sweep": {"parameter": "dose", "values": [100, 1000]},
        "methods": ["template"],
        "template": {"step_deg": 3.0, "refine": True},
        "output": "results/tiny_sweep.json",
    }
    path = _write_cfg(tmp_path, cfg)
    payload = run_config(path, base_dir=tmp_path)
    assert len(payload["conditions"]) == 2
    agg = payload["conditions"][1]["methods"]["template"]["aggregate"]
    assert 0.0 <= agg["phase_accuracy"]["mean"] <= 1.0
    assert (tmp_path / "results/tiny_sweep.json").exists()
    with open(tmp_path / "results/tiny_sweep.json", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["config"]["name"] == "tiny_sweep"


def test_sweep_is_deterministic_given_seed(tmp_path):
    cfg = {
        "mode": "sweep",
        "name": "det_check",
        "seed": 11,
        "scans_per_condition": 1,
        "scene": dict(BASE_SCENE),
        "detector": {"n_px": 32},
        "sweep": {"parameter": "dose", "values": [300]},
        "methods": ["template"],
        "template": {"step_deg": 3.0, "refine": True},
        "output": "results/det_check.json",
    }
    path = _write_cfg(tmp_path, cfg)
    a = run_config(path, base_dir=tmp_path)
    b = run_config(path, base_dir=tmp_path)
    ka = a["conditions"][0]["methods"]["template"]["per_scan"][0]
    kb = b["conditions"][0]["methods"]["template"]["per_scan"][0]
    for key in ("phase_accuracy", "orientation_mae_deg"):
        assert ka[key] == kb[key]


def test_clustering_mode(tmp_path):
    cfg = {
        "mode": "clustering",
        "name": "tiny_cluster",
        "seed": 7,
        "scans_per_condition": 1,
        "scene": dict(BASE_SCENE),
        "detector": {"n_px": 32},
        "clustering": {"k_range": [2, 4]},
        "output": "results/tiny_cluster.json",
    }
    path = _write_cfg(tmp_path, cfg)
    payload = run_config(path, base_dir=tmp_path)
    row = payload["per_scan"][0]
    assert -1.0 <= row["ari"] <= 1.0
    assert row["selected_k"] in (2, 3, 4)


def test_template_tuning_mode(tmp_path):
    cfg = {
        "mode": "template_tuning",
        "name": "tiny_tuning",
        "seed": 9,
        "scans_per_condition": 1,
        "scene": dict(BASE_SCENE),
        "detector": {"n_px": 32},
        "tuning": {"steps_deg": [5.0, 2.0], "refine": [True]},
        "cnn": {"model": "models/orientnet.pt"},
        "output": "results/tiny_tuning.json",
    }
    # No CNN model in tmp_path: template settings must still run, so patch the
    # config to skip the CNN reference by monkeypatching is avoided; instead we
    # train nothing and expect failure only at the CNN step. Simplest honest
    # check: run with a real tiny model.
    import orient4d.net as net_mod
    from orient4d.train import TrainSettings, train

    model, _ = train(TrainSettings(steps=2, batch_size=4, val_every=2, val_size=8, seed=0), width=8)
    (tmp_path / "models").mkdir(exist_ok=True)
    net_mod.save_model(model, str(tmp_path / "models/orientnet.pt"))
    path = _write_cfg(tmp_path, cfg)
    payload = run_config(path, base_dir=tmp_path)
    assert len(payload["settings"]) == 2
    assert payload["settings"][0]["n_templates"] < payload["settings"][1]["n_templates"]
    assert "aggregate" in payload["cnn_reference"]


def test_library_detector_mismatch_override(tmp_path):
    cfg = {
        "mode": "compare",
        "name": "tiny_mismatch",
        "seed": 13,
        "scans_per_condition": 1,
        "scene": dict(BASE_SCENE),
        "detector": {"n_px": 32, "direct_fraction": 0.4},
        "methods": ["template"],
        "template": {"step_deg": 3.0, "library_detector": {"direct_fraction": 0.25}},
        "clustering": {"enabled": False},
        "output": "results/tiny_mismatch.json",
    }
    path = _write_cfg(tmp_path, cfg)
    payload = run_config(path, base_dir=tmp_path)
    agg = payload["methods"]["template"]["aggregate"]
    assert 0.0 <= agg["phase_accuracy"]["mean"] <= 1.0
