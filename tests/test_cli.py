import numpy as np

from orient4d.cli import main
from orient4d.io import load_scan


def test_simulate_map_virtual_cluster_pipeline(tmp_path):
    scan_path = str(tmp_path / "scan.npz")
    fig1 = str(tmp_path / "scene.png")
    rc = main(
        [
            "simulate",
            "--scan",
            "10",
            "--grains",
            "3",
            "--dose",
            "300",
            "--pattern-px",
            "32",
            "--seed",
            "1",
            "--out",
            scan_path,
            "--figure",
            fig1,
        ]
    )
    assert rc == 0
    assert (tmp_path / "scene.png").exists()
    scan = load_scan(scan_path)
    assert scan.data.shape == (10, 10, 32, 32)

    maps_path = str(tmp_path / "maps.npz")
    fig2 = str(tmp_path / "map.png")
    rc = main(
        [
            "map",
            scan_path,
            "--method",
            "template",
            "--step",
            "3",
            "--out",
            maps_path,
            "--figure",
            fig2,
        ]
    )
    assert rc == 0
    with np.load(maps_path) as npz:
        assert npz["phase_map"].shape == (10, 10)
    assert (tmp_path / "map.png").exists()

    rc = main(["virtual", scan_path, "--figure", str(tmp_path / "virt.png")])
    assert rc == 0
    assert (tmp_path / "virt.png").exists()

    rc = main(["cluster", scan_path, "--k", "3", "--figure", str(tmp_path / "clus.png")])
    assert rc == 0
    assert (tmp_path / "clus.png").exists()


def test_demo_missing_sample_fails_cleanly(tmp_path):
    rc = main(["demo", "--sample", str(tmp_path / "nope.npz")])
    assert rc == 1
