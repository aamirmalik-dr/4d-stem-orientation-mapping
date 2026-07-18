"""Saving and loading scans, plus a bring-your-own-data loader.

The native format is a compressed ``.npz`` holding the count data (uint16
when it fits, float32 otherwise), the ground-truth maps, and a JSON metadata
string with the detector geometry and scene parameters, so a committed sample
is fully self-describing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .sim import DetectorGeometry, Scan4D, SceneParams


def save_scan(scan: Scan4D, path: str | Path) -> None:
    """Write a scan and its ground truth to a compressed .npz file."""
    data = scan.data
    if data.max() <= 65535 and np.allclose(data, np.round(data)):
        data = data.astype(np.uint16)
    meta = {
        "det": asdict(scan.det),
        "params": asdict(scan.params),
        "format": "orient4d-scan-v1",
    }
    np.savez_compressed(
        path,
        data=data,
        grain_id=scan.grain_id,
        phase_id=scan.phase_id,
        theta=scan.theta,
        purity=scan.purity,
        meta=np.array(json.dumps(meta)),
    )


def load_scan(path: str | Path) -> Scan4D:
    """Load a scan written by :func:`save_scan`."""
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta"]))
        det_kwargs = meta["det"]
        params_kwargs = meta["params"]
        params_kwargs["scan_shape"] = tuple(params_kwargs["scan_shape"])
        params_kwargs["phase_names"] = tuple(params_kwargs["phase_names"])
        return Scan4D(
            data=npz["data"].astype(np.float32),
            grain_id=npz["grain_id"],
            phase_id=npz["phase_id"],
            theta=npz["theta"],
            purity=npz["purity"],
            det=DetectorGeometry(**det_kwargs),
            params=SceneParams(**params_kwargs),
        )


def load_external(path: str | Path, k_max: float = 1.35) -> tuple[np.ndarray, DetectorGeometry]:
    """Load user-supplied 4D data from a .npy or .npz file.

    Accepts a 4D array of shape (scan_rows, scan_cols, det_rows, det_cols).
    An ``.npz`` must hold the array under the key ``data``. Patterns with a
    non-square detector are center-cropped to square. See ``data/README.md``
    for converting vendor formats with HyperSpy or py4DSTEM.

    Args:
        path: Path to the .npy or .npz file.
        k_max: Detector half-width in reciprocal angstrom for your camera
            length; needed for virtual-image radii and template libraries.

    Returns:
        Tuple of (data as float32, inferred DetectorGeometry).
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            if "data" not in npz:
                raise ValueError(f"{path} has no 'data' array; keys: {list(npz.keys())}")
            arr = npz["data"]
    else:
        arr = np.load(path, allow_pickle=False)
    if arr.ndim != 4:
        raise ValueError(f"expected a 4D array (rows, cols, det_y, det_x); got shape {arr.shape}")
    if arr.shape[2] != arr.shape[3]:
        side = min(arr.shape[2], arr.shape[3])
        y0 = (arr.shape[2] - side) // 2
        x0 = (arr.shape[3] - side) // 2
        arr = arr[:, :, y0 : y0 + side, x0 : x0 + side]
    det = DetectorGeometry(n_px=int(arr.shape[2]), k_max=float(k_max))
    return arr.astype(np.float32), det
