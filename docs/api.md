# Python API

Everything below is runnable as-is from the repository root in an
environment with the package installed (`pip install -e .`).

## Simulate a polycrystalline scan

```python
from orient4d import SceneParams, DetectorGeometry, simulate_scene_scan

params = SceneParams(
    scan_shape=(48, 48),   # probe raster
    n_grains=8,            # Voronoi grains
    dose=300.0,            # electrons per diffraction pattern
    mosaic_sigma=0.3,      # intra-grain orientation spread, degrees
)
scan = simulate_scene_scan(params, DetectorGeometry(), seed=0)
print(scan.data.shape)     # (48, 48, 64, 64) Poisson counts
print(scan.theta[0, 0])    # exact local orientation at position (0, 0)
print(scan.purity.mean())  # probe-weight fraction of the dominant grain
```

`Scan4D` carries the exact ground truth per probe position: `grain_id`,
`phase_id` (index into `orient4d.PHASE_NAMES`), `theta` (degrees, reduced to
the phase's symmetry fold) and `purity` (1.0 in a grain interior, lower on
boundaries where the probe straddles two grains).

## Single patterns

```python
from orient4d import PHASES, DetectorGeometry, clean_pattern, simulate_pattern
import numpy as np

det = DetectorGeometry(n_px=64, k_max=1.35)
expected = clean_pattern(PHASES["hexagonal"], theta_deg=12.0, det=det)  # sums to 1
noisy = simulate_pattern(
    PHASES["square"], 40.0, det, dose=100.0, rng=np.random.default_rng(0)
)
```

## Virtual imaging

```python
from orient4d import bright_field, annular_dark_field, spot_dark_field

vbf = bright_field(scan.data, scan.det, radius=0.12)          # (48, 48)
vadf = annular_dark_field(scan.data, scan.det, 0.30, 1.30)
# Grain-selective: aperture on one Bragg position (ky, kx in 1/angstrom)
sel = spot_dark_field(scan.data, scan.det, center_k=(0.0, 0.47), radius=0.08)
```

## Template matching

```python
from orient4d import build_library
from orient4d.template import map_scan

lib = build_library(scan.det, step_deg=0.5)      # 300 templates for 2 phases
phase_map, theta_map, res = map_scan(scan.data, lib)
print(res.seconds_per_pattern * 1e3, "ms per pattern")
```

## CNN inference and training

```python
from orient4d import load_model
from orient4d.net import map_scan as cnn_map_scan

model = load_model("models/orientnet.pt")
phase_map, theta_map = cnn_map_scan(model, scan.data)
```

Retraining from scratch (about 15 minutes on CPU; every sample is simulated
on the fly with domain randomisation):

```python
from orient4d.train import TrainSettings, train
from orient4d import save_model

model, history = train(TrainSettings(steps=6000))
save_model(model, "models/orientnet.pt")
```

## Unsupervised grain clustering

```python
from orient4d import cluster_grains, grain_agreement

res = cluster_grains(scan.data, k=8)             # or k=None for silhouette selection
print(grain_agreement(res.labels, scan.grain_id))  # {'ari': ..., 'nmi': ...}
```

## Scoring

```python
from orient4d import orientation_phase_metrics

m = orientation_phase_metrics(
    theta_map, phase_map, scan.theta, scan.phase_id, mask=scan.purity >= 0.95
)
print(m["orientation_mae_deg"], m["phase_accuracy"])
```

Orientation error is symmetry-aware: predictions and truth are compared
modulo each phase's fold (60 degrees hexagonal, 90 square), and only pixels
whose phase was predicted correctly enter the angular statistics; phase
errors are reported separately through `phase_accuracy`.

## Benchmarks

```python
from orient4d.benchmark import run_config

payload = run_config("configs/dose_sweep.yaml")   # writes results/dose_sweep.json
```

Every committed number regenerates from the YAML configs in `configs/`; the
seeds are fixed in the configs, and condition/scan seeds derive
deterministically from them.

## File I/O

```python
from orient4d import save_scan, load_scan, load_external

save_scan(scan, "data/full/my_scan.npz")   # counts + ground truth + geometry
scan2 = load_scan("data/full/my_scan.npz")
data, det = load_external("external.npy", k_max=1.4)  # bring-your-own data
```
