# Model card: orientnet.pt

## What it is

A 157,014-parameter convolutional network (`orient4d.net.OrientPhaseNet`,
width 24) that maps one 64x64 nanodiffraction pattern to (a) two phase
logits and (b) one (cos, sin) pair per phase encoding the symmetry-
multiplied in-plane orientation, cos(n theta) and sin(n theta) with n = 6
for the hexagonal phase and n = 4 for the square phase. Predicting on the
n-fold circle makes the 0/fold wrap continuous in the loss; the angle is
decoded as atan2 / n and reduced to the fold of the predicted phase.

Input normalisation: each pattern is divided by its total counts (dose
invariance), square-rooted (variance stabilisation), scaled by 8, and
bilinearly resampled to 64x64 if the detector resolution differs.

## Training data

Entirely synthetic, simulated on the fly by this repository's kinematical
forward model during training; no external data of any kind. 6000 Adam
steps of batch 64 (384k patterns, no sample ever repeated), 16.3 minutes on
CPU, seed 20260718, fixed in `results/training_history.json`. Domain
randomisation per sample: uniform phase and orientation, log-uniform dose
10-3000 electrons per pattern, camera-length jitter (sigma 2 percent),
descan jitter (sigma 0.4 px), direct-beam budget 15-35 percent, background
budget 8-25 percent, and with probability 0.3 a two-grain mixture whose
label is the dominant grain (weight 0.55-0.95), mimicking probe positions
on grain boundaries.

Loss: cross-entropy on phase + 2.0 x MSE on the true phase's (cos, sin)
pair.

## Measured performance

All numbers from the committed fixed-seed benchmarks (see RESULTS.md):

- Dose 300 operating point: interior orientation MAE 0.595 +/- 0.051 deg,
  phase accuracy 0.999, 0.228 ms per pattern on CPU.
- Dose sweep: 3.25 deg MAE at dose 10, saturating at a 0.4-0.5 deg
  regression floor above dose 300.
- Forward-model mismatch (library-miscalibration scenario): 1.14 deg MAE,
  statistically close to the template matcher's 0.99 deg.
- Validation on the randomised training distribution (doses 10-3000 with
  mixtures): phase accuracy 0.982, angle MAE 1.32 deg.

## Honest limitations

- On this in-model benchmark the CNN never beats fair-tuned template
  matching; its measured value is a constant per-pattern cost and a
  narrowed gap under forward-model mismatch, not higher accuracy. Do not
  cite this model as evidence that deep learning improves 4D-STEM
  orientation mapping accuracy.
- The angle head has a regression floor around 0.4-0.5 degrees; it does not
  reach the shot-noise limit at high dose.
- Trained purely on this simulator (kinematical, two specific phases,
  Gaussian spots, fixed k_max 1.35 1/angstrom). It will not transfer to
  experimental patterns or other crystal structures without retraining;
  for real data, retrain on your own forward model
  (`orient4d train`, `orient4d.train.TrainSettings`).
- Supports exactly the two shipped phases; a different phase list requires
  editing `orient4d.sim.PHASES` and retraining.

## Reproduce

    orient4d train --steps 6000 --out models/orientnet.pt --history results/training_history.json

Deterministic given the seed and CPU (PyTorch 2.13, seed 20260718).
File size 634 KB, saved as a dict of weights plus width metadata and loaded
with `torch.load(..., weights_only=True)`.
