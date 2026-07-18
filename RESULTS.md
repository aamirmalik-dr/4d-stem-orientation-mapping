# Benchmark results

Every number below was measured in a fresh Python 3.11 virtual environment
on CPU by running the committed configs (`orient4d benchmark configs/<name>.yaml`,
or `python scripts/run_all.py` for everything). Seeds are fixed in the
configs; re-running reproduces the JSONs in `results/` up to wall-clock
timing fields. Unless stated otherwise, numbers are means over 3
independent scans (48x48 probe positions, 8 grains, 2 phases, dose 300
electrons per pattern, mosaic spread 0.3 degrees) and orientation error is
scored on grain-interior pixels (probe purity >= 0.95) whose phase was
predicted correctly; the tables state all-pixel numbers where they differ.

Reading guide for the error scale: the intra-grain mosaic spread of 0.3
degrees is part of the ground truth, i.e. methods are scored against the
per-position local orientation, not the grain mean, so sub-0.3-degree MAE
means a method tracks orientation below the mosaic disorder.

## Headline: methods at the default operating point

`configs/compare.yaml`, results in `results/compare.json` and
`results/metrics.json`.

| Method | Orientation MAE (deg), interior | MAE all pixels | Phase accuracy (all) | Within 1 deg | ms per pattern |
|---|---|---|---|---|---|
| Template matching (0.5 deg library, refined) | **0.248 +/- 0.004** | 0.276 | 0.999 | 0.992 | **0.118** |
| CNN (157k params) | 0.595 +/- 0.051 | 0.678 | 0.999 | 0.806 | 0.228 |
| k-means clustering (unsupervised, at true k=8) | ARI 0.868 +/- 0.121, NMI 0.914 | | | | |

The honest summary up front: on this synthetic benchmark the classical
template matcher wins on orientation accuracy at every condition measured,
and is also faster per pattern at these library sizes. The CNN never beats
it outright. The reason is structural, not an implementation accident: the
library is rendered by the exact forward model that generated the data, so
template matching is close to a matched filter, which is the optimal
detector for this setting. The often-heard claim that deep learning helps
most at low dose is not supported here (see the dose sweep); the CNN's real
selling points in this benchmark are constant per-pattern cost independent
of library size and graceful degradation under forward-model mismatch, and
only the second materialises as a (near-)tie rather than a win.

## Dose sweep

`configs/dose_sweep.yaml`. Dose is total electrons per pattern; 25 percent
sits in the direct beam and 15 percent in the background, so at dose 10
roughly 6 electrons carry all Bragg information.

| Dose | Template MAE (deg) | CNN MAE (deg) | Template phase acc | CNN phase acc |
|---|---|---|---|---|
| 10 | **1.863 +/- 0.087** | 3.254 +/- 0.093 | **0.960** | 0.933 |
| 30 | **0.865 +/- 0.007** | 1.268 +/- 0.122 | 1.000 | 0.999 |
| 100 | **0.443 +/- 0.017** | 0.708 +/- 0.065 | 1.000 | 1.000 |
| 300 | **0.250 +/- 0.003** | 0.511 +/- 0.018 | 1.000 | 1.000 |
| 1000 | **0.143 +/- 0.019** | 0.499 +/- 0.138 | 1.000 | 1.000 |
| 3000 | **0.072 +/- 0.001** | 0.418 +/- 0.156 | 1.000 | 1.000 |

Two readings. First, template matching keeps improving as photons arrive
(0.072 degrees at dose 3000, photon-limited), while the CNN saturates at a
~0.4-0.5 degree regression floor from about dose 300; direct angle
regression does not converge to the shot-noise limit. Second, at the
starved end the CNN degrades faster, not slower: dose 10 is 1.86 versus
3.25 degrees. Deep learning is not a dose saviour when the classical
competitor knows the true forward model.

## Fair tuning of the classical baseline

`configs/template_tuning.yaml`: library step and sub-step parabolic
refinement swept on the same three fixed scans, so the template matcher is
scored at its best operating point rather than a strawman. CNN reference on
identical scans.

| Library step | Templates | MAE refined (deg) | MAE grid-snap (deg) | ms per pattern |
|---|---|---|---|---|
| 5.0 deg | 30 | 0.266 | 1.037 | 0.128 |
| 2.0 deg | 75 | **0.246** | 0.448 | 0.128 |
| 1.0 deg | 150 | 0.248 | 0.337 | 0.132 |
| 0.5 deg | 300 | 0.248 | 0.276 | 0.132 |
| 0.25 deg | 600 | 0.248 | 0.254 | 0.137 |
| CNN | | 0.595 | | 0.231 |

The load-bearing component is the parabolic refinement, not library
density: a 30-template library with refinement (0.266 degrees) already
beats the CNN, and beyond a 2-degree step the refined error is flat at the
noise floor. Per-pattern cost barely moves from 30 to 600 templates because
the correlation is a single matrix multiply; this scaling argument would
eventually favour a CNN, but only at library sizes far beyond this
two-phase, one-rotation-axis problem (a full 3D orientation library has
tens of thousands of entries).

## Forward-model mismatch: where the gap nearly closes

`configs/mismatch.yaml`: the scans carry a different intensity budget than
the library assumes (direct beam 45 instead of 25 percent, background 30
instead of 15 percent and narrower, descan wobble sigma 1.0 instead of 0.3
px). This models the practical case of a miscalibrated forward model. The
CNN runs unchanged; its training randomisation covered budgets of 15-35 /
8-25 percent, so this condition extrapolates for it too.

| Method | MAE (deg), interior | Within 1 deg | Phase accuracy (all) |
|---|---|---|---|
| Template matching | **0.989 +/- 0.034** | 0.714 | 0.991 |
| CNN | 1.137 +/- 0.035 | 0.534 | 0.992 |

The template matcher's advantage shrinks from a factor 2.4 to a factor
1.15, and both methods lose their sub-degree precision. The normalised
cross-correlation on square-root intensities is itself fairly robust to
radially symmetric budget changes, which is why the CNN narrows the gap but
does not flip it. An earlier framing of this repository would have been
tempted to headline "CNN wins under mismatch"; the measured answer is that
it does not, it only stops losing badly.

## Grain size sweep

`configs/grain_sweep.yaml`, fixed 48x48 scan. More grains means smaller
grains and more boundary pixels (interior fraction falls from 0.88 at 4
grains to 0.60 at 32).

| Grains | Interior fraction | Template MAE interior / all (deg) | CNN MAE interior / all (deg) |
|---|---|---|---|
| 4 | 0.875 | 0.242 / 0.284 | 0.613 / 0.693 |
| 8 | 0.828 | 0.247 / 0.285 | 0.514 / 0.619 |
| 16 | 0.738 | 0.247 / 0.314 | 0.538 / 0.688 |
| 32 | 0.603 | 0.247 / 0.371 | 0.590 / 0.809 |

Interior accuracy is flat for both methods; the all-pixel error grows with
the boundary fraction because boundary patterns genuinely contain two
lattices and the dominant-grain ground truth is only partially attainable
there. Phase accuracy stays above 0.997 everywhere.

## Orientation-spread (mosaicity) sweep

`configs/mosaic_sweep.yaml`. The per-position mosaic jitter is itself the
ground truth, so this measures whether methods track local orientation
rather than a grain average.

| Mosaic sigma (deg) | Template MAE (deg) | CNN MAE (deg) |
|---|---|---|
| 0.0 | 0.243 | 0.593 |
| 0.5 | 0.253 | 0.508 |
| 1.0 | 0.245 | 0.556 |
| 2.0 | 0.248 | 0.525 |
| 5.0 | 0.267 | 0.545 |

Both methods are flat across a 0-5 degree spread: they estimate each
pattern independently, so intra-grain disorder is tracked rather than
averaged away. This is the property that makes per-pattern mapping useful
for strain and sub-grain analysis in the first place.

## Pattern-resolution sweep

`configs/resolution_sweep.yaml`, detector pixels per side at fixed angular
range. Rendering pixel-integrates the spots, so low resolution bins flux
rather than aliasing it. Templates run natively per resolution; CNN input
is bilinearly resampled to its 64 px training resolution (documented
choice).

| Pixels | Template MAE (deg) | CNN MAE (deg) | Template phase acc | CNN phase acc |
|---|---|---|---|---|
| 16 | 0.650 +/- 0.069 | 1.637 +/- 0.113 | 0.993 | 0.998 |
| 32 | 0.301 | 0.555 | 1.000 | 1.000 |
| 48 | 0.256 | 0.527 | 1.000 | 1.000 |
| 64 | 0.250 | 0.511 | 1.000 | 1.000 |

16x16 patterns (spots under one pixel wide) still support sub-degree
mapping with templates and near-perfect phase identification; going beyond
32-48 px buys little at this dose. For dose-limited work, coarse detectors
waste nothing here.

## Unsupervised grain recovery

`configs/clustering.yaml`: PCA(20) features of sqrt patterns, k-means.

| Setting | Value |
|---|---|
| ARI at true k=8 | 0.868 +/- 0.121 |
| NMI at true k=8 | 0.914 +/- 0.068 |
| Silhouette-selected k | 6.7 +/- 0.5 (true 8) |
| ARI at selected k | 0.889 +/- 0.080 |

The silhouette criterion under-selects k because two of the eight grains
occasionally land within a few degrees of each other, making their patterns
identical up to noise; merging them is the physically correct unsupervised
answer, and the selected-k ARI is in fact slightly higher than at the true
k. On the committed sample scan (6 well-separated grains) clustering is
near perfect (ARI 0.992, measured in the tutorial notebook).

## CNN training

`orient4d train --steps 6000` (16.3 minutes CPU, history in
`results/training_history.json`, curves in `figures/training.png`).
Validation on the randomised stream (doses 10-3000, boundary mixtures
included): phase accuracy 0.982, angle MAE 1.32 degrees, median 0.66
degrees at step 6000. The gap between this harsh validation MAE and the
benchmark MAE (0.6 degrees at dose 300) is the dose mixture: the validation
stream spends a third of its samples below dose 100.

## What was checked to keep these numbers honest

- The classical baseline was tuned on its own sweep (step, refinement)
  before any comparison was quoted; the headline uses 0.5 degrees but the
  conclusion is unchanged at every setting.
- The mismatch benchmark was added specifically to probe the advertised
  CNN advantage (robustness to model error); the measured result, a
  near-tie rather than a CNN win, is reported as such.
- Boundary pixels are never silently dropped: interior and all-pixel
  numbers are both in the JSONs, and the interior mask threshold (purity
  0.95) is in every config.
- The mosaic ground truth follows the per-position jitter, so methods are
  scored against what the sample actually did, not a smoothed fiction.
