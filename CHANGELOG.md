# Changelog

## 0.1.1 (2026-09-26)

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- CI badge added to the README.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.1.0 (2026-07-18)

- Kinematical polycrystal simulator: structure factors from per-atom scattering factors, pixel-integrated Bragg spots, direct beam, background, Poisson counting, Voronoi grain scenes with camera-length jitter, descan jitter, mosaic spread, and probe-footprint mixing at boundaries; per-position grain, phase, orientation, and purity ground truth.
- Virtual bright-field, annular, and spot dark-field imaging; template matching by normalised cross-correlation with parabolic sub-step refinement; a 157k-parameter two-head CNN with a symmetry-multiplied circular angle target; PCA plus k-means grain clustering with silhouette k selection.
- Symmetry-aware angular error, phase accuracy, and ARI/NMI grain recovery with interior and boundary masks.
- Benchmarks over dose, grain count, mosaic spread, detector resolution, template-library tuning, clustering, and a forward-model-mismatch scenario through a `library_detector` override.
- The `orient4d` CLI, a bring-your-own-4D-data loader with HyperSpy and py4DSTEM conversion recipes, committed CNN weights and sample scan, results JSON, figures, model card, API docs, executed tutorial, and a CI workflow.
- Maintenance after publication: structure-factor comments and a uint16 bound corrected, claims scoped, README expanded and restructured, hero re-exported at higher resolution.
