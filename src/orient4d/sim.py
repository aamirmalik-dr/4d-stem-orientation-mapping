"""Kinematical 4D-STEM simulator: polycrystalline scenes with exact ground truth.

The forward model is deliberately simple and fully documented so that every
benchmark number in this repository can be traced to a known physical setting:

- Each phase is a 2D projected crystal (lattice vectors plus an atomic basis)
  viewed down a fixed zone axis. Reciprocal lattice points inside the detector
  carry kinematical intensities ``|F(g)|^2`` where the structure factor uses a
  single-Gaussian electron scattering factor per atom,
  ``f_j(g) = Z_j^0.75 * exp(-b_env * g^2)``. The envelope lumps the scattering
  factor fall-off and Debye-Waller damping into one constant.
- A diffraction pattern is rendered on an ``n_px x n_px`` detector by
  pixel-integrating an isotropic Gaussian at each Bragg position (exact erf
  integration, so low pattern resolutions bin flux instead of aliasing it),
  plus a direct beam and a smooth two-term background. The expected pattern is
  normalised to unit mass and scaled by ``dose`` (total electrons per
  pattern); recorded counts are Poisson.
- A scene is a Voronoi tessellation of the scan grid into grains, each with a
  phase and an in-plane orientation drawn uniformly from the symmetry-reduced
  range. The probe has a Gaussian real-space footprint, so positions near a
  grain boundary record an incoherent, area-weighted mixture of the
  neighbouring grains' patterns. Ground truth (grain id, phase, local
  orientation, purity) is tracked for every probe position.

Orientation convention: ``theta`` is the in-plane rotation in degrees,
reduced to ``[0, 360 / sym_order)`` where ``sym_order`` is the rotational
symmetry order of the projected pattern (6 for the hexagonal phase, 4 for the
square phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import erf

PHASE_NAMES: tuple[str, ...] = ("hexagonal", "square")


@dataclass(frozen=True)
class PhaseSpec:
    """A 2D projected crystal phase.

    Attributes:
        name: Phase identifier.
        a: Lattice parameter in angstrom.
        lattice: Bravais type, "hexagonal" or "square".
        basis: Atomic basis as (u, v, Z) fractional coordinates and atomic number.
        sym_order: Rotational symmetry order of the projected pattern.
    """

    name: str
    a: float
    lattice: str
    basis: tuple[tuple[float, float, int], ...]
    sym_order: int

    @property
    def fold(self) -> float:
        """Symmetry-reduced orientation range in degrees."""
        return 360.0 / self.sym_order

    def lattice_vectors(self) -> np.ndarray:
        """Real-space lattice vectors as rows of a (2, 2) array, in angstrom."""
        if self.lattice == "hexagonal":
            return np.array([[self.a, 0.0], [self.a / 2.0, self.a * np.sqrt(3.0) / 2.0]])
        if self.lattice == "square":
            return np.array([[self.a, 0.0], [0.0, self.a]])
        raise ValueError(f"unknown lattice type: {self.lattice}")


PHASES: dict[str, PhaseSpec] = {
    # Honeycomb monolayer (graphene-like): two carbon atoms per cell, 6-fold pattern.
    # With lattice vectors at 60 degrees, the B sublattice sits at (1/3, 1/3).
    "hexagonal": PhaseSpec(
        name="hexagonal",
        a=2.46,
        lattice="hexagonal",
        basis=((0.0, 0.0, 6), (1.0 / 3.0, 1.0 / 3.0, 6)),
        sym_order=6,
    ),
    # FCC metal down [001] (aluminium-like): square projected net, one column type.
    "square": PhaseSpec(
        name="square",
        a=2.03,
        lattice="square",
        basis=((0.0, 0.0, 13),),
        sym_order=4,
    ),
}

FOLDS: np.ndarray = np.array([PHASES[n].fold for n in PHASE_NAMES])

_ENVELOPE_B = 0.7  # angstrom^2, lumped scattering-factor + Debye-Waller envelope


@dataclass(frozen=True)
class DetectorGeometry:
    """Diffraction detector geometry and per-pattern intensity budget.

    Attributes:
        n_px: Detector size in pixels (patterns are n_px x n_px).
        k_max: Detector half-width in reciprocal angstrom (1/d convention).
        spot_sigma: Bragg spot Gaussian sigma in reciprocal angstrom.
        direct_fraction: Fraction of the dose in the unscattered direct beam.
        background_fraction: Fraction of the dose in the smooth background.
        background_width: Sigma of the broad diffuse-background Gaussian (1/angstrom).
    """

    n_px: int = 64
    k_max: float = 1.35
    spot_sigma: float = 0.045
    direct_fraction: float = 0.25
    background_fraction: float = 0.15
    background_width: float = 0.55

    @property
    def dk(self) -> float:
        """Reciprocal-space pixel size in 1/angstrom."""
        return 2.0 * self.k_max / self.n_px

    def pixel_centers(self) -> np.ndarray:
        """Pixel-center coordinates along one axis, in 1/angstrom."""
        return (np.arange(self.n_px) - (self.n_px - 1) / 2.0) * self.dk


def reciprocal_vectors(phase: PhaseSpec) -> np.ndarray:
    """Reciprocal lattice vectors (rows b1, b2) in 1/angstrom, no 2*pi factor."""
    a_mat = phase.lattice_vectors()
    return np.linalg.inv(a_mat).T


def lattice_reflections(phase: PhaseSpec, k_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate reflections of ``phase`` inside radius ``k_max``.

    Returns:
        Tuple of (g_vectors, intensities): positions (n, 2) in 1/angstrom at
        zero rotation, and kinematical intensities |F(g)|^2 (arbitrary units).
        The (0, 0) beam is excluded.
    """
    b_mat = reciprocal_vectors(phase)
    g_min = min(np.linalg.norm(b_mat[0]), np.linalg.norm(b_mat[1]))
    m = int(np.ceil(k_max / g_min)) + 2
    hk = np.array([(h, k) for h in range(-m, m + 1) for k in range(-m, m + 1) if (h, k) != (0, 0)])
    g = hk @ b_mat
    g_norm = np.linalg.norm(g, axis=1)
    keep = g_norm <= k_max * 1.05
    hk, g, g_norm = hk[keep], g[keep], g_norm[keep]
    f_atoms = np.array([[z**0.75 for _, _, z in phase.basis]])
    f_atoms = f_atoms * np.exp(-_ENVELOPE_B * g_norm[:, None] ** 2)
    uv = np.array([[u, v] for u, v, _ in phase.basis])
    phase_arg = 2.0 * np.pi * (hk @ uv.T)
    f_total = np.sum(f_atoms * np.exp(1j * phase_arg), axis=1)
    intensity = np.abs(f_total) ** 2
    return g, intensity


_REFLECTION_CACHE: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}


def _cached_reflections(phase: PhaseSpec, k_max: float) -> tuple[np.ndarray, np.ndarray]:
    key = (phase.name, round(float(k_max), 6))
    if key not in _REFLECTION_CACHE:
        _REFLECTION_CACHE[key] = lattice_reflections(phase, k_max)
    return _REFLECTION_CACHE[key]


def _pixel_profile(x0_px: float, sigma_px: float, n_px: int) -> np.ndarray:
    """Exact per-pixel integral of a unit-mass 1D Gaussian centred at x0_px."""
    edges = np.arange(n_px + 1) - 0.5
    cdf = 0.5 * (1.0 + erf((edges - x0_px) / (np.sqrt(2.0) * sigma_px)))
    return np.diff(cdf)


def clean_pattern(
    phase: PhaseSpec,
    theta_deg: float,
    det: DetectorGeometry,
    scale: float = 1.0,
    center_px: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Render the noise-free expected pattern, normalised to unit total mass.

    Args:
        phase: Crystal phase.
        theta_deg: In-plane rotation in degrees.
        det: Detector geometry.
        scale: Camera-length factor applied to all reciprocal vectors.
        center_px: Pattern-center (descan) offset in pixels, (row, col).

    Returns:
        (n_px, n_px) float64 array of expected fractional intensity, sum 1.
    """
    n = det.n_px
    g, intensity = _cached_reflections(phase, det.k_max / max(scale, 1e-6))
    t = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    g_rot = (g @ rot.T) * scale
    sigma_px = det.spot_sigma / det.dk
    c0 = (n - 1) / 2.0
    col_px = g_rot[:, 0] / det.dk + c0 + center_px[1]
    row_px = g_rot[:, 1] / det.dk + c0 + center_px[0]

    bragg = np.zeros((n, n))
    for x, y, amp in zip(col_px, row_px, intensity):
        bragg += amp * np.outer(_pixel_profile(y, sigma_px, n), _pixel_profile(x, sigma_px, n))
    total = bragg.sum()
    if total > 0:
        bragg *= (1.0 - det.direct_fraction - det.background_fraction) / total

    direct = np.outer(
        _pixel_profile(c0 + center_px[0], sigma_px * 1.2, n),
        _pixel_profile(c0 + center_px[1], sigma_px * 1.2, n),
    )
    direct *= det.direct_fraction / max(direct.sum(), 1e-12)

    kk = det.pixel_centers()
    broad = np.exp(-(kk**2) / (2.0 * det.background_width**2))
    broad2d = np.outer(broad, broad)
    background = 0.75 * broad2d / broad2d.sum() + 0.25 / (n * n)
    background *= det.background_fraction

    expected = bragg + direct + background
    return expected / expected.sum()


def simulate_pattern(
    phase: PhaseSpec,
    theta_deg: float,
    det: DetectorGeometry,
    dose: float,
    rng: np.random.Generator,
    scale: float = 1.0,
    center_px: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Simulate one Poisson-noised pattern with ``dose`` total expected electrons."""
    expected = clean_pattern(phase, theta_deg, det, scale=scale, center_px=center_px)
    return rng.poisson(expected * dose).astype(np.float32)


@dataclass(frozen=True)
class SceneParams:
    """Parameters of a polycrystalline 4D-STEM scan.

    Attributes:
        scan_shape: (rows, cols) of the probe raster.
        n_grains: Number of Voronoi grains.
        dose: Total expected electrons per diffraction pattern.
        mosaic_sigma: Intra-grain orientation spread (degrees, Gaussian sigma).
        probe_sigma: Probe real-space Gaussian sigma in scan-pixel units.
        phase_names: Phases to draw grains from.
        scale_jitter: Per-scan camera-length jitter (Gaussian sigma, fractional).
        center_jitter: Per-position descan jitter in detector pixels (sigma).
    """

    scan_shape: tuple[int, int] = (48, 48)
    n_grains: int = 8
    dose: float = 300.0
    mosaic_sigma: float = 0.3
    probe_sigma: float = 0.7
    phase_names: tuple[str, ...] = PHASE_NAMES
    scale_jitter: float = 0.02
    center_jitter: float = 0.3


@dataclass(frozen=True)
class Grain:
    """One grain: Voronoi seed position, phase, and base orientation."""

    grain_id: int
    seed_yx: tuple[float, float]
    phase_name: str
    theta: float


@dataclass(frozen=True)
class Scene:
    """A tessellated scene: params plus the per-grain phase and orientation table."""

    params: SceneParams
    grains: tuple[Grain, ...]


@dataclass
class Scan4D:
    """A simulated 4D-STEM scan with exact per-position ground truth.

    Attributes:
        data: (rows, cols, n_px, n_px) float32 Poisson counts.
        grain_id: (rows, cols) int16 dominant grain per position.
        phase_id: (rows, cols) int8 index into ``PHASE_NAMES``.
        theta: (rows, cols) float32 local orientation of the dominant grain (deg).
        purity: (rows, cols) float32 probe-weight fraction of the dominant grain.
        det: Detector geometry used.
        params: Scene parameters used.
        scene: The generating scene (grain table), if available.
    """

    data: np.ndarray
    grain_id: np.ndarray
    phase_id: np.ndarray
    theta: np.ndarray
    purity: np.ndarray
    det: DetectorGeometry
    params: SceneParams
    scene: Scene | None = field(default=None, repr=False)


def make_scene(params: SceneParams, rng: np.random.Generator) -> Scene:
    """Draw a random scene: Voronoi seeds with minimum separation, phases, angles."""
    h, w = params.scan_shape
    seeds: list[tuple[float, float]] = []
    for _ in range(2000):
        if len(seeds) == params.n_grains:
            break
        cand = (rng.uniform(0, h), rng.uniform(0, w))
        if all(np.hypot(cand[0] - s[0], cand[1] - s[1]) >= 3.0 for s in seeds):
            seeds.append(cand)
    if len(seeds) < params.n_grains:
        raise ValueError("could not place grain seeds with minimum separation; scan too small")
    grains = []
    for i, seed in enumerate(seeds):
        phase_name = params.phase_names[rng.integers(len(params.phase_names))]
        theta = float(rng.uniform(0.0, PHASES[phase_name].fold))
        grains.append(Grain(grain_id=i, seed_yx=seed, phase_name=phase_name, theta=theta))
    return Scene(params=params, grains=tuple(grains))


def grain_label_map(scene: Scene, points_yx: np.ndarray) -> np.ndarray:
    """Nearest-seed grain id for an array of (…, 2) scan coordinates."""
    seeds = np.array([g.seed_yx for g in scene.grains])
    d2 = ((points_yx[..., None, :] - seeds[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=-1)


def _probe_offsets(probe_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Sub-grid offsets and Gaussian weights sampling the probe footprint."""
    r = np.linspace(-1.6 * probe_sigma, 1.6 * probe_sigma, 5)
    dy, dx = np.meshgrid(r, r, indexing="ij")
    w = np.exp(-(dy**2 + dx**2) / (2.0 * probe_sigma**2))
    offsets = np.stack([dy.ravel(), dx.ravel()], axis=-1)
    return offsets, w.ravel() / w.sum()


def simulate_scan(
    scene: Scene,
    det: DetectorGeometry,
    rng: np.random.Generator,
) -> Scan4D:
    """Simulate the full 4D scan for ``scene`` on detector ``det``.

    At each probe position the Gaussian probe footprint is sampled on a 5x5
    sub-grid; each grain under the footprint contributes its clean pattern
    weighted by its probe-weight fraction (incoherent mixing). The dominant
    grain's orientation gets the per-position mosaic jitter; minority
    contributions use their grain's base orientation. Poisson noise is applied
    to the mixed expected pattern.
    """
    params = scene.params
    h, w = params.scan_shape
    n = det.n_px
    scale = float(1.0 + rng.normal(0.0, params.scale_jitter))
    offsets, weights = _probe_offsets(params.probe_sigma)

    data = np.zeros((h, w, n, n), dtype=np.float32)
    grain_id = np.zeros((h, w), dtype=np.int16)
    phase_id = np.zeros((h, w), dtype=np.int8)
    theta_map = np.zeros((h, w), dtype=np.float32)
    purity = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pts = np.array([i, j])[None, :] + offsets
            labels = grain_label_map(scene, pts)
            contrib: dict[int, float] = {}
            for lab, wt in zip(labels, weights):
                contrib[int(lab)] = contrib.get(int(lab), 0.0) + float(wt)
            keep = {g: wt for g, wt in contrib.items() if wt >= 0.02}
            total = sum(keep.values())
            dom = max(keep, key=lambda g: keep[g])
            dom_grain = scene.grains[dom]
            local_theta = dom_grain.theta + float(rng.normal(0.0, params.mosaic_sigma))
            center = tuple(rng.normal(0.0, params.center_jitter, size=2))
            expected = np.zeros((n, n))
            for g_idx, wt in keep.items():
                grain = scene.grains[g_idx]
                theta_g = local_theta if g_idx == dom else grain.theta
                expected += (wt / total) * clean_pattern(
                    PHASES[grain.phase_name], theta_g, det, scale=scale, center_px=center
                )
            data[i, j] = rng.poisson(expected * params.dose).astype(np.float32)
            grain_id[i, j] = dom
            phase_id[i, j] = PHASE_NAMES.index(dom_grain.phase_name)
            fold = PHASES[dom_grain.phase_name].fold
            theta_map[i, j] = local_theta % fold
            purity[i, j] = keep[dom] / total

    return Scan4D(
        data=data,
        grain_id=grain_id,
        phase_id=phase_id,
        theta=theta_map,
        purity=purity,
        det=det,
        params=params,
        scene=scene,
    )


def simulate_scene_scan(
    params: SceneParams,
    det: DetectorGeometry,
    seed: int,
) -> Scan4D:
    """Convenience wrapper: draw a scene and simulate its scan from one seed."""
    rng = np.random.default_rng(seed)
    scene = make_scene(params, rng)
    return simulate_scan(scene, det, rng)


def with_dose(params: SceneParams, dose: float) -> SceneParams:
    """Return a copy of ``params`` at a different dose."""
    return replace(params, dose=dose)
