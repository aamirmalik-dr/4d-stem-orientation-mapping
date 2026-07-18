"""orient4d: 4D-STEM orientation and phase mapping on simulated polycrystals."""

from .cluster import ClusterResult, cluster_grains, pattern_features
from .io import load_external, load_scan, save_scan
from .metrics import angular_error_deg, grain_agreement, orientation_phase_metrics
from .net import OrientPhaseNet, load_model, save_model
from .sim import (
    PHASE_NAMES,
    PHASES,
    DetectorGeometry,
    PhaseSpec,
    Scan4D,
    Scene,
    SceneParams,
    clean_pattern,
    make_scene,
    simulate_pattern,
    simulate_scan,
    simulate_scene_scan,
)
from .template import TemplateLibrary, build_library, match
from .virtual import annular_dark_field, bright_field, spot_dark_field, virtual_image

__version__ = "0.1.0"

__all__ = [
    "PHASE_NAMES",
    "PHASES",
    "ClusterResult",
    "DetectorGeometry",
    "OrientPhaseNet",
    "PhaseSpec",
    "Scan4D",
    "Scene",
    "SceneParams",
    "TemplateLibrary",
    "angular_error_deg",
    "annular_dark_field",
    "bright_field",
    "build_library",
    "clean_pattern",
    "cluster_grains",
    "grain_agreement",
    "load_external",
    "load_model",
    "load_scan",
    "make_scene",
    "match",
    "orientation_phase_metrics",
    "pattern_features",
    "save_model",
    "save_scan",
    "simulate_pattern",
    "simulate_scan",
    "simulate_scene_scan",
    "spot_dark_field",
    "virtual_image",
]
