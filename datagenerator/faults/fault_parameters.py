from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np


@dataclass
class FaultParameters:
    """Base parameters for fault generation."""

    mode: Literal["random", "self_branching", "stairs", "relay_ramps", "horst_graben"]
    throw: float
    dip: float
    strike: float
    length: float
    width: float
    depth: float
    sigma: float
    tilt: float
    branching_probability: Optional[float] = None
    relay_ramp_distance: Optional[float] = None
    horst_graben_width: Optional[float] = None


@dataclass
class FaultGeometryParameters:
    """Parameters for fault geometry calculations."""

    ellipsoid_axes: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    rotation_angles: Tuple[float, float, float]
    displacement_vector: Tuple[float, float, float]


@dataclass
class FaultQCParameters:
    """Parameters for quality control and visualization."""

    plot_fault_planes: bool = True
    plot_displacement_vectors: bool = True
    plot_fault_traces: bool = True
    save_plots: bool = True
    plot_directory: str = "fault_qc_plots"
