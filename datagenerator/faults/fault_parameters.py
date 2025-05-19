from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np


@dataclass
class FaultParameters:
    """Parameters for fault generation.

    Attributes:
        mode: Type of fault generation ("random", "self_branching", "stairs", "relay_ramps", "horst_graben")
        throw: Fault throw in meters
        dip: Fault dip angle in degrees
        strike: Fault strike angle in degrees
        length: Fault length in meters
        width: Fault width in meters
        depth: Fault depth in meters
        sigma: Standard deviation for fault displacement
        tilt: Fault tilt angle in degrees
        branching_probability: Probability of fault branching (for self_branching mode)
        stairs_count: Number of stairs in fault (for stairs mode)
        relay_ramp_distance: Distance between relay ramps (for relay_ramps mode)
        horst_graben_width: Width of horst and graben structures (for horst_graben mode)
    """

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
    stairs_count: Optional[int] = None
    relay_ramp_distance: Optional[float] = None
    horst_graben_width: Optional[float] = None


@dataclass
class FaultGeometryParameters:
    """Parameters for fault geometry calculations.

    Attributes:
        ellipsoid_axes: Semi-axes of the fault ellipsoid (length, width, depth)
        origin: Origin point for fault geometry (x, y, z)
        rotation_angles: Rotation angles in degrees (dip, strike, tilt)
        displacement_vector: Displacement vector (dx, dy, dz)
    """

    ellipsoid_axes: tuple[float, float, float]
    origin: tuple[float, float, float]
    rotation_angles: tuple[float, float, float]
    displacement_vector: tuple[float, float, float]


@dataclass
class FaultQCParameters:
    """Parameters for fault quality control and visualization.

    Attributes:
        plot_fault_planes: Whether to plot fault planes
        plot_displacement_vectors: Whether to plot displacement vectors
        plot_fault_traces: Whether to plot fault traces
        plot_directory: Directory to save plots
    """

    plot_fault_planes: bool = False
    plot_displacement_vectors: bool = False
    plot_fault_traces: bool = False
    plot_directory: Optional[str] = None
