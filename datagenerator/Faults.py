from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from .faults.types import (
    GeomodelProtocol,
    FaultError,
    FaultGenerationError,
    FaultGeometryError,
    FaultQCError,
    DepthMap,
    PropertyModel,
    FaultTrace,
    DisplacementVector,
)
from .faults.fault_parameters import (
    FaultParameters,
    FaultGeometryParameters,
    FaultQCParameters,
)
from .faults.fault_geometry import FaultGeometry
from .faults.fault_generator import FaultGenerator
from .faults.fault_qc import FaultQC
from .Horizons import Horizons
from .Geomodels import Geomodel
from .Parameters import Parameters


@dataclass
class LegacyParameters:
    """Adapter class to convert legacy Parameters to new FaultParameters."""

    parameters: Parameters

    def to_fault_parameters(self) -> FaultParameters:
        """Convert legacy parameters to new FaultParameters format."""
        # Map legacy mode names to standardized ones
        mode_mapping = {
            "random": "random",
            "self_branching": "self_branching",
            "stair_case": "stairs",
            "stairs": "stairs",
            "relay_ramp": "relay_ramps",
            "relay_ramps": "relay_ramps",
            "horst_and_graben": "horst_graben",
            "horst_graben": "horst_graben",
        }

        mode = mode_mapping.get(self.parameters.fault_mode)
        if mode is None:
            raise FaultError(f"Unknown fault mode: {self.parameters.fault_mode}")

        if mode == "random":
            return self._get_random_params()
        elif mode == "self_branching":
            return self._get_self_branching_params()
        elif mode == "stairs":
            return self._get_stairs_params()
        elif mode == "relay_ramps":
            return self._get_relay_ramp_params()
        elif mode == "horst_graben":
            return self._get_horst_graben_params()
        else:
            raise FaultError(f"Unknown fault mode: {mode}")

    def _get_random_params(self) -> FaultParameters:
        return FaultParameters(
            mode="random",
            throw=self.parameters.fault_throw,
            dip=self.parameters.fault_dip,
            strike=self.parameters.fault_strike,
            length=self.parameters.fault_length,
            width=self.parameters.fault_width,
            depth=self.parameters.fault_depth,
            sigma=self.parameters.fault_sigma,
            tilt=self.parameters.fault_tilt,
        )

    def _get_self_branching_params(self) -> FaultParameters:
        params = self._get_random_params()
        params.branching_probability = self.parameters.fault_branching_probability
        return params

    def _get_stairs_params(self) -> FaultParameters:
        params = self._get_random_params()
        params.stairs_count = self.parameters.fault_stairs_count
        return params

    def _get_relay_ramp_params(self) -> FaultParameters:
        params = self._get_random_params()
        params.relay_ramp_distance = self.parameters.fault_relay_ramp_distance
        return params

    def _get_horst_graben_params(self) -> FaultParameters:
        params = self._get_random_params()
        params.horst_graben_width = self.parameters.fault_horst_graben_width
        return params


class Faults(Horizons, Geomodel):
    """Legacy-compatible wrapper for the new fault system."""

    def __init__(
        self,
        parameters: Parameters,
        unfaulted_depth_maps: np.ndarray,
        onlap_horizon_list: np.ndarray,
        geomodels: Geomodel,
        fan_horizon_list: Optional[np.ndarray] = None,
        fan_thickness: Optional[np.ndarray] = None,
    ):
        # Initialize parent classes
        super().__init__(parameters)

        # Convert legacy parameters to new format
        legacy_params = LegacyParameters(parameters)
        self.fault_params = legacy_params.to_fault_parameters()

        # Initialize components
        self.geometry = FaultGeometry(
            FaultGeometryParameters(
                ellipsoid_axes=(
                    parameters.fault_length,
                    parameters.fault_width,
                    parameters.fault_depth,
                ),
                origin=self._calculate_origin(geomodels.geologic_age_store.shape),
                rotation_angles=(
                    parameters.fault_dip,
                    parameters.fault_strike,
                    parameters.fault_tilt,
                ),
                displacement_vector=(0.0, 0.0, parameters.fault_throw),
            )
        )

        self.generator = FaultGenerator(self.fault_params)
        self.qc = FaultQC(
            FaultQCParameters(
                plot_fault_planes=parameters.model_qc_volumes,
                plot_displacement_vectors=parameters.model_qc_volumes,
                plot_fault_traces=parameters.model_qc_volumes,
                plot_directory=parameters.output_directory,
            )
        )

        # Store input data
        self.unfaulted_depth_maps = unfaulted_depth_maps
        self.onlap_horizon_list = onlap_horizon_list
        self.fan_horizon_list = fan_horizon_list
        self.fan_thickness = fan_thickness
        self.vols = geomodels

        # Initialize storage
        cube_shape = geomodels.geologic_age_store.shape
        self.faulted_age_volume = self.cfg.zarr_init(
            "faulted_age_volume", shape=cube_shape
        )
        self.faulted_depth_maps = self.cfg.zarr_init(
            "faulted_depth_maps", shape=unfaulted_depth_maps.shape
        )
        self.faulted_depth_maps_gaps = self.cfg.zarr_init(
            "faulted_depth_maps_gaps", shape=unfaulted_depth_maps.shape
        )
        self.fault_planes = self.cfg.zarr_init("fault_planes", shape=cube_shape)
        self.fault_intersections = self.cfg.zarr_init(
            "fault_intersections", shape=cube_shape
        )
        self.fault_plane_throw = self.cfg.zarr_init(
            "fault_plane_throw", shape=cube_shape
        )
        self.fault_plane_azimuth = self.cfg.zarr_init(
            "fault_plane_azimuth", shape=cube_shape
        )
        self.faulted_onlap_segments = self.cfg.zarr_init(
            "faulted_onlap_segments", shape=cube_shape
        )

    def _calculate_origin(
        self, shape: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """Calculate origin point for fault geometry."""
        return (shape[0] / 2, shape[1] / 2, shape[2] / 2)

    def apply_faulting_to_geomodels_and_depth_maps(self) -> None:
        """Apply faulting to geomodels and depth maps using the new system."""
        try:
            # Generate faults
            fault_traces = self.generator.generate_faults()

            # Apply faulting to depth maps
            self.faulted_depth_maps[:] = self.generator.apply_faulting(
                self.unfaulted_depth_maps, fault_traces
            )

            # Apply faulting to geologic age
            self.faulted_age_volume[:] = self.generator.apply_faulting(
                self.vols.geologic_age_store[:], fault_traces
            )

            # Apply faulting to onlap segments
            self.faulted_onlap_segments[:] = self.generator.apply_faulting(
                self.onlap_horizon_list, fault_traces
            )

            # Create QC plots if enabled
            if self.cfg.model_qc_volumes:
                self.qc.create_qc_plots(
                    self.faulted_depth_maps[:], self.faulted_age_volume[:], fault_traces
                )

        except FaultGenerationError as e:
            raise FaultError(f"Failed to generate faults: {e}")
        except FaultGeometryError as e:
            raise FaultError(f"Failed to apply fault geometry: {e}")
        except FaultQCError as e:
            raise FaultError(f"Failed to create QC plots: {e}")

    def build_faulted_property_geomodels(self, facies: np.ndarray) -> None:
        """Build faulted property geomodels using the new system."""
        try:
            # Get fault traces
            fault_traces = self.generator.get_fault_traces()

            # Apply faulting to each property
            if self.cfg.include_channels:
                self.vols.floodplain_shale = self.generator.apply_faulting(
                    self.vols.floodplain_shale, fault_traces
                )
                self.vols.channel_fill = self.generator.apply_faulting(
                    self.vols.channel_fill, fault_traces
                )
                self.vols.shale_channel_drape = self.generator.apply_faulting(
                    self.vols.shale_channel_drape, fault_traces
                )
                self.vols.levee = self.generator.apply_faulting(
                    self.vols.levee, fault_traces
                )
                self.vols.crevasse = self.generator.apply_faulting(
                    self.vols.crevasse, fault_traces
                )

                # Reassign channel segments
                self.vols.channel_segments = self.generator.reassign_channel_segments(
                    self.faulted_age_volume[:],
                    self.vols.floodplain_shale,
                    self.vols.channel_fill,
                    self.vols.shale_channel_drape,
                    self.vols.levee,
                    self.vols.crevasse,
                    self.maps.channels,
                )

        except FaultGenerationError as e:
            raise FaultError(f"Failed to build faulted property models: {e}")

    def improve_depth_maps_post_faulting(
        self,
        unfaulted_geologic_age: np.ndarray,
        faulted_geologic_age: np.ndarray,
        onlap_clips: np.ndarray,
    ) -> Tuple[DepthMap, DepthMap]:
        """Improve depth maps after faulting using the new system."""
        try:
            if self.faulted_depth_maps is None:
                raise FaultError("Must apply faulting before improving depth maps")

            # Get fault traces
            fault_traces = self.generator.get_fault_traces()

            # Improve depth maps
            improved_maps, improved_gaps = self.generator.improve_depth_maps(
                self.faulted_depth_maps[:],
                unfaulted_geologic_age,
                faulted_geologic_age,
                onlap_clips,
                fault_traces,
            )

            return improved_maps, improved_gaps

        except FaultGenerationError as e:
            raise FaultError(f"Failed to improve depth maps: {e}")

    # Legacy method aliases for backward compatibility
    generate_faults = lambda self: self.generator.generate_faults()
    fault_parameters = lambda self: self.fault_params
    build_faults = lambda self, fp, verbose=False: self.generator.build_faults(
        fp, verbose
    )
    apply_faulting = lambda self, traces, stretch_times, verbose=False: self.generator.apply_faulting(
        traces, stretch_times, verbose
    )
    get_displacement_vector = (
        lambda self, *args, **kwargs: self.geometry.get_displacement_vector(
            *args, **kwargs
        )
    )
    apply_xyz_displacement = lambda self, *args: self.geometry.apply_xyz_displacement(
        *args
    )
    rotate_3d_ellipsoid = lambda self, *args: self.geometry.rotate_3d_ellipsoid(*args)
    apply_3d_rotation = lambda self, *args: self.geometry.apply_3d_rotation(*args)
    get_fault_plane_sobel = lambda self, *args: self.geometry.get_fault_plane_sobel(
        *args
    )
    get_fault_centre = lambda self, *args: self.geometry.get_fault_centre(*args)
    xyz_dis = lambda self, *args: self.geometry.xyz_dis(*args)
    fault_summary_plot = lambda self, *args: self.qc.fault_summary_plot(*args)
    create_qc_plots = lambda self: self.qc.create_qc_plots(
        self.faulted_depth_maps[:],
        self.faulted_age_volume[:],
        self.generator.get_fault_traces(),
    )


# Legacy helper functions
def find_zero_thickness_onlapping_layers(
    z: np.ndarray, onlap_list: np.ndarray
) -> np.ndarray:
    """Find layers with zero thickness in onlapping sequences."""
    return FaultGenerator.find_zero_thickness_onlapping_layers(z, onlap_list)


def fix_zero_thickness_fan_layers(
    z: np.ndarray, layer_number: int, thickness: np.ndarray
) -> np.ndarray:
    """Fix zero thickness in fan layers."""
    return FaultGenerator.fix_zero_thickness_fan_layers(z, layer_number, thickness)


def fix_zero_thickness_onlap_layers(
    faulted_depth_maps: np.ndarray, onlap_dict: dict
) -> np.ndarray:
    """Fix zero thickness in onlapping layers."""
    return FaultGenerator.fix_zero_thickness_onlap_layers(
        faulted_depth_maps, onlap_dict
    )
