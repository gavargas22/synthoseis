import os

import numpy as np
from datagenerator.Horizons import Horizons
from datagenerator.Geomodels import Geomodel
from datagenerator.Parameters import Parameters
from skimage import morphology, measure
from scipy.ndimage import minimum_filter, maximum_filter


class Closures(Horizons, Geomodel, Parameters):
    def __init__(self, parameters, faults, facies, onlap_horizon_list):
        self.closure_dict = dict()
        self.cfg = parameters
        self.faults = faults
        self.facies = facies
        self.onlap_list = onlap_horizon_list
        self.top_lith_facies = None
        self.closure_vol_shape = self.faults.faulted_age_volume.shape
        self.closure_segments = np.zeros(self.closure_vol_shape)
        self.oil_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.gas_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.brine_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.simple_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.strat_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.fault_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.hc_labels = np.zeros(self.closure_vol_shape, dtype="uint8")

        self.all_closure_segments = np.zeros(self.closure_vol_shape)

        # Class attributes added from Intersect3D
        self.wide_faults = np.zeros(self.closure_vol_shape)
        self.fat_faults = np.zeros(self.closure_vol_shape)
        self.onlaps_upward = np.zeros(self.closure_vol_shape)
        self.onlaps_downward = np.zeros(self.closure_vol_shape)

        # Faulted closures
        self.faulted_closures_oil = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.faulted_closures_gas = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.faulted_closures_brine = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.fault_closures_oil_segment_list = list()
        self.fault_closures_gas_segment_list = list()
        self.fault_closures_brine_segment_list = list()
        self.n_fault_closures_oil = 0
        self.n_fault_closures_gas = 0
        self.n_fault_closures_brine = 0

        self.faulted_all_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.fault_all_closures_segment_list = list()
        self.n_fault_all_closures = 0

        # Onlap closures
        self.onlap_closures_oil = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.onlap_closures_gas = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.onlap_closures_brine = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.onlap_closures_oil_segment_list = list()
        self.onlap_closures_gas_segment_list = list()
        self.onlap_closures_brine_segment_list = list()
        self.n_onlap_closures_oil = 0
        self.n_onlap_closures_gas = 0
        self.n_onlap_closures_brine = 0

        self.onlap_all_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.onlap_all_closures_segment_list = list()
        self.n_onlap_all_closures_oil = 0

        # Simple closures
        self.simple_closures_oil = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.simple_closures_gas = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.simple_closures_brine = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.simple_closures_oil_segment_list = list()
        self.simple_closures_gas_segment_list = list()
        self.simple_closures_brine_segment_list = list()
        self.n_4way_closures_oil = 0
        self.n_4way_closures_gas = 0
        self.n_4way_closures_brine = 0

        self.simple_all_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.simple_all_closures_segment_list = list()
        self.n_4way_all_closures = 0

        # False closures
        self.false_closures_oil = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.false_closures_gas = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.false_closures_brine = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.n_false_closures_oil = 0
        self.n_false_closures_gas = 0
        self.n_false_closures_brine = 0

        self.false_all_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
        self.n_false_all_closures = 0

        if self.cfg.include_salt:
            self.salt_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
            self.wide_salt = np.zeros(self.closure_vol_shape)
            self.salt_closures_oil = np.zeros(self.closure_vol_shape, dtype="uint8")
            self.salt_closures_gas = np.zeros(self.closure_vol_shape, dtype="uint8")
            self.salt_closures_brine = np.zeros(self.closure_vol_shape, dtype="uint8")
            self.salt_closures_oil_segment_list = list()
            self.salt_closures_gas_segment_list = list()
            self.salt_closures_brine_segment_list = list()
            self.n_salt_closures_oil = 0
            self.n_salt_closures_gas = 0
            self.n_salt_closures_brine = 0

            self.salt_all_closures = np.zeros(self.closure_vol_shape, dtype="uint8")
            self.salt_all_closures_segment_list = list()
            self.n_salt_all_closures = 0

    def create_closure_labels_from_depth_maps(
        self, depth_maps, depth_maps_infilled, max_col_height
    ):
        if self.cfg.verbose:
            print("\n\t... inside insertClosureLabels3D ")
            print(
                f"\t... depth_maps min {depth_maps.min():.2f}, mean {depth_maps.mean():.2f},"
                f" max {depth_maps.max():.2f}, cube_shape {self.cfg.cube_shape}"
            )

        # create 3D cube to hold segmentation results
        closure_segments = np.zeros(self.faults.faulted_lithology.shape, "float32")

        # create grids with grid indices
        ii, jj = self.build_meshgrid()

        # loop through horizons in 'depth_maps'
        voxel_change_count = np.zeros(self.cfg.cube_shape, dtype=np.uint8)
        layers_with_closure = 0

        avg_sand_thickness = list()
        avg_shale_thickness = list()
        avg_unit_thickness = list()
        for ihorizon in range(depth_maps.shape[2] - 1):
            avg_unit_thickness.append(
                np.mean(
                    depth_maps_infilled[..., ihorizon + 1]
                    - depth_maps_infilled[..., ihorizon]
                )
            )

            if self.top_lith_facies[ihorizon] > 0:
                # If facies is not shale, calculate a closure map for the layer
                if self.cfg.verbose:
                    print(
                        f"\n...closure voxels computation for layer {ihorizon} in horizon list."
                    )
                avg_sand_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )
                # compute a closure map
                # - identical to top structure map when not in closure, 'max flooding' depth when in closure
                # - use thicknesses converted to samples instead of ft or ms
                # - assumes that fault intersections are inserted in input map with value of 0.
                # - assumes that input map values represent depth (i.e., bigger values are deeper)
                top_structure_depth_map = depth_maps[:, :, ihorizon].copy()
                top_structure_depth_map[np.isnan(top_structure_depth_map)] = (
                    0.0  # replace nans with 0.
                )
                top_structure_depth_map /= float(self.cfg.digi)
                if self.cfg.partial_voxels:
                    top_structure_depth_map -= (
                        1.0  # account for voxels partially in layer
                    )
                base_structure_depth_map = depth_maps_infilled[
                    :, :, ihorizon + 1
                ].copy()
                base_structure_depth_map[np.isnan(top_structure_depth_map)] = (
                    0.0  # replace nans with 0.
                )
                base_structure_depth_map /= float(self.cfg.digi)
                print(
                    " ...inside create_closure_labels_from_depth_maps... ihorizon, self.top_lith_facies[ihorizon] = ",
                    ihorizon,
                    self.top_lith_facies[ihorizon],
                )
                # if there is non-zero thickness between top/base closure
                if top_structure_depth_map.min() != top_structure_depth_map.max():
                    max_column = max_col_height[ihorizon] / self.cfg.digi
                    if self.cfg.verbose:
                        print(
                            f"   ...avg depth for layer {ihorizon}.",
                            top_structure_depth_map.mean(),
                        )
                    if self.cfg.verbose:
                        print(
                            f"   ...maximum column height for layer {ihorizon}.",
                            max_column,
                        )

                    if ihorizon == 27000 or ihorizon == 1000:
                        closure_depth_map = _flood_fill(
                            top_structure_depth_map,
                            max_column_height=max_column,
                            verbose=True,
                            debug=True,
                        )
                    else:
                        closure_depth_map = _flood_fill(
                            top_structure_depth_map, max_column_height=max_column
                        )
                    closure_depth_map[closure_depth_map == 0] = top_structure_depth_map[
                        closure_depth_map == 0
                    ]
                    closure_depth_map[closure_depth_map == 1] = top_structure_depth_map[
                        closure_depth_map == 1
                    ]
                    closure_depth_map[closure_depth_map == 1e5] = (
                        top_structure_depth_map[closure_depth_map == 1e5]
                    )
                    # Select the maximum value between the top sand map and the flood-filled closure map
                    closure_depth_map = np.max(
                        np.dstack((closure_depth_map, top_structure_depth_map)), axis=-1
                    )
                    closure_depth_map = np.min(
                        np.dstack((closure_depth_map, base_structure_depth_map)),
                        axis=-1,
                    )
                    if self.cfg.verbose:
                        print(
                            f"\n    ... layer {ihorizon},"
                            f"\n\ttop structure map min, max {top_structure_depth_map.min():.2f},"
                            f" {top_structure_depth_map.max():.2f}\n\tclosure_depth_map min, max"
                            f" {closure_depth_map.min():.2f} {closure_depth_map.max()}"
                        )
                    closure_thickness = closure_depth_map - top_structure_depth_map
                    closure_thickness_no_nan = closure_thickness[
                        ~np.isnan(closure_thickness)
                    ]
                    max_closure = int(np.around(closure_thickness_no_nan.max(), 0))
                    if self.cfg.verbose:
                        print(f"    ... layer {ihorizon}, max_closure {max_closure}")

                    # locate 3D zone in closure after checking that closures exist for this horizon
                    # if False in (top_structure_depth_map == closure_depth_map):
                    if max_closure > 0:
                        # locate voxels anywhere in layer where top_structure_depth_map < closure_depth_map
                        # put label in cube between top_structure_depth_map and closure_depth_map
                        top_structure_depth_map_integer = top_structure_depth_map
                        closure_depth_map_integer = closure_depth_map

                        if self.cfg.verbose:
                            closure_map_min = closure_depth_map_integer[
                                closure_depth_map_integer > 0.1
                            ].min()
                            closure_map_max = closure_depth_map_integer[
                                closure_depth_map_integer > 0.1
                            ].max()
                            print(
                                f"\t... (2) layer: {ihorizon}, max_closure; {max_closure}, top structure map min, "
                                f"max: {top_structure_depth_map.min()}, {top_structure_depth_map_integer.max()},"
                                f" closure map min, max: {closure_map_min}, {closure_map_max}"
                            )

                        slices_with_substitution = 0
                        print("    ... max_closure: {}".format(max_closure))
                        for k in range(
                            max_closure + 1
                        ):  # add one more sample than seemingly needed for round-off
                            # Subtract 2 from the closure cube shape since adding one later
                            horizon_slice = (k + top_structure_depth_map).clip(
                                0, closure_segments.shape[2] - 2
                            )
                            sublayer_kk = horizon_slice[
                                horizon_slice < closure_depth_map.astype("int")
                            ]
                            sublayer_ii = ii[
                                horizon_slice < closure_depth_map.astype("int")
                            ]
                            sublayer_jj = jj[
                                horizon_slice < closure_depth_map.astype("int")
                            ]

                            if sublayer_ii.size > 0:
                                slices_with_substitution += 1

                                i_indices = sublayer_ii
                                j_indices = sublayer_jj
                                k_indices = sublayer_kk + 1

                                try:
                                    closure_segments[
                                        i_indices, j_indices, k_indices.astype("int")
                                    ] += 1.0
                                    voxel_change_count[
                                        i_indices, j_indices, k_indices.astype("int")
                                    ] += 1
                                except IndexError:
                                    print("\nIndex is out of bounds.")
                                    print(f"\tclosure_segments: {closure_segments}")
                                    print(f"\tvoxel_change_count: {voxel_change_count}")
                                    print(f"\ti_indices: {i_indices}")
                                    print(f"\tj_indices: {j_indices}")
                                    print(f"\tk_indices: {k_indices.astype('int')}")
                                    pass

                        if slices_with_substitution > 0:
                            layers_with_closure += 1

                        if self.cfg.verbose:
                            print(
                                "    ... finished putting closures in closures_segments for layer ...",
                                ihorizon,
                            )

                    else:
                        continue
            else:
                # Calculate shale unit thicknesses
                avg_shale_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )

        if len(avg_sand_thickness) == 0:
            avg_sand_thickness = 0
        if len(avg_shale_thickness) == 0:
            avg_shale_thickness = 0
        if len(avg_unit_thickness) == 0:
            avg_unit_thickness = 0
        self.cfg.write_to_logfile(
            f"Sand Unit Thickness (m): mean: {np.mean(avg_sand_thickness):.2f}, "
            f"std: {np.std(avg_sand_thickness):.2f}, min: {np.nanmin(avg_sand_thickness):.2f}, "
            f"max: {np.max(avg_sand_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Shale Unit Thickness (m): mean: {np.mean(avg_shale_thickness):.2f}, "
            f"std: {np.std(avg_shale_thickness):.2f}, min: {np.min(avg_shale_thickness):.2f}, "
            f"max: {np.max(avg_shale_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Overall Unit Thickness (m): mean: {np.mean(avg_unit_thickness):.2f}, "
            f"std: {np.std(avg_unit_thickness):.2f}, min: {np.min(avg_unit_thickness):.2f}, "
            f"max: {np.max(avg_unit_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_mean",
            val=np.mean(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_std",
            val=np.std(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_min",
            val=np.min(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_max",
            val=np.max(avg_sand_thickness),
        )
        #
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_mean",
            val=np.mean(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_std",
            val=np.std(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_min",
            val=np.min(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_max",
            val=np.max(avg_shale_thickness),
        )

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_mean",
            val=np.mean(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_std",
            val=np.std(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_min",
            val=np.min(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_max",
            val=np.max(avg_unit_thickness),
        )

        non_zero_pixels = closure_segments[closure_segments != 0.0].shape[0]
        pct_non_zero = float(non_zero_pixels) / (
            closure_segments.shape[0]
            * closure_segments.shape[1]
            * closure_segments.shape[2]
        )
        if self.cfg.verbose:
            print(
                "    ...closure_segments min {}, mean {}, max {}, % non-zero {}".format(
                    closure_segments.min(),
                    closure_segments.mean(),
                    closure_segments.max(),
                    pct_non_zero,
                )
            )

        print(f"\t... layers_with_closure {layers_with_closure}")
        print("\t... finished putting closures in closure_segments ...\n")

        if self.cfg.verbose:
            print(
                f"\n   ...closure segments created. min: {closure_segments.min()}, "
                f"mean: {closure_segments.mean():.2f}, max: {closure_segments.max()}"
                f" voxel count: {closure_segments[closure_segments != 0].shape}"
            )

        return closure_segments

    def create_closure_labels_from_all_depth_maps(
        self, depth_maps, depth_maps_infilled, max_col_height
    ):
        if self.cfg.verbose:
            print("\n\t... inside insertClosureLabels3D ")
            print(
                f"\t... depth_maps min {depth_maps.min():.2f}, mean {depth_maps.mean():.2f},"
                f" max {depth_maps.max():.2f}, cube_shape {self.cfg.cube_shape}"
            )

        # create 3D cube to hold segmentation results
        closure_segments = np.zeros(self.faults.faulted_lithology.shape, "float32")

        # create grids with grid indices
        ii, jj = self.build_meshgrid()

        # loop through horizons in 'depth_maps'
        voxel_change_count = np.zeros(self.cfg.cube_shape, dtype=np.uint8)
        layers_with_closure = 0

        avg_sand_thickness = list()
        avg_shale_thickness = list()
        avg_unit_thickness = list()
        for ihorizon in range(depth_maps.shape[2] - 1):
            avg_unit_thickness.append(
                np.mean(
                    depth_maps_infilled[..., ihorizon + 1]
                    - depth_maps_infilled[..., ihorizon]
                )
            )
            # calculate a closure map for the layer
            if self.cfg.verbose:
                print(
                    f"\n...closure voxels computation for layer {ihorizon} in horizon list."
                )

            # compute a closure map
            # - identical to top structure map when not in closure, 'max flooding' depth when in closure
            # - use thicknesses converted to samples instead of ft or ms
            # - assumes that fault intersections are inserted in input map with value of 0.
            # - assumes that input map values represent depth (i.e., bigger values are deeper)
            top_structure_depth_map = depth_maps[:, :, ihorizon].copy()
            top_structure_depth_map[np.isnan(top_structure_depth_map)] = (
                0.0  # replace nans with 0.
            )
            top_structure_depth_map /= float(self.cfg.digi)
            if self.cfg.partial_voxels:
                top_structure_depth_map -= 1.0  # account for voxels partially in layer
            base_structure_depth_map = depth_maps_infilled[:, :, ihorizon + 1].copy()
            base_structure_depth_map[np.isnan(top_structure_depth_map)] = (
                0.0  # replace nans with 0.
            )
            base_structure_depth_map /= float(self.cfg.digi)
            print(
                " ...inside create_closure_labels_from_depth_maps... ihorizon = ",
                ihorizon,
            )
            # if there is non-zero thickness between top/base closure
            if top_structure_depth_map.min() != top_structure_depth_map.max():
                max_column = max_col_height[ihorizon] / self.cfg.digi
                if self.cfg.verbose:
                    print(
                        f"   ...avg depth for layer {ihorizon}.",
                        top_structure_depth_map.mean(),
                    )
                if self.cfg.verbose:
                    print(
                        f"   ...maximum column height for layer {ihorizon}.", max_column
                    )

                if ihorizon == 27000 or ihorizon == 1000:
                    closure_depth_map = _flood_fill(
                        top_structure_depth_map,
                        max_column_height=max_column,
                        verbose=True,
                        debug=True,
                    )
                else:
                    closure_depth_map = _flood_fill(
                        top_structure_depth_map, max_column_height=max_column
                    )
                closure_depth_map[closure_depth_map == 0] = top_structure_depth_map[
                    closure_depth_map == 0
                ]
                closure_depth_map[closure_depth_map == 1] = top_structure_depth_map[
                    closure_depth_map == 1
                ]
                closure_depth_map[closure_depth_map == 1e5] = top_structure_depth_map[
                    closure_depth_map == 1e5
                ]
                # Select the maximum value between the top sand map and the flood-filled closure map
                closure_depth_map = np.max(
                    np.dstack((closure_depth_map, top_structure_depth_map)), axis=-1
                )
                closure_depth_map = np.min(
                    np.dstack((closure_depth_map, base_structure_depth_map)), axis=-1
                )
                if self.cfg.verbose:
                    print(
                        f"\n    ... layer {ihorizon},"
                        f"\n\ttop structure map min, max {top_structure_depth_map.min():.2f},"
                        f" {top_structure_depth_map.max():.2f}\n\tclosure_depth_map min, max"
                        f" {closure_depth_map.min():.2f} {closure_depth_map.max()}"
                    )
                closure_thickness = closure_depth_map - top_structure_depth_map
                closure_thickness_no_nan = closure_thickness[
                    ~np.isnan(closure_thickness)
                ]
                max_closure = int(np.around(closure_thickness_no_nan.max(), 0))
                if self.cfg.verbose:
                    print(f"    ... layer {ihorizon}, max_closure {max_closure}")

                # locate 3D zone in closure after checking that closures exist for this horizon
                # if False in (top_structure_depth_map == closure_depth_map):
                if max_closure > 0:
                    # locate voxels anywhere in layer where top_structure_depth_map < closure_depth_map
                    # put label in cube between top_structure_depth_map and closure_depth_map
                    top_structure_depth_map_integer = top_structure_depth_map
                    closure_depth_map_integer = closure_depth_map

                    if self.cfg.verbose:
                        closure_map_min = closure_depth_map_integer[
                            closure_depth_map_integer > 0.1
                        ].min()
                        closure_map_max = closure_depth_map_integer[
                            closure_depth_map_integer > 0.1
                        ].max()
                        print(
                            f"\t... (2) layer: {ihorizon}, max_closure; {max_closure}, top structure map min, "
                            f"max: {top_structure_depth_map.min()}, {top_structure_depth_map_integer.max()},"
                            f" closure map min, max: {closure_map_min}, {closure_map_max}"
                        )

                    slices_with_substitution = 0
                    print("    ... max_closure: {}".format(max_closure))
                    for k in range(
                        max_closure + 1
                    ):  # add one more sample than seemingly needed for round-off
                        # Subtract 2 from the closure cube shape since adding one later
                        horizon_slice = (k + top_structure_depth_map).clip(
                            0, closure_segments.shape[2] - 2
                        )
                        sublayer_kk = horizon_slice[
                            horizon_slice < closure_depth_map.astype("int")
                        ]
                        sublayer_ii = ii[
                            horizon_slice < closure_depth_map.astype("int")
                        ]
                        sublayer_jj = jj[
                            horizon_slice < closure_depth_map.astype("int")
                        ]

                        if sublayer_ii.size > 0:
                            slices_with_substitution += 1

                            i_indices = sublayer_ii
                            j_indices = sublayer_jj
                            k_indices = sublayer_kk + 1

                            try:
                                closure_segments[
                                    i_indices, j_indices, k_indices.astype("int")
                                ] += 1.0
                                voxel_change_count[
                                    i_indices, j_indices, k_indices.astype("int")
                                ] += 1
                            except IndexError:
                                print("\nIndex is out of bounds.")
                                print(f"\tclosure_segments: {closure_segments}")
                                print(f"\tvoxel_change_count: {voxel_change_count}")
                                print(f"\ti_indices: {i_indices}")
                                print(f"\tj_indices: {j_indices}")
                                print(f"\tk_indices: {k_indices.astype('int')}")
                                pass

                    if slices_with_substitution > 0:
                        layers_with_closure += 1

                    if self.cfg.verbose:
                        print(
                            "    ... finished putting closures in closures_segments for layer ...",
                            ihorizon,
                        )

                else:
                    continue
            else:
                # Calculate shale unit thicknesses
                avg_shale_thickness.append(
                    np.mean(
                        depth_maps_infilled[..., ihorizon + 1]
                        - depth_maps_infilled[..., ihorizon]
                    )
                )

        if len(avg_sand_thickness) == 0:
            avg_sand_thickness = 0
        if len(avg_shale_thickness) == 0:
            avg_shale_thickness = 0
        if len(avg_unit_thickness) == 0:
            avg_unit_thickness = 0
        self.cfg.write_to_logfile(
            f"Sand Unit Thickness (m): mean: {np.mean(avg_sand_thickness):.2f}, "
            f"std: {np.std(avg_sand_thickness):.2f}, min: {np.nanmin(avg_sand_thickness):.2f}, "
            f"max: {np.max(avg_sand_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Shale Unit Thickness (m): mean: {np.mean(avg_shale_thickness):.2f}, "
            f"std: {np.std(avg_shale_thickness):.2f}, min: {np.min(avg_shale_thickness):.2f}, "
            f"max: {np.max(avg_shale_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            f"Overall Unit Thickness (m): mean: {np.mean(avg_unit_thickness):.2f}, "
            f"std: {np.std(avg_unit_thickness):.2f}, min: {np.min(avg_unit_thickness):.2f}, "
            f"max: {np.max(avg_unit_thickness):.2f}"
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_mean",
            val=np.mean(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_std",
            val=np.std(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_min",
            val=np.min(avg_sand_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="sand_unit_thickness_combined_max",
            val=np.max(avg_sand_thickness),
        )
        #
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_mean",
            val=np.mean(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_std",
            val=np.std(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_min",
            val=np.min(avg_shale_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="shale_unit_thickness_combined_max",
            val=np.max(avg_shale_thickness),
        )

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_mean",
            val=np.mean(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_std",
            val=np.std(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_min",
            val=np.min(avg_unit_thickness),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="overall_unit_thickness_combined_max",
            val=np.max(avg_unit_thickness),
        )

        non_zero_pixels = closure_segments[closure_segments != 0.0].shape[0]
        pct_non_zero = float(non_zero_pixels) / (
            closure_segments.shape[0]
            * closure_segments.shape[1]
            * closure_segments.shape[2]
        )
        if self.cfg.verbose:
            print(
                "    ...closure_segments min {}, mean {}, max {}, % non-zero {}".format(
                    closure_segments.min(),
                    closure_segments.mean(),
                    closure_segments.max(),
                    pct_non_zero,
                )
            )

        print(f"\t... layers_with_closure {layers_with_closure}")
        print("\t... finished putting closures in closure_segments ...\n")

        if self.cfg.verbose:
            print(
                f"\n   ...closure segments created. min: {closure_segments.min()}, "
                f"mean: {closure_segments.mean():.2f}, max: {closure_segments.max()}"
                f" voxel count: {closure_segments[closure_segments != 0].shape}"
            )

        return closure_segments

    def find_top_lith_horizons(self):
        """
        Find horizons which are the top of layers where the lithology changes

        Combine layers of the same lithology and retain the top of these new layers for closure calculations.
        """
        top_lith_indices = list(np.array(self.onlap_list) - 1)
        for i, _ in enumerate(self.facies[:-1]):
            if i == 0:
                continue
            print(
                f"i: {i}, sand_layer_label[i-1]: {self.facies[i - 1]},"
                f" sand_layer_label[i]: {self.facies[i]}"
            )
            if self.facies[i] != self.facies[i - 1]:
                top_lith_indices.append(i)
                if self.cfg.verbose:
                    print(
                        "  ... layer lith different than layer above it. i = {}".format(
                            i
                        )
                    )
        top_lith_indices.sort()
        if self.cfg.verbose:
            print(
                "\n   ...layers selected for closure computations...\n",
                top_lith_indices,
            )
        self.top_lith_indices = np.array(top_lith_indices)
        self.top_lith_facies = self.facies[top_lith_indices]

        # return top_lith_indices

    def create_closures(self):
        if self.cfg.verbose:
            print("\n\n ... create 3D labels for closure")

        # Convert nan to 0's
        old_depth_maps = np.nan_to_num(self.faults.faulted_depth_maps[:], copy=True)
        old_depth_maps_gaps = np.nan_to_num(
            self.faults.faulted_depth_maps_gaps[:], copy=True
        )

        # Convert from samples to units
        old_depth_maps_gaps = self.convert_map_from_samples_to_units(
            old_depth_maps_gaps
        )
        old_depth_maps = self.convert_map_from_samples_to_units(old_depth_maps)

        # keep only horizons corresponding to top of layers where lithology changes
        self.find_top_lith_horizons()
        all_lith_indices = np.arange(old_depth_maps.shape[-1])
        import sys

        print("All lith indices (last, then all):", self.facies[-1], all_lith_indices)
        sys.stdout.flush()

        depth_maps_gaps_top_lith = old_depth_maps_gaps[
            :, :, self.top_lith_indices
        ].copy()
        depth_maps_gaps_all_lith = old_depth_maps_gaps[:, :, all_lith_indices].copy()
        depth_maps_top_lith = old_depth_maps[:, :, self.top_lith_indices].copy()
        depth_maps_all_lith = old_depth_maps[:, :, all_lith_indices].copy()
        max_column_heights = variable_max_column_height(
            self.top_lith_indices,
            self.faults.faulted_depth_maps_gaps.shape[-1],
            self.cfg.max_column_height[0],
            self.cfg.max_column_height[1],
        )
        all_max_column_heights = variable_max_column_height(
            all_lith_indices,
            self.faults.faulted_depth_maps_gaps.shape[-1],
            self.cfg.max_column_height[0],
            self.cfg.max_column_height[1],
        )

        if self.cfg.verbose:
            print("\n   ...facies for closure computations...\n", self.top_lith_facies)
            print(
                "\n   ...max column heights for closure computations...\n",
                max_column_heights,
            )

        self.closure_segments[:] = self.create_closure_labels_from_depth_maps(
            depth_maps_gaps_top_lith, depth_maps_top_lith, max_column_heights
        )

        self.all_closure_segments[:] = self.create_closure_labels_from_all_depth_maps(
            depth_maps_gaps_all_lith, depth_maps_all_lith, all_max_column_heights
        )

        if self.cfg.verbose:
            print(
                "     ...+++... number of nan's in depth_maps_gaps before insertClosureLabels3D ...+++... {}".format(
                    old_depth_maps_gaps[np.isnan(old_depth_maps_gaps)].shape
                )
            )
            print(
                "     ...+++... number of nan's in depth_maps_gaps after insertClosureLabels3D ...+++... {}".format(
                    self.faults.faulted_depth_maps_gaps[
                        np.isnan(self.faults.faulted_depth_maps_gaps)
                    ].shape
                )
            )
            print(
                "     ...+++... number of nan's in depth_maps after insertClosureLabels3D ...+++... {}".format(
                    self.faults.faulted_depth_maps[
                        np.isnan(self.faults.faulted_depth_maps)
                    ].shape
                )
            )
            _closure_segments = self.closure_segments[:]
            print(
                "     ...+++... number of closure voxels in self.closure_segments ...+++... {}".format(
                    _closure_segments[_closure_segments > 0.0].shape
                )
            )
            del _closure_segments

        labels_clean, self.closure_segments[:] = self.segment_closures(
            self.closure_segments[:], remove_shale=True
        )
        label_values, labels_clean = self.parse_label_values_and_counts(labels_clean)

        labels_clean_all, self.all_closure_segments[:] = self.segment_closures(
            self.all_closure_segments[:], remove_shale=False
        )
        label_values_all, labels_clean_all = self.parse_label_values_and_counts(
            labels_clean_all
        )
        self.write_cube_to_disk(self.all_closure_segments[:], "all_closure_segments")

        # Assign fluid types
        (
            self.oil_closures[:],
            self.gas_closures[:],
            self.brine_closures[:],
        ) = self.assign_fluid_types(label_values, labels_clean)
        all_closures_final = (labels_clean_all != 0).astype("uint8")

        # Identify closures by type (simple, faulted, onlap or salt bounded)
        self.find_faulted_closures(label_values, labels_clean)
        self.find_onlap_closures(label_values, labels_clean)
        self.find_simple_closures(label_values, labels_clean)
        self.find_false_closures(label_values, labels_clean)

        self.find_faulted_all_closures(label_values_all, labels_clean_all)
        self.find_onlap_all_closures(label_values_all, labels_clean_all)
        self.find_simple_all_closures(label_values_all, labels_clean_all)
        self.find_false_all_closures(label_values_all, labels_clean_all)

        if self.cfg.include_salt:
            self.find_salt_bounded_closures(label_values, labels_clean)
            self.find_salt_bounded_all_closures(label_values_all, labels_clean_all)

        # Remove false closures from oil & gas closure cubes
        if self.n_false_closures_oil > 0:
            print(f"Removing {self.n_false_closures_oil} false oil closures")
            self.oil_closures[self.false_closures_oil == 1] = 0.0
        if self.n_false_closures_gas > 0:
            print(f"Removing {self.n_false_closures_gas} false gas closures")
            self.gas_closures[self.false_closures_gas == 1] = 0.0

        # Remove false closures from allclosure cube
        if self.n_false_all_closures > 0:
            print(f"Removing {self.n_false_all_closures} false all closures")
            self.all_closure_segments[self.false_all_closures == 1] = 0.0

        # Create a closure cube with voxel count as labels, and include closure type in decimal
        # e.g. simple closure of size 5000 = 5000.1
        #      faulted closure of size 5000 = 5000.2
        #      onlap closure of size 5000 = 5000.3
        #      salt-bounded closure of size 5000 = 5000.4
        hc_closure_codes = np.zeros_like(self.gas_closures, dtype="float32")

        # AZ: COULD RUN THESE CLOSURE SIZE FILTERS ON ALL_CLOSURES, IF DESIRED

        if "simple" in self.cfg.closure_types:
            print("Filtering 4 Way Closures")
            (
                self.simple_closures_oil[:],
                self.n_4way_closures_oil,
            ) = self.closure_size_filter(
                self.simple_closures_oil[:],
                self.cfg.closure_min_voxels_simple,
                self.n_4way_closures_oil,
            )
            (
                self.simple_closures_gas[:],
                self.n_4way_closures_gas,
            ) = self.closure_size_filter(
                self.simple_closures_gas[:],
                self.cfg.closure_min_voxels_simple,
                self.n_4way_closures_gas,
            )

            # Add simple closures to closure code cube
            hc_closures = (
                self.simple_closures_oil[:] + self.simple_closures_gas[:]
            ).astype("float32")
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.1
            )
        else:  # if closure type not in config, set HC closures to 0
            self.simple_closures_oil[:] *= 0
            self.simple_closures_gas[:] *= 0
            self.simple_all_closures[:] *= 0

        self.oil_closures[self.simple_closures_oil[:] > 0.0] = 1.0
        self.oil_closures[self.simple_closures_oil[:] < 0.0] = 0.0
        self.gas_closures[self.simple_closures_gas[:] > 0.0] = 1.0
        self.gas_closures[self.simple_closures_gas[:] < 0.0] = 0.0

        all_closures_final[self.simple_all_closures[:] > 0.0] = 1.0
        all_closures_final[self.simple_all_closures[:] < 0.0] = 0.0

        if "faulted" in self.cfg.closure_types:
            print("Filtering 4 Way Closures")
            # Grow the faulted closures to the fault planes
            self.faulted_closures_oil[:] = self.grow_to_fault2(
                self.faulted_closures_oil[:]
            )
            self.faulted_closures_gas[:] = self.grow_to_fault2(
                self.faulted_closures_gas[:]
            )

            (
                self.faulted_closures_oil[:],
                self.n_fault_closures_oil,
            ) = self.closure_size_filter(
                self.faulted_closures_oil[:],
                self.cfg.closure_min_voxels_faulted,
                self.n_fault_closures_oil,
            )
            (
                self.faulted_closures_gas[:],
                self.n_fault_closures_gas,
            ) = self.closure_size_filter(
                self.faulted_closures_gas[:],
                self.cfg.closure_min_voxels_faulted,
                self.n_fault_closures_gas,
            )

            self.faulted_all_closures[:] = self.grow_to_fault2(
                self.faulted_all_closures[:],
                grow_only_sand_closures=False,
                remove_small_closures=False,
            )

            # Add faulted closures to closure code cube
            hc_closures = self.faulted_closures_oil[:] + self.faulted_closures_gas[:]
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.2
            )
        else:  # if closure type not in config, set HC closures to 0
            self.faulted_closures_oil[:] *= 0
            self.faulted_closures_gas[:] *= 0
            self.faulted_all_closures[:] *= 0

        self.oil_closures[self.faulted_closures_oil[:] > 0.0] = 1.0
        self.oil_closures[self.faulted_closures_oil[:] < 0.0] = 0.0
        self.gas_closures[self.faulted_closures_gas[:] > 0.0] = 1.0
        self.gas_closures[self.faulted_closures_gas[:] < 0.0] = 0.0

        all_closures_final[self.faulted_all_closures[:] > 0.0] = 1.0
        all_closures_final[self.faulted_all_closures[:] < 0.0] = 0.0

        if "onlap" in self.cfg.closure_types:
            print("Filtering Onlap Closures")
            (
                self.onlap_closures_oil[:],
                self.n_onlap_closures_oil,
            ) = self.closure_size_filter(
                self.onlap_closures_oil[:],
                self.cfg.closure_min_voxels_onlap,
                self.n_onlap_closures_oil,
            )
            (
                self.onlap_closures_gas[:],
                self.n_onlap_closures_gas,
            ) = self.closure_size_filter(
                self.onlap_closures_gas[:],
                self.cfg.closure_min_voxels_onlap,
                self.n_onlap_closures_gas,
            )

            # Add faulted closures to closure code cube
            hc_closures = self.onlap_closures_oil[:] + self.onlap_closures_gas[:]
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.3
            )
            # labels = labels.astype('float32')
            # if num > 0:
            #     for x in range(1, num + 1):
            #         y = 0.3 + labels[labels == x].size
            #         labels[labels == x] = y
            #     hc_closure_codes += labels
        else:  # if closure type not in config, set HC closures to 0
            self.onlap_closures_oil[:] *= 0
            self.onlap_closures_gas[:] *= 0
            self.onlap_all_closures[:] *= 0

        self.oil_closures[self.onlap_closures_oil[:] > 0.0] = 1.0
        self.oil_closures[self.onlap_closures_oil[:] < 0.0] = 0.0
        self.gas_closures[self.onlap_closures_gas[:] > 0.0] = 1.0
        self.gas_closures[self.onlap_closures_gas[:] < 0.0] = 0.0
        all_closures_final[self.onlap_all_closures[:] > 0.0] = 1.0
        all_closures_final[self.onlap_all_closures[:] < 0.0] = 0.0

        if self.cfg.include_salt:
            # Grow the salt-bounded closures to the salt body
            salt_closures_oil_grown = np.zeros_like(self.salt_closures_oil[:])
            salt_closures_gas_grown = np.zeros_like(self.salt_closures_gas[:])

            if np.max(self.salt_closures_oil[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_oil[:], "salt_closures_oil_initial"
                )
                print(
                    f"Salt-bounded Oil Closure voxel count: {self.salt_closures_oil[:][self.salt_closures_oil[:] > 0].size}"
                )
                salt_closures_oil_grown = self.grow_to_salt(self.salt_closures_oil[:])
                self.salt_closures_oil[:] = salt_closures_oil_grown
                print(
                    f"Salt-bounded Oil Closure voxel count: {self.salt_closures_oil[:][self.salt_closures_oil[:] > 0].size}"
                )
            if np.max(self.salt_closures_gas[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_gas[:], "salt_closures_gas_initial"
                )
                print(
                    f"Salt-bounded Gas Closure voxel count: {self.salt_closures_gas[:][self.salt_closures_gas[:] > 0].size}"
                )
                salt_closures_gas_grown = self.grow_to_salt(self.salt_closures_gas[:])
                self.salt_closures_gas[:] = salt_closures_gas_grown
                print(
                    f"Salt-bounded Gas Closure voxel count: {self.salt_closures_gas[:][self.salt_closures_gas[:] > 1].size}"
                )
            if np.max(self.salt_all_closures[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_all_closures[:], "salt_all_closures_initial"
                )  # maybe remove later
                print(
                    f"Salt-bounded All Closure voxel count: {self.salt_all_closures[:][self.salt_all_closures[:] > 0].size}"
                )
                salt_all_closures_grown = self.grow_to_salt(self.salt_all_closures[:])
                self.salt_all_closures[:] = salt_all_closures_grown
                print(
                    f"Salt-bounded All Closure voxel count: {self.salt_all_closures[:][self.salt_all_closures[:] > 1].size}"
                )
            else:
                salt_all_closures_grown = np.zeros_like(self.salt_all_closures)

            if np.max(self.salt_closures_oil[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_oil[:], "salt_closures_oil_grown"
                )
            if np.max(self.salt_closures_gas[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_closures_gas[:], "salt_closures_gas_grown"
                )
            if np.max(self.salt_all_closures[:]) > 0.0:
                self.write_cube_to_disk(
                    self.salt_all_closures[:], "salt_all_closures_grown"
                )  # maybe remove later

            (
                self.salt_closures_oil[:],
                self.n_salt_closures_oil,
            ) = self.closure_size_filter(
                self.salt_closures_oil[:],
                self.cfg.closure_min_voxels,
                self.n_salt_closures_oil,
            )
            (
                self.salt_closures_gas[:],
                self.n_salt_closures_gas,
            ) = self.closure_size_filter(
                self.salt_closures_gas[:],
                self.cfg.closure_min_voxels,
                self.n_salt_closures_gas,
            )

            # Append salt-bounded closures to main closure cubes for oil and gas
            if np.max(salt_closures_oil_grown) > 0.0:
                self.oil_closures[salt_closures_oil_grown > 0.0] = 1.0
                self.oil_closures[salt_closures_oil_grown < 0.0] = 0.0
            if np.max(salt_closures_gas_grown) > 0.0:
                self.gas_closures[salt_closures_gas_grown > 0.0] = 1.0
                self.gas_closures[salt_closures_gas_grown < 0.0] = 0.0
            if np.max(salt_all_closures_grown) > 0.0:
                all_closures_final[salt_all_closures_grown > 0.0] = 1.0
                all_closures_final[salt_all_closures_grown < 0.0] = 0.0

            # Add faulted closures to closure code cube
            hc_closures = self.salt_closures_oil[:] + self.salt_closures_gas[:]
            labels, num = measure.label(
                hc_closures, connectivity=2, background=0, return_num=True
            )
            hc_closure_codes = self.parse_closure_codes(
                hc_closure_codes, labels, num, code=0.4
            )

        # Write hc_closure_codes to disk
        self.write_cube_to_disk(hc_closure_codes, "closure_segments_hc_voxelcount")

        # Create closure volumes by type
        if self.simple_closures[:] is None:
            self.simple_closures[:] = self.simple_closures_oil[:].astype("uint8")
        else:
            self.simple_closures[:] += self.simple_closures_oil[:].astype("uint8")
        self.simple_closures[:] += self.simple_closures_gas[:].astype("uint8")
        self.simple_closures[:] += self.simple_closures_brine[:].astype("uint8")
        # Onlap closures
        if self.strat_closures is None:
            self.strat_closures[:] = self.onlap_closures_oil[:].astype("uint8")
        else:
            self.strat_closures[:] += self.onlap_closures_oil[:].astype("uint8")
        self.strat_closures[:] += self.onlap_closures_gas[:].astype("uint8")
        self.strat_closures[:] += self.onlap_closures_brine[:].astype("uint8")
        # Fault closures
        if self.fault_closures is None:
            self.fault_closures[:] = self.faulted_closures_oil[:].astype("uint8")
        else:
            self.fault_closures[:] += self.faulted_closures_oil[:].astype("uint8")
        self.fault_closures[:] += self.faulted_closures_gas[:].astype("uint8")
        self.fault_closures[:] += self.faulted_closures_brine[:].astype("uint8")

        # Salt-bounded closures
        if self.cfg.include_salt:
            if self.salt_closures is None:
                self.salt_closures[:] = self.salt_closures_oil[:].astype("uint8")
            else:
                self.salt_closures[:] += self.salt_closures_oil[:].astype("uint8")
            self.salt_closures[:] += self.salt_closures_gas[:].astype("uint8")

        # Convert closure cubes from int16 to uint8 for writing to disk
        self.closure_segments[:] = self.closure_segments[:].astype("uint8")

        # add any oil/gas/brine closures into all_closures_final in case missed
        all_closures_final[:][self.oil_closures[:] > 0] = 1
        all_closures_final[:][self.gas_closures[:] > 0] = 1
        all_closures_final[:][self.gas_closures[:] > 0] = 1
        # Write all_closures_final to disk
        self.write_cube_to_disk(all_closures_final.astype("uint8"), "trap_label")

        # add any oil/gas/brine closures into reservoir in case missed
        self.faults.reservoir[:][self.oil_closures[:] > 0] = 1
        self.faults.reservoir[:][self.gas_closures[:] > 0] = 1
        self.faults.reservoir[:][self.brine_closures[:] > 0] = 1
        # write reservoir_label to disk
        self.write_cube_to_disk(
            self.faults.reservoir[:].astype("uint8"), "reservoir_label"
        )

        if self.cfg.qc_plots:
            from datagenerator.util import plot_xsection
            from datagenerator.util import find_line_with_most_voxels

            # visualize closures QC
            inline_index_cl = find_line_with_most_voxels(
                self.closure_segments, 0.5, self.cfg
            )
            plot_xsection(
                volume=labels_clean,
                maps=self.faults.faulted_depth_maps_gaps,
                line_num=inline_index_cl,
                title="Example Trav through 3D model\nclosures after faulting",
                png_name="QC_plot__AfterFaulting_closure_segments.png",
                cmap="gist_ncar_r",
                cfg=self.cfg,
            )

    def closure_size_filter(self, closure_type, threshold, count):
        labels, num = measure.label(
            closure_type, connectivity=2, background=0, return_num=True
        )
        if (
            num > 0
        ):  # TODO add whether smallest closure is below threshold constraint too
            s = [labels[labels == x].size for x in range(1, 1 + np.max(labels))]
            labels = morphology.remove_small_objects(labels, threshold, connectivity=2)
            t = [labels[labels == x].size for x in range(1, 1 + np.max(labels))]
            print(
                f"Closure sizes before filter: {s}\nThreshold: {threshold}\n"
                f"Closure sizes after filter: {t}"
            )
            count = len(t)
        return labels, count

    def closure_type_info_for_log(self):
        fluid_types = ["oil", "gas", "brine"]
        if "faulted" in self.cfg.closure_types:
            # Faulted closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.faulted_closures_oil[:],
                    self.faulted_closures_gas[:],
                    self.faulted_closures_brine[:],
                ],
                [
                    self.n_fault_closures_oil,
                    self.n_fault_closures_gas,
                    self.n_fault_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_fault_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_fault_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_fault_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_fault_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"Faulted {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

        if "onlap" in self.cfg.closure_types:
            # Onlap Closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.onlap_closures_oil[:],
                    self.onlap_closures_gas[:],
                    self.onlap_closures_brine[:],
                ],
                [
                    self.n_onlap_closures_oil,
                    self.n_onlap_closures_gas,
                    self.n_onlap_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_onlap_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_onlap_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_onlap_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_onlap_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"Onlap {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

        if "simple" in self.cfg.closure_types:
            # Simple Closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.simple_closures_oil[:],
                    self.simple_closures_gas[:],
                    self.simple_closures_brine[:],
                ],
                [
                    self.n_4way_closures_oil,
                    self.n_4way_closures_gas,
                    self.n_4way_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_4way_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_4way_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_4way_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_4way_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"4-Way {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

        if self.cfg.include_salt:
            # Salt-Bounded Closures
            for name, fluid, num in zip(
                fluid_types,
                [
                    self.salt_closures_oil[:],
                    self.salt_closures_gas[:],
                    self.salt_closures_brine[:],
                ],
                [
                    self.n_salt_closures_oil,
                    self.n_salt_closures_gas,
                    self.n_salt_closures_brine,
                ],
            ):
                n_voxels = fluid[fluid[:] > 0.0].size
                msg = f"n_salt_closures_{name}: {num:03d}\n"
                msg += f"n_voxels_salt_closures_{name}: {n_voxels:08d}\n"
                print(msg)
                self.cfg.write_to_logfile(msg)
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_salt_closures_{name}",
                    val=num,
                )
                self.cfg.write_to_logfile(
                    msg=None,
                    mainkey="model_parameters",
                    subkey=f"n_voxels_salt_closures_{name}",
                    val=n_voxels,
                )
                closure_statistics = self.calculate_closure_statistics(
                    fluid, f"Salt {name.capitalize()}"
                )
                if closure_statistics:
                    print(closure_statistics)
                    self.cfg.write_to_logfile(closure_statistics)

    def get_voxel_counts(self, closures):
        next_label = 0
        label_values = [0]
        label_counts = [closures[closures == 0].size]
        for i in range(closures.max() + 1):
            try:
                next_label = closures[closures > next_label].min()
            except (TypeError, ValueError):
                break
            label_values.append(next_label)
            label_counts.append(closures[closures == next_label].size)
            print(
                f"Label: {i}, label_values: {label_values[-1]}, label_counts: {label_counts[-1]}"
            )

        print(
            f"{72 * '*'}\n\tNum Closures: {len(label_counts) - 1}\n\tVoxel counts\n{label_counts[1:]}\n{72 * '*'}"
        )
        for vox_count in label_counts:
            if vox_count < self.cfg.closure_min_voxels:
                print(f"voxel_count: {vox_count}")

    def populate_closure_dict(self, labels, fluid, seismic_nmf=None):
        clist = []
        max_num = np.max(labels)
        if seismic_nmf is not None:
            # calculate ai_gi
            ai, gi = compute_ai_gi(self.cfg, seismic_nmf)
        for i in range(1, max_num + 1):
            _c = np.where(labels == i)
            cl = dict()
            cl["model_id"] = os.path.basename(self.cfg.work_subfolder)
            cl["fluid"] = fluid
            cl["n_voxels"] = len(_c[0])
            # np.min() or x.min() returns type numpy.int64 which SQLITE cannot handle. Convert to int
            cl["x_min"] = int(np.min(_c[0]))
            cl["x_max"] = int(np.max(_c[0]))
            cl["y_min"] = int(np.min(_c[1]))
            cl["y_max"] = int(np.max(_c[1]))
            cl["z_min"] = int(np.min(_c[2]))
            cl["z_max"] = int(np.max(_c[2]))
            cl["zbml_min"] = np.min(self.faults.faulted_depth[_c])
            cl["zbml_max"] = np.max(self.faults.faulted_depth[_c])
            cl["zbml_avg"] = np.mean(self.faults.faulted_depth[_c])
            cl["zbml_std"] = np.std(self.faults.faulted_depth[_c])
            cl["zbml_25pct"] = np.percentile(self.faults.faulted_depth[_c], 25)
            cl["zbml_median"] = np.percentile(self.faults.faulted_depth[_c], 50)
            cl["zbml_75pct"] = np.percentile(self.faults.faulted_depth[_c], 75)
            cl["ng_min"] = np.min(self.faults.faulted_net_to_gross[_c])
            cl["ng_max"] = np.max(self.faults.faulted_net_to_gross[_c])
            cl["ng_avg"] = np.mean(self.faults.faulted_net_to_gross[_c])
            cl["ng_std"] = np.std(self.faults.faulted_net_to_gross[_c])
            cl["ng_25pct"] = np.percentile(self.faults.faulted_net_to_gross[_c], 25)
            cl["ng_median"] = np.median(self.faults.faulted_net_to_gross[_c])
            cl["ng_75pct"] = np.percentile(self.faults.faulted_net_to_gross[_c], 75)
            # Check for intersections with faults, salt and onlaps for closure type
            cl["intersects_fault"] = False
            cl["intersects_onlap"] = False
            cl["intersects_salt"] = False
            if np.max(self.wide_faults[_c] > 0):
                cl["intersects_fault"] = True
            if np.max(self.onlaps_upward[_c] > 0):
                cl["intersects_onlap"] = True
            if self.cfg.include_salt and np.max(self.wide_salt[_c] > 0):
                cl["intersects_salt"] = True

            if seismic_nmf is not None:
                # Using only the top of the closure, calculate seismic properties
                labels_copy = labels.copy()
                labels_copy[labels_copy != i] = 0
                top_closure = get_top_of_closure(labels_copy)
                near = seismic_nmf[0, ...][np.where(top_closure == 1)]
                cl["near_min"] = np.min(near)
                cl["near_max"] = np.max(near)
                cl["near_avg"] = np.mean(near)
                cl["near_std"] = np.std(near)
                cl["near_25pct"] = np.percentile(near, 25)
                cl["near_median"] = np.percentile(near, 50)
                cl["near_75pct"] = np.percentile(near, 75)
                mid = seismic_nmf[1, ...][np.where(top_closure == 1)]
                cl["mid_min"] = np.min(mid)
                cl["mid_max"] = np.max(mid)
                cl["mid_avg"] = np.mean(mid)
                cl["mid_std"] = np.std(mid)
                cl["mid_25pct"] = np.percentile(mid, 25)
                cl["mid_median"] = np.percentile(mid, 50)
                cl["mid_75pct"] = np.percentile(mid, 75)
                far = seismic_nmf[2, ...][np.where(top_closure == 1)]
                cl["far_min"] = np.min(far)
                cl["far_max"] = np.max(far)
                cl["far_avg"] = np.mean(far)
                cl["far_std"] = np.std(far)
                cl["far_25pct"] = np.percentile(far, 25)
                cl["far_median"] = np.percentile(far, 50)
                cl["far_75pct"] = np.percentile(far, 75)
                intercept = ai[np.where(top_closure == 1)]
                cl["intercept_min"] = np.min(intercept)
                cl["intercept_max"] = np.max(intercept)
                cl["intercept_avg"] = np.mean(intercept)
                cl["intercept_std"] = np.std(intercept)
                cl["intercept_25pct"] = np.percentile(intercept, 25)
                cl["intercept_median"] = np.percentile(intercept, 50)
                cl["intercept_75pct"] = np.percentile(intercept, 75)
                gradient = gi[np.where(top_closure == 1)]
                cl["gradient_min"] = np.min(gradient)
                cl["gradient_max"] = np.max(gradient)
                cl["gradient_avg"] = np.mean(gradient)
                cl["gradient_std"] = np.std(gradient)
                cl["gradient_25pct"] = np.percentile(gradient, 25)
                cl["gradient_median"] = np.percentile(gradient, 50)
                cl["gradient_75pct"] = np.percentile(gradient, 75)

            clist.append(cl)

        return clist

    def write_closure_info_to_log(self, seismic_nmf=None):
        """store info about closure in log file"""
        top_sand_layers = [x for x in self.top_lith_indices if self.facies[x] == 1.0]
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="top_sand_layers",
            val=top_sand_layers,
        )
        o = measure.label(self.oil_closures[:], connectivity=2, background=0)
        g = measure.label(self.gas_closures[:], connectivity=2, background=0)
        b = measure.label(self.brine_closures[:], connectivity=2, background=0)
        oil_closures = self.populate_closure_dict(o, "oil", seismic_nmf)
        gas_closures = self.populate_closure_dict(g, "gas", seismic_nmf)
        brine_closures = self.populate_closure_dict(b, "brine", seismic_nmf)
        all_closures = oil_closures + gas_closures + brine_closures
        for i, c in enumerate(all_closures):
            self.cfg.sqldict[f"closure_{i + 1}"] = c
        num_labels = np.max(o) + np.max(g)
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="number_hc_closures",
            val=num_labels,
        )
        # Add total number of closure voxels, with ratio of closure voxels given as a percentage
        closure_voxel_count = o[o > 0].size + g[g > 0].size
        closure_voxel_pct = closure_voxel_count / o.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count",
            val=closure_voxel_count,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct",
            val=closure_voxel_pct * 100,
        )
        # Same for Brine
        _brine_voxels = b[b == 1].size
        _brine_voxels_pct = _brine_voxels / b.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count_brine",
            val=_brine_voxels,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct_brine",
                                                                                        
                       val=_brine_voxels_pct * 100,
        )
        # Same for Oil
        _oil_voxels = o[o == 1].size
        _oil_voxels_pct = _oil_voxels / o.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count_oil",
            val=_oil_voxels,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct_oil",
            val=_oil_voxels_pct * 100,
        )
        # Same for Gas
        _gas_voxels = g[g == 1].size
        _gas_voxels_pct = _gas_voxels / g.size
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_count_gas",
            val=_gas_voxels,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="closure_voxel_pct_gas",
            val=_gas_voxels_pct,
        )
        # Write old logfile as well as the sql dict
        msg = f"layers for closure computation: {str(self.top_lith_indices)}\n"
        msg += f"Number of HC Closures : {num_labels}\n"
        msg += (
            f"Closure voxel count: {closure_voxel_count} - {closure_voxel_pct:5.2%}\n"
        )
        msg += (
            f"Closure voxel count: (brine) {_brine_voxels} - {_brine_voxels_pct:5.2%}\n"
        )
        msg += f"Closure voxel count: (oil) {_oil_voxels} - {_oil_voxels_pct:5.2%}\n"
        msg += f"Closure voxel count: (gas) {_gas_voxels} - {_gas_voxels_pct:5.2%}\n"
        print(msg)
        for i in range(self.facies.shape[0]):
            if self.facies[i] == 1:
                msg += f"  layers for closure computation:   {i}, sand\n"
            else:
                msg += f"  layers for closure computation:   {i}, shale\n"
        self.cfg.write_to_logfile(msg)

    def parse_label_values_and_counts(self, labels_clean):
        """parse label values and counts"""
        if self.cfg.verbose:
            print(" Inside parse_label_values_and_counts")
        next_label = 0
        label_values = [0]
        label_counts = [labels_clean[labels_clean == 0].size]
        for i in range(1, labels_clean.max() + 1):
            try:
                next_label = labels_clean[labels_clean > next_label].min()
            except (TypeError, ValueError):
                break
            label_values.append(next_label)
            label_counts.append(labels_clean[labels_clean == next_label].size)
            print(
                f"Label: {i}, label_values: {label_values[-1]}, label_counts: {label_counts[-1]}"
            )
        # force labels to use consecutive integer values
        for i, ilabel in enumerate(label_values):
            labels_clean[labels_clean == ilabel] = i
            label_values[i] = i
        # labels_clean = self.remove_small_objects(labels_clean)  # already applied to labels_clean
        # Remove label_value 0
        label_values.remove(0)
        return label_values, labels_clean

    def segment_closures(self, _closure_segments, remove_shale=True):
        """Segment the closures so that they can be randomly filled with hydrocarbons"""

        _closure_segments = np.clip(_closure_segments, 0.0, 1.0)
        # remove tiny clusters
        _closure_segments = minimum_filter(
            _closure_segments.astype("int16"), size=(3, 3, 1)
        )
        _closure_segments = maximum_filter(_closure_segments, size=(3, 3, 1))

        if remove_shale:
            # restrict closures to sand (non-shale) voxels
            if self.faults.faulted_lithology.shape[2] == _closure_segments.shape[2]:
                sand_shale = self.faults.faulted_lithology[:].copy()
            else:
                sand_shale = self.faults.faulted_lithology[
                    :, :, :: self.cfg.infill_factor
                ].copy()
            _closure_segments[sand_shale <= 0.0] = 0
            del sand_shale
        labels = measure.label(_closure_segments, connectivity=2, background=0)

        labels_clean = self.remove_small_objects(labels)
        return labels_clean, _closure_segments

    def find_faulted_closures(self, closure_segment_list, closure_segments):
        self._dilate_faults()
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            if faults_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    # Faulted oil closure
                    self.faulted_closures_oil[i, j, k] = 1.0
                    self.n_fault_closures_oil += 1
                    self.fault_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    # Faulted gas closure
                    self.faulted_closures_gas[i, j, k] = 1.0
                    self.n_fault_closures_gas += 1
                    self.fault_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    # Faulted brine closure
                    self.faulted_closures_brine[i, j, k] = 1.0
                    self.n_fault_closures_brine += 1
                    self.fault_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is faulted but does not have oil, gas or brine assigned"
                    )

    def find_onlap_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            onlaps_within_closure = self.onlaps_upward[i, j, k]
            if onlaps_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    self.onlap_closures_oil[i, j, k] = 1.0
                    self.n_onlap_closures_oil += 1
                    self.onlap_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    self.onlap_closures_gas[i, j, k] = 1.0
                    self.n_onlap_closures_gas += 1
                    self.onlap_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    self.onlap_closures_brine[i, j, k] = 1.0
                    self.n_onlap_closures_brine += 1
                    self.onlap_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is onlap but does not have oil, gas or brine assigned"
                    )

    def find_simple_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            onlaps = self._threshold_volumes(self.faults.faulted_onlap_segments[:])
            onlaps_within_closure = onlaps[i, j, k]
            oil_within_closure = self.oil_closures[i, j, k]
            gas_within_closure = self.gas_closures[i, j, k]
            brine_within_closure = self.brine_closures[i, j, k]
            if faults_within_closure.max() == 0 and onlaps_within_closure.max() == 0:
                if oil_within_closure.max() > 0:
                    self.simple_closures_oil[i, j, k] = 1.0
                    self.n_4way_closures_oil += 1
                elif gas_within_closure.max() > 0:
                    self.simple_closures_gas[i, j, k] = 1.0
                    self.n_4way_closures_gas += 1
                elif brine_within_closure.max() > 0:
                    self.simple_closures_brine[i, j, k] = 1.0
                    self.n_4way_closures_brine += 1
                else:
                    print(
                        "Closure is not faulted or onlap but does not have oil, gas or brine assigned"
                    )

    def find_false_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.fat_faults[i, j, k]
            onlaps_within_closure = self.onlaps_downward[i, j, k]
            for fluid, false, num in zip(
                [self.oil_closures, self.gas_closures, self.brine_closures],
                [
                    self.false_closures_oil,
                    self.false_closures_gas,
                    self.false_closures_brine,
                ],
                [
                    self.n_false_closures_oil,
                    self.n_false_closures_gas,
                    self.n_false_closures_brine,
                ],
            ):
                fluid_within_closure = fluid[i, j, k]
                if fluid_within_closure.max() > 0:
                    if onlaps_within_closure.max() > 0:
                        _faulted_closure_threshold = float(
                            faults_within_closure[faults_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        _onlap_closure_threshold = float(
                            onlaps_within_closure[onlaps_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        if (
                            _faulted_closure_threshold > 0.65
                            and _onlap_closure_threshold > 0.65
                        ):
                            false[i, j, k] = 1
                            num += 1

    def find_salt_bounded_closures(self, closure_segment_list, closure_segments):
        self._dilate_salt()
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            salt_within_closure = self.wide_salt[i, j, k]
            if salt_within_closure.max() > 0:
                if self.oil_closures[i, j, k].max() > 0:
                    # salt bounded oil closure
                    self.salt_closures_oil[i, j, k] = 1.0
                    self.n_salt_closures_oil += 1
                    self.salt_closures_oil_segment_list.append(iclosure)
                elif self.gas_closures[i, j, k].max() > 0:
                    # salt bounded gas closure
                    self.salt_closures_gas[i, j, k] = 1.0
                    self.n_salt_closures_gas += 1
                    self.salt_closures_gas_segment_list.append(iclosure)
                elif self.brine_closures[i, j, k].max() > 0:
                    # salt bounded brine closure
                    self.salt_closures_brine[i, j, k] = 1.0
                    self.n_salt_closures_brine += 1
                    self.salt_closures_brine_segment_list.append(iclosure)
                else:
                    print(
                        "Closure is salt bounded but does not have oil, gas or brine assigned"
                    )

    def find_faulted_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            if faults_within_closure.max() > 0:
                self.faulted_all_closures[i, j, k] = 1.0
                self.n_fault_all_closures += 1
                self.fault_all_closures_segment_list.append(iclosure)

    def find_onlap_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            onlaps_within_closure = self.onlaps_upward[i, j, k]
            if onlaps_within_closure.max() > 0:
                self.onlap_all_closures[i, j, k] = 1.0
                self.n_onlap_all_closures += 1
                self.onlap_all_closures_segment_list.append(iclosure)

    def find_simple_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.wide_faults[i, j, k]
            onlaps = self._threshold_volumes(self.faults.faulted_onlap_segments[:])
            onlaps_within_closure = onlaps[i, j, k]
            if faults_within_closure.max() == 0 and onlaps_within_closure.max() == 0:
                self.simple_all_closures[i, j, k] = 1.0
                self.n_4way_all_closures += 1

    def find_false_all_closures(self, closure_segment_list, closure_segments):
        for iclosure in closure_segment_list:
            i, j, k = np.where(closure_segments == iclosure)
            faults_within_closure = self.fat_faults[i, j, k]
            onlaps_within_closure = self.onlaps_downward[i, j, k]
            for fluid, false, num in zip(
                [self.oil_closures, self.gas_closures, self.brine_closures],
                [
                    self.false_closures_oil,
                    self.false_closures_gas,
                    self.false_closures_brine,
                ],
                [
                    self.n_false_closures_oil,
                    self.n_false_closures_gas,
                    self.n_false_closures_brine,
                ],
            ):
                fluid_within_closure = fluid[i, j, k]
                if fluid_within_closure.max() > 0:
                    if onlaps_within_closure.max() > 0:
                        _faulted_closure_threshold = float(
                            faults_within_closure[faults_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        _onlap_closure_threshold = float(
                            onlaps_within_closure[onlaps_within_closure > 0].size
                            / fluid_within_closure[fluid_within_closure > 0].size
                        )
                        if (
                            _faulted_closure_threshold > 0.65
                            and _onlap_closure_threshold > 0.65
                        ):
                            false[i, j, k] = 1
                            num += 1

    def _dilate_faults(self):
        thresholded_faults = self._threshold_volumes(self.faults.fault_planes[:])
        self.wide_faults[:] = self.grow_lateral(
            thresholded_faults, iterations=9, dist=1
        )
        self.fat_faults[:] = self.grow_lateral(
            thresholded_faults, iterations=21, dist=1
        )
        if self.cfg.include_salt:
            # Treat the salt body as a fault to grow closures to boundary
            thresholded_salt = self._threshold_volumes(
                self.faults.salt_model.salt_segments[:]
            )
            wide_salt = self.grow_lateral(thresholded_salt, iterations=9, dist=1)
            self.wide_salt[:] = wide_salt
            # Add salt to faults to cehck if growing the closure works
            self.wide_faults[:] += wide_salt

    @staticmethod
    def _threshold_volumes(volume, threshold=0.5):
        volume[volume >= threshold] = 1.0
        volume[volume < threshold] = 0.0
        return volume

    def grow_lateral(self, geobody, iterations, dist=1, verbose=False):
        from scipy.ndimage.morphology import grey_dilation

        dist_size = 2 * dist + 1
        mask = np.zeros((dist_size, dist_size, 1))
        mask[:, :, :] = 1
        _geobody = geobody.copy()
        if verbose:
            print(" ...grow_lateral: _geobody.shape = ", _geobody[_geobody > 0].shape)
        for k in range(iterations):
            try:
                _geobody = grey_dilation(_geobody, footprint=mask)
                if verbose:
                    print(
                        " ...grow_lateral: k, _geobody.shape = ",
                        k,
                        _geobody[_geobody > 0].shape,
                    )
            except:
                break
        return _geobody

    def write_closure_volumes_to_disk(self):
        """Write the closure volumes to disk as separate cubes"""
        if self.cfg.verbose:
            print("\n   ... writing closure volumes to disk")

        # Simple closures
        if self.simple_closures is not None:
            self.write_cube_to_disk(self.simple_closures[:], "simple_closures")
        if self.simple_closures_oil is not None:
            self.write_cube_to_disk(self.simple_closures_oil[:], "simple_closures_oil")
        if self.simple_closures_gas is not None:
            self.write_cube_to_disk(self.simple_closures_gas[:], "simple_closures_gas")
        if self.simple_closures_brine is not None:
            self.write_cube_to_disk(self.simple_closures_brine[:], "simple_closures_brine")
        # Stratigraphic (onlap) closures
        if self.strat_closures is not None:
            self.write_cube_to_disk(self.strat_closures[:], "strat_closures")
        if self.onlap_closures_oil is not None:
            self.write_cube_to_disk(self.onlap_closures_oil[:], "onlap_closures_oil")
        if self.onlap_closures_gas is not None:
            self.write_cube_to_disk(self.onlap_closures_gas[:], "onlap_closures_gas")
        if self.onlap_closures_brine is not None:
            self.write_cube_to_disk(self.onlap_closures_brine[:], "onlap_closures_brine")
        # Fault closures
        if self.fault_closures is not None:
            self.write_cube_to_disk(self.fault_closures[:], "fault_closures")
        if self.faulted_closures_oil is not None:
            self.write_cube_to_disk(self.faulted_closures_oil[:], "faulted_closures_oil")
        if self.faulted_closures_gas is not None:
            self.write_cube_to_disk(self.faulted_closures_gas[:], "faulted_closures_gas")
        if self.faulted_closures_brine is not None:
            self.write_cube_to_disk(self.faulted_closures_brine[:], "faulted_closures_brine")
        # All closures
        if self.all_closure_segments is not None:
            self.write_cube_to_disk(self.all_closure_segments[:], "all_closure_segments")

    def calculate_closure_statistics(self, fluid, closure_type):
        """Calculate and log statistics for closures"""
        if fluid[fluid > 0.0].size == 0:
            return
        closure_volumes = fluid[fluid > 0.0].size
        closure_mean = fluid[fluid > 0.0].mean()
        closure_std = fluid[fluid > 0.0].std()
        closure_min = fluid[fluid > 0.0].min()
        closure_max = fluid[fluid > 0.0].max()
        msg = f"{closure_type} closures: "
        msg += f"n: {closure_volumes}, "
        msg += f"mean: {closure_mean:.2f}, "
        msg += f"std: {closure_std:.2f}, "
        msg += f"min: {closure_min:.2f}, "
        msg += f"max: {closure_max:.2f}"
        print(msg)
        return msg

    def assign_fluid_types(self, label_values, labels_clean):
        """randomly assign oil or gas to closure"""
        print(
            " labels_clean.min(), labels_clean.max() = ",
            labels_clean.min(),
            labels_clean.max(),
        )
        _brine_closures = (labels_clean * 0.0).astype("uint8")
        _oil_closures = (labels_clean * 0.0).astype("uint8")
        _gas_closures = (labels_clean * 0.0).astype("uint8")

        fluid_type_code = np.random.randint(3, size=labels_clean.max() + 1)

        _closure_segments = self.closure_segments[:]
        for i in range(1, labels_clean.max() + 1):
            voxel_count = labels_clean[labels_clean == i].size
            if voxel_count > 0:
                print(f"Voxel Count: {voxel_count}\tFluid type: {fluid_type_code[i]}")
            # not in closure = 0
            # closure with brine filled reservoir fluid_type_code = 1 (same as background)
            # closure with oil filled reservoir fluid_type_code = 2
            # closure with gas filled reservoir fluid_type_code = 3
            if i in label_values:
                if fluid_type_code[i] == 0:
                    # brine: change labels_clean contents to fluid_type_code = 1 (same as background)
                    _brine_closures[
                        np.logical_and(labels_clean == i, _closure_segments > 0)
                    ] = 1
                elif fluid_type_code[i] == 1:
                    # oil: change labels_clean contents to fluid_type_code = 2
                    _oil_closures[labels_clean == i] = 1
                elif fluid_type_code[i] == 2:
                    # gas: change labels_clean contents to fluid_type_code = 3
                    _gas_closures[labels_clean == i] = 1
        return _oil_closures, _gas_closures, _brine_closures

    def remove_small_objects(self, labels, min_filter=True):
        try:
            # Use the global minimum voxel size initially, before closure types are identified
            labels_clean = morphology.remove_small_objects(
                labels, self.cfg.closure_min_voxels
            )
            if self.cfg.verbose:
                print("labels_clean succeeded.")
                print(
                    " labels.min:{}, labels.max: {}".format(labels.min(), labels.max())
                )
                print(
                    " labels_clean min:{}, labels_clean max: {}".format(
                        labels_clean.min(), labels_clean.max()
                    )
                )
        except Exception as e:
            print(
                f"Closures/create_closures: labels_clean (remove_small_objects) did not succeed: {e}"
            )
            if min_filter:
                labels_clean = minimum_filter(labels, size=(3, 3, 3))
                if self.cfg.verbose:
                    print(
                        " labels.min:{}, labels.max: {}".format(
                            labels.min(), labels.max()
                        )

                    )
                    print(
                        " labels_clean min:{}, labels_clean max: {}".format(
                            labels_clean.min(), labels_clean.max()
                        )

                    )
        return labels_clean


def _flood_fill(horizon, max_column_height=20.0, verbose=False, debug=False):
    """Locate areas on horizon that are in structural closure.

    # horizon: depth horizon as 2D numpy array.
    # - assume that fault intersections are inserted with value of 0.
    # - assume that values represent depth (i.e., bigger values are deeper)
    """
    from scipy import ndimage

    # copy input array
    temp_event = horizon.copy()

    emptypicks = temp_event * 0.0
    emptypicks[temp_event < 1.0] = 1.0
    emptypicks_dilated = ndimage.grey_dilation(
        emptypicks, size=(3, 3), structure=np.ones((3, 3))
    )
    # dilation removes some of the event - turn this off to honour the input events exactly
    # Changed to avoid vertical closure-boundaries near faults
    # emptypicks_dilated = emptypicks
    if verbose:
        print(
            " emptypicks_dilated min,mean,max = ",
            emptypicks_dilated.min(),
            emptypicks_dilated.mean(),
            emptypicks_dilated.max(),
        )

    # create boundary around edges of 2D array
    temp_event[:, :3] = 0.0
    temp_event[:, -3:] = 0.0
    temp_event[:3, :] = 0.0
    temp_event[-3:, :] = 0.0

    # replace pixels with value=0 with vertical 'wall' that is max_column_height deeper than nearby pixels
    temp_event[
        np.logical_and(emptypicks_dilated == 2.0, temp_event != 0.0)
    ] += max_column_height
    temp_event[emptypicks == 1.0] += 0.0

    # put deep point at map origin to 'collect' flood-fill run-off
    temp_event[0, 0] = 1.0e5

    # create flags to indicate pick vs no-pick in 2D array
    flags = np.zeros((horizon.shape[0], horizon.shape[1]), "int")
    flags[temp_event > 0.0] = 1
    flags[0, 0] = 1

    flood_filled = -fill_to_spill(-temp_event, flags)

    # set pixels near fault gaps to empty
    flood_filled[np.logical_and(emptypicks_dilated == 2.0, temp_event != 0.0)] = 0.0

    # limit closure heights to max_column_height
    # - Note that this typically causes under-filling of shallow 4-way closures
    if debug:
        import pdb

        pdb.set_trace()
    ff = flood_filled.copy()
    diff = horizon - ff
    diff[flood_filled == 0.0] = 0.0
    diff[diff != 0.0] = 1.0

    from skimage import morphology
    from skimage import measure

    labels = measure.label(diff, connectivity=2, background=0)
    labels_clean = morphology.remove_small_objects(labels, 50)
    labels_clean_list = list(set(labels_clean.flatten()))
    labels_clean_list.sort()
    for i in labels_clean_list:
        if i == 0:
            continue
        trap_crest = -horizon[labels_clean == i].min()
        initial_size = horizon[labels_clean == i].size
        spill_depth_map = np.ones_like(horizon) * (trap_crest - max_column_height)
        spill_points = np.dstack((-flood_filled, spill_depth_map))
        spill_point_map = spill_points.max(axis=-1)
        spill_point_map[spill_point_map == -100000.0] = 0.0
        spill_point_map[spill_point_map > 0.0] = 0.0
        flood_filled[labels_clean == i] = -spill_point_map[labels_clean == i]
        flood_filled[flood_filled < horizon] = horizon[flood_filled < horizon]
        final_size = horizon[
            np.where((horizon - flood_filled != 0.0) & (labels_clean == i))
        ].size
        print(
            "  ...inside _flood_fill: i, initial_size, final_size = ",
            i,
            initial_size,
            final_size,
        )
        del spill_depth_map
        del spill_points
        del spill_point_map
    del ff
    del diff
    del labels
    del labels_clean

    return flood_filled


def variable_max_column_height(top_lith_idx, n_layers, min_height, max_height):
    """
    Create a 1-D array of maximum column heights using linear function
    in layer numbers. Shallow closures will have small vertical closure
    heights. Deep closures will have larger vertical closure heights.

    Would be better to use a pressure profile to determine maximum
    column heights at given depths.

    Parameters
    ----------
    top_lith_idx : array-like
        1-D array of horizon numbers corresponding to top of layers
        where lithology changes
    n_layers : int
        Total number of layers in the model
    min_height : float
        Minimum column height for shallow layers
    max_height : float
        Maximum column height for deep layers

    Returns
    -------
    numpy.ndarray
        1-D array of maximum column heights, one for each horizon index
    """
    # Normalize horizon indices to [0, 1] range
    if len(top_lith_idx) == 0:
        return np.array([])

    # Linear interpolation: deeper horizons (higher indices) get higher heights
    normalized_indices = np.array(top_lith_idx) / max(1, n_layers - 1)
    heights = min_height + (max_height - min_height) * normalized_indices

    return heights


def fill_to_spill(test_array, array_flags, empty_value=1.0e22, quiet=True):
    if not quiet:
        print("   ... start fillToSpill ......")
    temp_array = test_array.copy()
    flags = array_flags.copy()
    test_array_max = 2.0 * (temp_array[~np.isnan(temp_array)]).max()
    temp_array[array_flags == 255] = -empty_value
    flood_filled = test_array_max - flood_fill_heap(
        test_array_max - temp_array, empty_value=empty_value
    )
    if not quiet:
        print("   ... finish fillToSpill ......")

    flood_filled[array_flags != 1] = 0
    flood_filled[flood_filled == 1.0e5] = 0
    flags[flood_filled == empty_value] = 0

    return flood_filled


def flood_fill_heap(test_array, empty_value=1.0e22, quiet=True):
    # from internet: http://arcgisandpython.blogspot.co.uk/2012/01/python-flood-fill-algorithm.html

    import heapq
    from scipy import ndimage

    input_array = np.copy(test_array)
    num_validPoints = (
        test_array.flatten().shape[0]
        - input_array[np.isnan(input_array)].shape[0]
        - input_array[input_array > empty_value / 2].shape[0]
    )
    if not quiet:
        print(
            "     ... flood_fill_heap ... number of valid input horizon picks = ",
            num_validPoints,
        )

    validPoints = input_array[~np.isnan(input_array)]
    validPoints = validPoints[validPoints < empty_value / 2]
    validPoints = validPoints[validPoints < 1.0e5]
    validPoints = validPoints[validPoints > np.percentile(validPoints, 2)]

    if len(validPoints) > 2:
        amin = validPoints.min()
        amax = validPoints.max()
    else:
        return test_array

    if not quiet:
        print(
            "    ... validPoints stats = ",
            validPoints.min(),
            np.median(validPoints),
            validPoints.mean(),
            validPoints.max(),
        )
        print(
            "    ... validPoints %tiles = ",
            np.percentile(validPoints, 0),
            np.percentile(validPoints, 1),
            np.percentile(validPoints, 5),
            np.percentile(validPoints, 10),
            np.percentile(validPoints, 25),
            np.percentile(validPoints, 50),
            np.percentile(validPoints, 75),
            np.percentile(validPoints, 90),
            np.percentile(validPoints, 95),
            np.percentile(validPoints, 99),
            np.percentile(validPoints, 100),
        )
        from datagenerator.util import import_matplotlib

        plt = import_matplotlib()
        plt.figure(5)
        plt.clf()
        plt.imshow(np.flipud(input_array), vmin=amin, vmax=amax, cmap="jet_r")
        plt.colorbar()
        plt.show()
        plt.savefig("flood_fill.png", format="png")
        plt.close()
        print("     ... min & max for surface = ", amin, amax)

    # set empty values and nan's to huge
    input_array[np.isnan(input_array)] = empty_value

    # Set h_max to a value larger than the array maximum to ensure that the while loop will terminate
    h_max = np.max(input_array * 2.0)

    # Build mask of cells with data not on the edge of the image
    # Use 3x3 square structuring element
    el = ndimage.generate_binary_structure(2, 2).astype(int)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    inside_mask[input_array == empty_value] = False
    edge_mask = np.invert(inside_mask)
    # Initialize output array as max value test_array except edges
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max

    if not quiet:
        plt.figure(6)
        plt.clf()
        plt.imshow(np.flipud(input_array), cmap="jet_r")
        plt.colorbar()
        plt.show()
        plt.savefig("flood_fill2.png", format="png")
        plt.close()

    # Build priority queue and place edge pixels into priority queue
    # Last value is flag to indicate if cell is an edge cell
    put = heapq.heappush
    get = heapq.heappop
    fill_heap = [
        (output_array[t_row, t_col], int(t_row), int(t_col), 1)
        for t_row, t_col in np.transpose(np.where(edge_mask))
    ]
    heapq.heapify(fill_heap)

    # Iterate until priority queue is empty
    while 1:
        try:
            h_crt, t_row, t_col, edge_flag = get(fill_heap)
        except IndexError:
            break
        for n_row, n_col in [
            ((t_row - 1), t_col),
            ((t_row + 1), t_col),
            (t_row, (t_col - 1)),
            (t_row, (t_col + 1)),
        ]:
            # Skip cell if outside array edges
            if edge_flag:
                try:
                    if not inside_mask[n_row, n_col]:
                        continue
                except IndexError:
                    continue
            if output_array[n_row, n_col] == h_max:
                output_array[n_row, n_col] = max(h_crt, input_array[n_row, n_col])
                put(fill_heap, (output_array[n_row, n_col], n_row, n_col, 0))
    output_array[output_array == empty_value] = np.nan
    return output_array
