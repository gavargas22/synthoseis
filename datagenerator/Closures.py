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
            val=np.max(avg_sand_th