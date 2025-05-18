import os
import math
import numpy as np
from scipy import stats
from scipy.interpolate import griddata
from datagenerator.Parameters import Parameters
from itertools import groupby
from datagenerator.util import import_matplotlib
import noise
from datagenerator.util import write_data_to_hdf
from scipy.ndimage import gaussian_filter, grey_closing
from .logging_config import setup_global_logging, get_logger


class Horizons:
    def __init__(self, parameters):
        self.cfg = parameters
        self.max_layers = 0

    def insert_feature_into_horizon_stack(self, feature, layer_number, maps):
        """
        Insert an object into horizons which is to be inserted into a single layer

        Made for fans, but should work for any object.
        Shallower layer thicknesses are adjusted to insert the feature (clipped downwards onto the feature).
        """
        layer_count = 1
        if feature.ndim == 3:
            layer_count = feature.shape[-1]
        new_maps = np.zeros(
            (
                self.cfg.cube_shape[0],
                self.cfg.cube_shape[1],
                maps.shape[2] + layer_count,
            )
        )
        # Shift horizons below layer downwards by 1 to insert object in specified layer
        new_maps[..., layer_number + layer_count :] = maps[..., layer_number:]
        # Insert object into layer_number
        new_maps[..., layer_number] = maps[..., layer_number] - feature
        # copy horizons above layer_number into the new set of maps
        new_maps[..., :layer_number] = maps[..., :layer_number]

        # Don't allow negative thickness after inserting the fan
        for i in range(new_maps.shape[-1] - 1, 1, -1):
            layer_thickness = new_maps[..., i] - new_maps[..., i - 1]
            if np.min(layer_thickness) < 0:
                np.clip(layer_thickness, 0, a_max=None, out=layer_thickness)
                new_maps[..., i - 1] = new_maps[..., i] - layer_thickness
        # Increase the max number of layers in model by 1
        self.max_layers += layer_count
        return new_maps

    def insert_seafloor(self, maps):
        if self.cfg.verbose:
            print("\n ... inserting 'water bottom' reflector in work cube ...\n")
        wb_time_map = maps[:, :, 1] - 1.5
        wb_stats = [
            f"{x * self.cfg.digi / self.cfg.infill_factor:.2f}"
            for x in [wb_time_map.min(), wb_time_map.mean(), wb_time_map.max()]
        ]
        self.cfg.write_to_logfile("Seabed Min: {}, Mean: {}, Max: {}".format(*wb_stats))

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="seabed_min",
            val=wb_time_map.min(),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="seabed_mean",
            val=wb_time_map.mean(),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="seabed_max",
            val=wb_time_map.max(),
        )

        maps[:, :, 0] = wb_time_map.copy()
        return maps

    def create_random_net_over_gross_map(
        self, avg=(0.45, 0.9), stdev=(0.01, 0.05), octave=9
    ):
        random_net_over_gross_map = self._perlin(base=None, octave=octave)

        avg_net_over_gross = np.random.uniform(*avg)
        avg_net_over_gross_stdev = np.random.uniform(*stdev)

        # set stdev and mean of map to desired values
        random_net_over_gross_map -= random_net_over_gross_map.mean()
        random_net_over_gross_map *= (
            avg_net_over_gross_stdev / random_net_over_gross_map.std()
        )
        random_net_over_gross_map += avg_net_over_gross
        # clip to stay within a reasonable N/G range
        random_net_over_gross_map = random_net_over_gross_map.clip(*avg)
        return random_net_over_gross_map

    def _perlin(self, base=None, octave=1, lac=1.9, do_rotate=True):

        xsize = self.cfg.cube_shape[0]
        ysize = self.cfg.cube_shape[1]
        if base is None:
            base = np.random.randint(255)
        # randomly rotate image
        if do_rotate:
            number_90_deg_rotations = 0
            fliplr = False
            flipud = False
            if xsize == ysize and np.random.binomial(1, 0.5) == 1:
                number_90_deg_rotations = int(np.random.uniform(1, 4))
                # temp = np.rot90(temp, number_90_deg_rotations)
            # randomly flip left and right, top and bottom
            if np.random.binomial(1, 0.5) == 1:
                fliplr = True
            if np.random.binomial(1, 0.5) == 1:
                flipud = True

        temp = np.array(
            [
                [
                    noise.pnoise2(
                        float(i) / xsize,
                        float(j) / ysize,
                        lacunarity=lac,
                        octaves=octave,
                        base=base,
                    )
                    for j in range(ysize)
                ]
                for i in range(xsize)
            ]
        )
        temp = np.rot90(temp, number_90_deg_rotations)
        if fliplr:
            temp = np.fliplr(temp)
        if flipud:
            temp = np.flipud(temp)
        return temp

    def _fit_plane_strike_dip(self, azimuth, dip, grid_shape, verbose=False):
        # Fits a plane given dip and max dip direction (azimuth)
        # - create a point at the center of the grid, elevation is zero
        xyz1 = np.array([grid_shape[0] / 2.0, grid_shape[1] / 2.0, 0.0])
        # y = np.array([0., 1., 1.])
        # - create a point in the strike direction at same elevation of zero
        strike_angle = azimuth + 90.0
        if strike_angle > 360.0:
            strike_angle -= 360.0
        if strike_angle > 180.0:
            strike_angle -= 180.0
        strike_angle *= np.pi / 180.0
        distance = np.min(grid_shape) / 4.0
        x = distance * math.cos(strike_angle) + grid_shape[0] / 2.0
        y = distance * math.sin(strike_angle) + grid_shape[1] / 2.0
        xyz2 = np.array([x, y, 0.0])
        # - create a point in the max dip direction
        dip_angle = dip * 1.0
        if dip_angle > 360.0:
            dip_angle -= 360.0
        if dip_angle > 180.0:
            dip_angle -= 180.0
        dip_angle *= np.pi / 180.0
        strike_angle = azimuth
        if strike_angle > 360.0:
            strike_angle -= 360.0
        if strike_angle > 180.0:
            strike_angle -= 180.0
            dip_elev = -distance * math.sin(dip_angle) * math.sqrt(2.0)
        else:
            dip_elev = distance * math.sin(dip_angle) * math.sqrt(2.0)
        strike_angle *= np.pi / 180.0
        x = distance * math.cos(strike_angle) + grid_shape[0] / 2.0
        y = distance * math.sin(strike_angle) + grid_shape[1] / 2.0
        xyz3 = np.array([x, y, dip_elev])
        # - combine 3 points into single array
        xyz = np.vstack((xyz1, xyz2, xyz3))
        # - fit three points to a plane and compute elevation for all grids on surface
        a, b, c = self.fit_plane_lsq(xyz)
        z = self.eval_plane(range(grid_shape[0]), range(grid_shape[1]), a, b, c)
        # - make plot if not quiet
        if verbose:

            plt = import_matplotlib()
            print(f"strike angle = {strike_angle}")
            print(f"dip angle = {dip_angle}")
            print(f"points are: {xyz}")
            plt.figure(1)
            plt.clf()
            plt.grid()
            plt.imshow(z, origin="origin")
            plt.plot(xyz[:, 0], xyz[:, 1], "yo")
            plt.colorbar()
            plt.savefig(os.path.join(self.cfg.work_subfolder, "dipping_plane.png"))
            plt.close()
        return z

    @staticmethod
    def fit_plane_lsq(xyz):
        # Fits a plane to a point cloud,
        # Where Z = aX + bY + c        ----Eqn #1
        # Rearranging Eqn1: aX + bY -Z +c =0
        # Gives normal (a,b,-1)
        # Normal = (a,b,-1)
        # [rows, cols] = xyz.shape
        rows = xyz.shape[0]
        g = np.ones((rows, 3))
        g[:, 0] = xyz[:, 0]  # X
        g[:, 1] = xyz[:, 1]  # Y
        z = xyz[:, 2]
        (a, b, c), _, _, _ = np.linalg.lstsq(g, z, rcond=-1)
        normal = np.array([a, b, c])
        return normal

    @staticmethod
    def eval_plane(x, y, a, b, c):
        # evaluates and returns Z for each input X,Y
        z = np.zeros((len(x), len(y)), "float")
        for i in range(len(x)):
            for j in range(len(y)):
                z[i, j] = a * x[i] + b * y[j] + c
        return z

    @staticmethod
    def halton(dim, nbpts):
        h = np.empty(nbpts * dim)
        h.fill(np.nan)
        p = np.empty(nbpts)
        p.fill(np.nan)
        p1 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        lognbpts = math.log(nbpts + 1)
        for i in range(dim):
            b = p1[i]
            n = int(math.ceil(lognbpts / math.log(b)))
            for t in range(n):
                p[t] = pow(b, -(t + 1))

            for j in range(nbpts):
                d = j + 1
                sum_ = math.fmod(d, b) * p[0]
                for t in range(1, n):
                    d = math.floor(d / b)
                    sum_ += math.fmod(d, b) * p[t]

                h[j * dim + i] = sum_

        return h.reshape(nbpts, dim)

    @staticmethod
    def rotate_point(x, y, angle_in_degrees):
        """Calculate new coordinates for point after coordinate rotation about (0,0) by angleDegrees"""
        angle = angle_in_degrees * np.pi / 180.0
        x1 = math.cos(angle) * x + math.sin(angle) * y
        y1 = -math.sin(angle) * x + math.cos(angle) * y
        return x1, y1

    def convert_map_from_samples_to_units(self, maps):
        """Use digi to convert a copy of the provided maps from samples to units"""
        converted_maps = maps.copy()
        converted_maps *= float(self.cfg.digi)
        return converted_maps

    def write_maps_to_disk(self, horizons, name):
        """Write horizons to disk."""
        fname = os.path.join(self.cfg.work_subfolder, name)
        np.save(fname, horizons)

    def write_onlap_episodes(
        self, onlap_horizon_list, depth_maps_gaps, depth_maps_infilled, n=35
    ):
        """Write gapped horizons with onlaps + n shallower horizons to a separate files."""
        tilting_zmaps = list()
        interval_thickness_size = depth_maps_gaps.shape[0] * depth_maps_gaps.shape[1]
        for ihor in onlap_horizon_list:
            for jhor in range(ihor - n, ihor + 1):
                if jhor > 0:
                    # compute thickness compared to next horizon using infilled depth maps
                    interval_thickness = (
                        depth_maps_infilled[:, :, jhor + 1]
                        - depth_maps_infilled[:, :, jhor]
                    )
                    # compute percentage of interval filled with zeros
                    interval_thickness_zeros_size = interval_thickness[
                        interval_thickness < 1.0e-5
                    ].shape[0]
                    pct_zeros = (
                        float(interval_thickness_zeros_size) / interval_thickness_size
                    )
                    if pct_zeros > 0.02:
                        tilting_zmaps.append(jhor)
        # Make the tilting maps unique and sorted
        tilting_zmaps = list(set(tilting_zmaps))
        tilting_zmaps.sort()
        onlaps = np.zeros(
            (depth_maps_gaps.shape[0], depth_maps_gaps.shape[1], len(tilting_zmaps))
        )
        for count, h in enumerate(tilting_zmaps):
            onlaps[..., count] = depth_maps_gaps[..., h]

        # Write to disk
        self.write_maps_to_disk(onlaps * self.cfg.digi, "depth_maps_onlaps")

        if self.cfg.hdf_store:
            # Write onlap maps to hdf
            write_data_to_hdf(
                "depth_maps_onlaps", onlaps * self.cfg.digi, self.cfg.hdf_master
            )

    def write_fan_horizons(self, fan_horizon_list, depth_maps):
        """Write fan layers to a separate file."""
        z = depth_maps.copy()
        fan_horizons = np.zeros((z.shape[0], z.shape[1], len(fan_horizon_list)))
        for count, horizon in enumerate(fan_horizon_list):
            thickness_map = z[..., horizon + 1] - z[..., horizon]
            z[..., horizon][thickness_map <= 0] = np.nan
            if self.cfg.verbose:
                print(
                    f"Fan horizon number: {horizon} Max thickness: {np.nanmax(thickness_map)}"
                )
            fan_horizons[..., count] = z[..., horizon]
        self.write_maps_to_disk(fan_horizons, "depth_maps_fans")


class RandomHorizonStack(Horizons):
    def __init__(self, parameters):
        self.cfg: Parameters = parameters
        # Initialize global logging based on verbose flag
        setup_global_logging(verbose=self.cfg.verbose)
        # Get logger for this module
        self.logger = get_logger(__name__)
        self.logger.debug(
            "Initializing RandomHorizonStack with parameters: %s", self.cfg
        )
        self.depth_maps = None
        self.depth_maps_gaps = None
        self.max_layers = 0
        # Look up tables
        self.thicknesses = None
        self.onlaps = None
        self.channels = None
        self.dips = None
        self.azimuths = None
        self.facies = None
        # Start the process of creating depth maps
        self.create_depth_maps()

    def _generate_lookup_tables(self):
        self.logger.debug("Generating lookup tables for horizons")
        # Thicknesses
        self.thicknesses = self._get_thicknesses()
        self.logger.debug(
            "Generated thicknesses with shape: %s", self.thicknesses.shape
        )
        # Onlaps
        onlap_layer_list = self._get_onlap_layer_list()
        self.logger.debug("Generated onlap layer list: %s", onlap_layer_list)
        # Dips
        self.dips = self._get_dips()
        self.logger.debug(
            "Generated dips with mean: %.2f, max: %.2f",
            np.mean(self.dips),
            np.max(self.dips),
        )
        # Azimuths
        self.azimuths = self._get_azimuths()
        self.logger.debug("Generated azimuths with mean: %.2f", np.mean(self.azimuths))
        # Channels
        self.channels = np.random.binomial(1, 3.0 / 100.0, self.cfg.num_lyr_lut)
        self.logger.debug(
            "Generated channels with %d active channels", np.sum(self.channels)
        )

        onlap_array_dim = int(500 / 1250 * self.cfg.cube_shape[2])
        self.onlaps = np.zeros(onlap_array_dim, "int")
        self.onlaps[onlap_layer_list] = 1
        self.logger.debug(
            "Created onlap flags array with dimension: %d", onlap_array_dim
        )

        if not self.cfg.include_channels:
            self.channels *= 0
            self.logger.debug("Channels disabled in configuration")
        else:
            # Make sure channel episodes don't overlap too much
            for i in range(len(self.channels)):
                if self.channels[i] == 1:
                    self.channels[i + 1 : i + 6] *= 0
            self.logger.debug("Channel overlap prevention applied")

    def _get_thicknesses(self):
        """Generate random layer thicknesses using a Gamma distribution.

        This method generates realistic geological layer thicknesses for a synthetic seismic model.
        It uses a Gamma distribution with shape parameter 4.0 and scale parameter 2.0 to create
        thicknesses that mimic natural sedimentary layers.

        The Gamma distribution is chosen because:
        1. It only generates positive values (layer thicknesses cannot be negative)
        2. It creates a right-skewed distribution (many thin layers, fewer thick layers)
        3. The parameters (4.0, 2.0) are tuned to create geologically realistic thicknesses

        Parameters
        ----------
        None
            Uses self.cfg.num_lyr_lut from the class configuration to determine how many
            thicknesses to generate

        Returns
        -------
        numpy.ndarray
            An array of layer thicknesses with length equal to num_lyr_lut.
            Each value represents the thickness of a geological layer in the model.
            Values are always positive and follow a Gamma distribution.

        Notes
        -----
        The Gamma distribution parameters (4.0, 2.0) were chosen to create a distribution
        that matches typical sedimentary layer thicknesses, where:
        - Most layers are relatively thin
        - Some layers are thicker
        - Very thick layers are rare
        - No layers can have zero or negative thickness
        """
        return stats.gamma.rvs(4.0, 2, size=self.cfg.num_lyr_lut)

    def _get_azimuths(self):
        """Generate random azimuth angles (in degrees) for each geological horizon in a synthetic seismic model.

        This method is a crucial component of the synthetic seismic model generation process,
        specifically responsible for creating realistic azimuth angles (compass bearings) for
        each geological horizon in the model. These azimuth angles determine the direction
        of maximum dip (steepest slope) for each layer, which is essential for creating
        geologically realistic structures that mimic real-world sedimentary basins.

        In geological terms, azimuth represents the compass direction (0-360 degrees) in which
        a layer dips most steeply. For example:
        - An azimuth of 0° means the layer dips most steeply to the North
        - An azimuth of 90° means the layer dips most steeply to the East
        - An azimuth of 180° means the layer dips most steeply to the South
        - An azimuth of 270° means the layer dips most steeply to the West

        The method works by:
        1. Generating random values uniformly distributed between 0 and 360 degrees
        2. Creating one value for each layer in the model (determined by self.cfg.num_lyr_lut)
        3. Returning these values as a numpy array

        The resulting array of azimuth angles is used throughout the model generation process to:
        - Create realistic geological structures by orienting horizons appropriately
        - Generate synthetic seismic data that accurately represents the geological structure
        - Model potential hydrocarbon traps and migration pathways
        - Create realistic structural features like anticlines and synclines

        Parameters
        ----------
        None
            Uses self.cfg.num_lyr_lut from the class configuration to determine how many
            azimuth angles to generate

        Returns
        -------
        numpy.ndarray
            An array of azimuth angles in degrees, with length equal to num_lyr_lut.
            Each value represents the azimuth (compass bearing) of maximum dip for a
            specific horizon in the model. Values range from 0 to 360 degrees.

        Notes
        -----
        The uniform distribution was chosen because in real sedimentary basins, the
        direction of maximum dip can occur in any direction with equal probability.
        This is different from the dip angles themselves (generated separately), which
        tend to be skewed towards small angles.

        The azimuth angles generated by this method are used in conjunction with dip
        angles (generated separately) to create the full 3D orientation of each horizon.

        Example
        -------
        If num_lyr_lut = 5, the resulting azimuths might look like:
        [45.2, 178.9, 267.3, 92.1, 315.7]
        where each value represents the compass bearing of maximum dip for a layer
        """
        azimuths = np.random.uniform(low=0.0, high=360.0, size=(self.cfg.num_lyr_lut,))

        return azimuths

    def _get_dips(self) -> np.ndarray:
        """Generate realistic dip angles for geological horizons in a synthetic seismic model.

        This method is a crucial component of the synthetic seismic model generation process,
        specifically responsible for creating realistic dip angles (angles of inclination) for
        each geological horizon in the model. These dip angles determine how sedimentary layers
        are tilted relative to horizontal, which is essential for creating geologically realistic
        structures that mimic real-world sedimentary basins.

        In geological terms, dip angles represent the maximum angle of inclination of a layer
        relative to horizontal. In real sedimentary basins, most layers have relatively shallow
        dips (close to horizontal), while occasional steeper dips occur in areas of tectonic
        activity or structural deformation. This method replicates this natural distribution
        using a power law distribution.

        The method works by:
        1. Generating random values using a power law distribution (np.random.power) with
           exponent 100. This creates a distribution where:
           - ~90% of values are very close to 0 (representing near-horizontal layers)
           - ~9% of values are moderate (representing gently dipping layers)
           - ~1% of values are higher (representing steeply dipping layers)
        2. Inverting the distribution (1.0 - power_law_values) to get the desired skew
        3. Scaling the values by 7.0 to create a reasonable range for sedimentary basin dips
        4. Applying dip_factor_max from the configuration to allow model customization

        The resulting array of dip angles is used throughout the model generation process to:
        - Create realistic geological structures by tilting horizons appropriately
        - Generate synthetic seismic data that accurately represents the geological structure
        - Model potential hydrocarbon traps and migration pathways
        - Create realistic structural features like anticlines and synclines

        Parameters
        ----------
        None
            Uses self.cfg.num_lyr_lut and self.cfg.dip_factor_max from the class configuration

        Returns
        -------
        numpy.ndarray
            An array of dip angles in degrees, with length equal to num_lyr_lut.
            Each value represents the dip angle for a specific horizon in the model.
            The distribution is heavily skewed towards small angles (< 1 degree) with
            occasional steeper dips up to 7.0 * dip_factor_max degrees.

        Notes
        -----
        The power law distribution (exponent 100) was chosen because it creates a very
        steep distribution that closely matches real-world observations of dip angles
        in sedimentary basins. The multiplication by 7.0 creates a reasonable range
        for sedimentary basin dips, while dip_factor_max allows for model customization
        to simulate different geological settings (e.g., passive margins vs. active
        tectonic regions).

        The dip angles generated by this method are used in conjunction with azimuths
        (generated separately) to create the full 3D orientation of each horizon.

        Example
        -------
        If dip_factor_max = 1.0, the resulting dips might look like:
        [0.1, 0.3, 0.05, 1.2, 0.2, 0.4, 0.15, 2.1, ...]
        where most values are small (< 1 degree) with occasional steeper dips
        """
        dips = (
            (1.0 - np.random.power(100, self.cfg.num_lyr_lut))
            * 7.0
            * self.cfg.dip_factor_max
        )

        return dips

    def _get_onlap_layer_list(self, low: int = 5, high: int = 200):
        """Generate a list of layer numbers where onlap (tilting) episodes will occur in the geological model.

        This method is crucial for simulating realistic geological sequences where layers of sediment
        progressively onlap (overlap) onto older layers due to changes in depositional conditions or
        tectonic tilting. Onlap episodes are important geological features that indicate changes in
        relative sea level, basin subsidence, or sediment supply.

        The method works by:
        1. Randomly determining how many onlap episodes to create using a triangular distribution
           centered around 4 episodes (range 1-7)
        2. Randomly selecting layer numbers between 5 and 200 where these onlap episodes will occur
        3. Sorting these layer numbers to ensure they occur in chronological order

        The specific parameters used:
        - Minimum layer (low=5): Ensures onlaps don't occur too close to the base of the model
        - Maximum layer (high=200): Prevents onlaps from occurring too close to the top
        - Number of episodes: Uses triangular distribution with:
          * Minimum: 1 episode
          * Mode: 4 episodes
          * Maximum: 7 episodes

        Returns
        -------
        numpy.ndarray
            A sorted array of integers representing the layer numbers where onlap episodes
            will be simulated. Each number corresponds to a specific horizon in the model
            where a tilting event will occur, causing subsequent layers to onlap onto it.

        Notes
        -----
        - The method is called during model initialization to set up the basic structure
          of the geological model
        - These onlap layers are later used by the Onlaps class to modify the depth maps
          and create realistic geological sequences
        - The actual onlap simulation involves tilting the layers above each onlap surface
          to create the characteristic overlapping pattern seen in real geological sequences

        Example
        -------
        >>> horizons = RandomHorizonStack(parameters)
        >>> onlap_layers = horizons._get_onlap_layer_list()
        >>> print(onlap_layers)
        array([ 23,  45,  78, 156])  # Example output showing 4 onlap episodes
        """
        onlap_layer_list = np.sort(
            np.random.uniform(
                low=low, high=high, size=int(np.random.triangular(1, 4, 7) + 0.5)
            ).astype("int")
        )

        return onlap_layer_list

    def _random_layer_thickness(self):
        rand_oct = int(np.random.triangular(left=1.3, mode=2.65, right=5.25))

        low_thickness_factor = np.random.triangular(left=0.05, mode=0.2, right=0.95)

        high_thickness_factor = np.random.triangular(left=1.05, mode=1.8, right=2.2)

        thickness_factor_map = self._perlin(octave=rand_oct)
        thickness_factor_map -= thickness_factor_map.mean()
        thickness_factor_map *= (high_thickness_factor - low_thickness_factor) / (
            thickness_factor_map.max() - thickness_factor_map.min()
        )
        thickness_factor_map += 1.0
        if thickness_factor_map.min() < 0.0:
            thickness_factor_map -= thickness_factor_map.min() - 0.05

        if self.cfg.verbose:
            print(
                f" {thickness_factor_map.mean()}, "
                f"{thickness_factor_map.min()}, "
                f" {thickness_factor_map.max()}, "
                f" {thickness_factor_map.std()}"
            )

        return thickness_factor_map

    def _generate_random_depth_structure_map(
        self,
        dip_range=(0.25, 1.0),
        num_points_range=(2, 25),
        elevation_std=100.0,
        zero_at_corners=True,
        initial=False,
    ):
        ############################################################################
        # generate a 2D array representing a depth (or TWT) structure map.
        # - grid_size controls output size in (x,y), units are samples
        # - dip_range controls slope of dipping plane upon which random residual points are placed
        # - num_points controls number of randomly positioned points are used for residual grid
        # - elevation_std controls std dev for residual elevation of random points in residual grid
        ############################################################################
        grid_size = self.cfg.cube_shape[:2]
        # define output grid (padded by 15% in each direction)
        xi = np.linspace(-0.15, 1.15, int(grid_size[0] * 1.3))
        yi = np.linspace(-0.15, 1.15, int(grid_size[1] * 1.3))

        # start with a gently dipping plane similar to that found on passive shelf margins (< 1 degree dip)
        azimuth = np.random.uniform(0.0, 360.0)
        dip = np.random.uniform(dip_range[0], dip_range[1])

        # build a residual surface to add to the dipping plane
        number_halton_points = int(np.random.uniform(100, 500) + 0.5)
        number_random_points = int(
            np.random.uniform(num_points_range[0], num_points_range[1]) + 0.5
        )
        z = np.random.rand(number_random_points)
        z -= z.mean()
        if initial:
            elevation_std = self.cfg.initial_layer_stdev
        z *= elevation_std / z.std()

        dipping_plane = self._fit_plane_strike_dip(
            azimuth, dip, grid_size, verbose=False
        )
        xx = self.halton(2, number_halton_points)[-number_random_points:]

        # make up some randomly distributed data
        # - adjust for padding
        xx *= xi[-1] - xi[0]
        xx += xi[0]
        x = xx[:, 0]
        y = xx[:, 1]

        if zero_at_corners:
            x = np.hstack((x, [-0.15, -0.15, 1.15, 1.15]))
            y = np.hstack((y, [-0.15, 1.15, -0.15, 1.15]))
            z = np.hstack((z, [-0.15, -0.15, -0.15, -0.15]))

        # grid the data.
        zi = griddata(
            np.column_stack((x, y)),
            z,
            (xi[:, np.newaxis], yi[np.newaxis, :]),
            method="cubic",
        )
        zi = zi.reshape(xi.shape[0], yi.shape[0])
        xi_min_index = np.argmin((xi - 0.0) ** 2)
        xi_max_index = xi_min_index + grid_size[0]
        yi_min_index = np.argmin((yi - 0.0) ** 2)
        yi_max_index = yi_min_index + grid_size[1]
        zi = zi[xi_min_index:xi_max_index, yi_min_index:yi_max_index]
        # add to the gently sloping plane
        zi += dipping_plane

        if initial:
            zi += dipping_plane - dipping_plane.min()
            zi += self.cfg.cube_shape[-1] * self.cfg.infill_factor - zi.min()
            zi_argmin_i, zi_argmin_j = np.unravel_index(zi.argmin(), zi.shape)
            if self.cfg.verbose:
                print(
                    f"\tIndices for shallowest point in cube: {zi_argmin_i}, {zi_argmin_j}"
                )
            self.cfg.write_to_logfile(
                f"number_random_points: {number_random_points:.0f}"
            )
            self.cfg.write_to_logfile(f"dip angle: {dip:.2f}")
            self.cfg.write_to_logfile(
                f"dipping_plane min: {dipping_plane.min():.2f}, mean: {dipping_plane.mean():.2f},"
                f" max: {dipping_plane.max():.2f}"
            )
            self.cfg.write_to_logfile(
                f"zi min: {zi.min():.2f}, mean: {zi.mean():.2f}, max: {zi.max():.2f}"
            )

            self.cfg.write_to_logfile(
                msg=None,
                mainkey="model_parameters",
                subkey="number_random_points",
                val=number_random_points,
            )
            self.cfg.write_to_logfile(
                msg=None,
                mainkey="model_parameters",
                subkey="dip_angle",
                val=number_random_points,
            )
            self.cfg.write_to_logfile(
                msg=None,
                mainkey="model_parameters",
                subkey="dipping_plane_min",
                val=dipping_plane.min(),
            )
            self.cfg.write_to_logfile(
                msg=None,
                mainkey="model_parameters",
                subkey="dipping_plane_mean",
                val=dipping_plane.mean(),
            )
            self.cfg.write_to_logfile(
                msg=None,
                mainkey="model_parameters",
                subkey="dipping_plane_max",
                val=dipping_plane.max(),
            )
            self.cfg.write_to_logfile(
                msg=None, mainkey="model_parameters", subkey="zi_min", val=zi.min()
            )
            self.cfg.write_to_logfile(
                msg=None, mainkey="model_parameters", subkey="zi_mean", val=zi.mean()
            )
            self.cfg.write_to_logfile(
                msg=None, mainkey="model_parameters", subkey="zi_max", val=zi.max()
            )

        return xx, zi

    def _create_thickness_map(self, random_thickness_factor_map, layer_number):
        """
        Create a random_thickness_map for a given layer.

        Creates a dipping_plane with differential dip from previous layer and computes a new thickness_map which is
         always positive (i.e. no erosion allowed)

        Parameters
        ----------
        random_thickness_factor_map : ndarray - a random thickness factor map for this layer
        layer_number : int - the layer number being built in the stack of horizons

        Returns
        -------
        thickness_map : ndarray - the thickness_map for the given layer
        """
        dipping_plane = self._fit_plane_strike_dip(
            azimuth=self.azimuths[layer_number],
            dip=self.dips[layer_number],
            grid_shape=self.cfg.cube_shape[:2],
            verbose=False,
        )
        dipping_plane -= dipping_plane.min()
        if self.cfg.verbose:
            print(
                f"azi, dip, dipping_plane min/mean/max = {self.azimuths[layer_number]}, {self.dips[layer_number]}, "
                f"{dipping_plane.min():.2f}, {dipping_plane.mean():.2f}, {dipping_plane.max():.2f}"
            )

        # compute new thickness map. For now, don't allow erosion (i.e. thickness_map is always positive)
        thickness_map = (
            (
                np.ones(self.cfg.cube_shape[:2], "float")
                * self.thicknesses[layer_number]
                - dipping_plane
            )
            * self.cfg.infill_factor
            * random_thickness_factor_map
        )
        thickness_map = np.clip(
            thickness_map, 0.0, self.cfg.thickness_max * self.cfg.infill_factor * 1.5
        )
        return thickness_map

    def _create_initial_thickness_factor_map(self):
        """
        Create the thickness_factor map for the layer at the base of the model

        The thickness factor map is a 2D array that controls the relative
        thickness variations across a layer in the geological model.

        It's essentially a multiplier that determines how thick or thin different
        parts of a layer will be. A value of 1.0 means normal thickness,
        values > 1.0 mean thicker than normal, and values < 1.0 mean thinner than normal

        Returns
        -------
        random_thickness_factor_map : ndarray
            A random thickness factor map for the initial layer at base
        """
        _, random_thickness_factor_map = self._generate_random_depth_structure_map(
            dip_range=[0.0, 0.0], num_points_range=(25, 100), elevation_std=0.45
        )
        random_thickness_factor_map -= random_thickness_factor_map.mean() - 1
        #
        return random_thickness_factor_map

    def create_depth_maps(self):
        """Building layers in reverse order - starting at bottom and depositing new layers on top.

        Each layer has random residual dip and pseudo-random residual thickness.
        Layers are written directly to zarr storage to minimize memory usage.
        """
        self.logger.debug("Starting creation of depth maps")
        self._generate_lookup_tables()
        # Create initial depth map at base of model using initial=True
        _, previous_depth_map = self._generate_random_depth_structure_map(
            dip_range=[0.0, 75],
            num_points_range=(3, 5),
            initial=True,
            elevation_std=self.cfg.initial_layer_stdev,
        )
        self.logger.debug(
            "Created initial depth map with shape: %s", previous_depth_map.shape
        )

        # Build layers while minimum depth is less than the seabed_min_depth as set in config (given in metres).
        # Convert min_depth into samples, don't forget about the infill factor
        shallowest_depth_to_build = (
            self.cfg.seabed_min_depth / float(self.cfg.digi)
        ) * self.cfg.infill_factor

        # Initialize zarr array with a single layer - we'll resize as needed
        self.depth_maps = self.cfg.zarr_init(
            "depth_maps", shape=(*previous_depth_map.shape, 1)
        )
        self.depth_maps[:, :, 0] = previous_depth_map
        self.max_layers = 1

        # Build layers in a loop from deep to shallow until the minimum depth is reached and then break out.
        for i in range(20000):
            # Do the special case for the random_thickness_factor_map for the initial layer (at base)
            if i == 0:
                if self.cfg.verbose:
                    self.logger.info("Building random depth map at base of model")
                random_thickness_factor_map = (
                    self._create_initial_thickness_factor_map()
                )
            else:  # Otherwise create standard random thickness factor map
                if self.cfg.verbose:
                    self.logger.info("Building Layer %d", i)
                random_thickness_factor_map = self._random_layer_thickness()

            # Create the layer's thickness map using the random_thickness_factor_map
            thickness_map = self._create_thickness_map(random_thickness_factor_map, i)
            current_depth_map = previous_depth_map - thickness_map

            if self.cfg.verbose:
                self.logger.debug(
                    "current_depth_map min/mean/max = %.2f, %.2f, %.2f",
                    current_depth_map.min(),
                    current_depth_map.mean(),
                    current_depth_map.max(),
                )
                self.logger.debug(
                    "thickness_map min/mean/max = %.2f, %.2f, %.2f",
                    thickness_map.min(),
                    thickness_map.mean(),
                    thickness_map.max(),
                )

            # break out of loop when minimum depth is reached
            if current_depth_map.min() <= shallowest_depth_to_build:
                break

            # Resize zarr array to accommodate new layer
            self.depth_maps.resize(
                (*self.depth_maps.shape[:-1], self.depth_maps.shape[-1] + 1)
            )
            # Write new layer to zarr array
            self.depth_maps[:, :, self.max_layers] = current_depth_map
            self.max_layers += 1

            # Update previous depth map for next iteration
            previous_depth_map = current_depth_map.copy()

            if self.cfg.verbose:
                self.logger.debug(
                    "Layer %d, depth_maps.shape = %s", i, self.depth_maps.shape
                )

            # Clear temporary arrays to free memory
            del thickness_map
            del current_depth_map

        if self.cfg.verbose:
            self.logger.info("Finished creating horizon layers")
            self.logger.info("Final depth maps shape: %s", self.depth_maps.shape)


class Onlaps(Horizons):
    def __init__(self, parameters, depth_maps, thicknesses, max_layers):
        self.cfg = parameters
        self.depth_maps = depth_maps
        self.thicknesses = thicknesses
        self.max_layers = max_layers
        self.onlap_horizon_list = list()
        self.logger = get_logger(__name__)
        self._generate_onlap_lookup_table()

    def _calculate_onlap_array_dimension(self) -> int:
        """Calculate the dimension of the onlap array based on the cube shape and onlap ratio.

        The onlap array dimension determines how much of the total depth (cube_shape[2])
        should be allocated for onlap episodes. This is calculated using the onlap_ratio
        from the configuration.

        Returns
        -------
        int
            The dimension of the onlap array, representing the number of depth samples
            that can potentially contain onlap episodes.
        """
        return int(self.cfg.onlap_ratio * self.cfg.cube_shape[2])

    def _create_onlap_flags_array(self, onlap_layer_list: np.ndarray) -> np.ndarray:
        """Create a binary array indicating where onlap episodes occur.

        Parameters
        ----------
        onlap_layer_list : np.ndarray
            Array of layer indices where onlap episodes should occur

        Returns
        -------
        np.ndarray
            Binary array where 1 indicates an onlap episode and 0 indicates no onlap
        """
        array_dim = self._calculate_onlap_array_dimension()
        onlap_flags = np.zeros(array_dim, dtype="int")
        onlap_flags[onlap_layer_list] = 1
        return onlap_flags

    def _generate_onlap_lookup_table(self):
        """Generate a lookup table for onlap episodes in the geological model.

        An onlap is a geological feature where younger sedimentary layers overlap older ones,
        typically occurring during marine transgression (rising sea levels). This method
        creates a plan for where these onlap episodes will occur in the synthetic model.

        Geological Context:
        - Onlaps represent periods of rising sea level where new sediment is deposited
          over existing layers
        - They create important stratigraphic features that can trap hydrocarbons
        - The distribution of onlaps affects the overall geometry of the reservoir

        Technical Implementation:
        1. Randomly selects between 1-7 locations in the model where onlaps will occur
        2. Ensures onlaps don't occur in the first 5 layers (to avoid very shallow features)
        3. Creates a binary array marking these locations
        4. Logs configuration and statistics for model validation

        The method uses a triangular distribution to determine the number of onlap episodes,
        favoring 4 episodes (mode) but allowing for natural variation between 1-7 episodes.
        This creates realistic geological models while maintaining computational efficiency.

        Returns
        -------
        None
            The method modifies the instance variables:
            - self.onlap_layer_list: Array of layer numbers where onlaps occur
            - self.onlaps: Binary array marking onlap locations
            - Logs configuration and statistics at DEBUG level

        Notes
        -----
        The onlap ratio (from configuration) determines what portion of the model
        can contain onlaps, ensuring geologically plausible distributions.
        """
        # Generate list of layers where onlaps will occur
        onlap_layer_list = np.sort(
            np.random.uniform(
                low=5,
                high=self.max_layers,
                size=int(np.random.triangular(1, 4, 7) + 0.5),
            ).astype("int")
        )

        self.logger.debug(
            "Onlap configuration: num_lyr_lut=%d, onlap_layer_list=%s",
            self.cfg.num_lyr_lut,
            onlap_layer_list,
        )

        # Create binary array marking onlap layers
        self.onlaps = self._create_onlap_flags_array(onlap_layer_list)

        self.logger.debug(
            "Onlap statistics: flags=%d, ratio=%.2f, array_dim=%d",
            self.onlaps[self.onlaps == 1].shape[0],
            self.cfg.onlap_ratio,
            len(self.onlaps),
        )

    def insert_tilting_episodes(self):
        ############################################################################
        # insert tilting (onlap) episodes
        # - computations loop from deep horizons toward shallow (top of horizon cube)
        ############################################################################
        if self.cfg.verbose:
            print("\n ... simulating tilting (onlap) episodes ...")

        azi_list = []
        dip_list = []

        count = 0
        onlaps_horizon_list = []
        layer_has_onlaps = np.zeros((self.depth_maps.shape[2]), "bool")
        for i in range(self.depth_maps.shape[2] - 2, 0, -1):
            if self.onlaps[i] == 1:
                # A tilting (onlap) episode occurs for this horizon.
                # Add a random dipping layer to this and all shallower layers, and adjust depth for layer thickness
                count += 1
                if self.cfg.verbose:
                    print(
                        " ... simulate a tilting (onlap) episode ... at output horizon number {}".format(
                            i
                        )
                    )
                onlaps_horizon_list.append(i)

                azi = np.random.uniform(low=0.0, high=360.0)
                azi_list.append(azi)
                dip = np.random.uniform(low=5.0, high=20.0)
                dip_list.append(dip)

                dipping_plane2 = (
                    self._fit_plane_strike_dip(
                        azi_list[count - 1],
                        dip_list[count - 1],
                        self.depth_maps.shape,
                        verbose=False,
                    )
                    * self.cfg.infill_factor
                )
                dipping_plane2_offset = (
                    self.thicknesses[i] * self.cfg.infill_factor / 2.0
                    - dipping_plane2.max()
                )

                # adjust all shallower layers
                previous_depth_map = self.depth_maps[:, :, i + 1].copy()
                for i_horizon in range(i, 0, -1):
                    current_depth_map = self.depth_maps[:, :, i_horizon].copy()
                    thickness_map = previous_depth_map - current_depth_map
                    prior_thickness_map = thickness_map.copy()
                    thickness_map += dipping_plane2
                    thickness_map += dipping_plane2_offset
                    thickness_map = np.clip(thickness_map, 0.0, thickness_map.max())
                    current_depth_map = previous_depth_map - thickness_map
                    if np.mean(thickness_map - prior_thickness_map) > 0.5:
                        layer_has_onlaps[i_horizon] = 1
                    self.depth_maps[:, :, i_horizon] = current_depth_map.copy()

        # print("\n\n ... depth_maps.shape = {}".format(depth_maps.shape))
        if self.cfg.verbose:
            print(
                f" ... Finished inserting tilting (onlap) episodes. {count} episodes were added..."
            )
        self.cfg.write_to_logfile(
            f"number_onlap_episodes: {count}\nonlaps_horizon_list: {str(onlaps_horizon_list)}"
        )

        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="number_onlap_episodes",
            val=count,
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="onlaps_horizon_list",
            val=str(onlaps_horizon_list),
        )
        self.onlap_horizon_list = onlaps_horizon_list
        return self.onlap_horizon_list


class BasinFloorFans(Horizons):
    def __init__(self, parameters, max_layers):
        self.cfg = parameters
        self.max_layers = max_layers
        self.fan_layers = None
        # self._generate_fan_lookup_table()

    def _generate_fan_lookup_table(self):
        # TODO test where & how often fans should be added. Also check if they overlap with onlaps and if so, what to do
        # Onlaps
        layers_with_fans = np.sort(
            np.random.uniform(
                low=5, high=self.max_layers - 1, size=int(np.random.choice([1, 2, 3]))
            ).astype("int")
        )
        self.fan_layers = layers_with_fans
        self.fan_thicknesses = []
        self.fans = np.zeros(self.max_layers, "int")
        self.fans[list(self.fan_layers)] = 1

    def _generate_basin_floor_fan(self, layer_number):
        # Select parameters for fan
        _scale = np.random.uniform(50.0, 200.0)  # length in pixels
        _aspect = np.random.uniform(1.5, 4.0)  # length  width
        _azimuth = np.random.uniform(0.0, 360.0)
        _factor = np.random.uniform(1.5, 3.5)
        _asymmetry_factor = np.random.uniform(-1.0, 1.0)
        _smoothing_size = np.random.uniform(1.33, 3.0)

        # Choose whether to create a pair of fans
        pair_of_fans = np.random.choice([True, False])

        fan_parameters = (
            f"{_scale:4.2f}, {_aspect:4.2f}, {_azimuth:4.2f}, {_factor:4.2f}, {_asymmetry_factor:4.2f},"
            f" {_smoothing_size:4.2f}"
        )
        if self.cfg.verbose:
            print(fan_parameters)
        zi = self._generate_fan_thickness_map(
            scale=_scale,
            aspect=_aspect,
            rotate_angle=_azimuth,
            dip=3.5,
            entry_point_factor=_factor,
            asymmetry_factor=_asymmetry_factor,
            smoothing_size=_smoothing_size,
        )

        if pair_of_fans:
            if self.cfg.verbose:
                print("Creating a pair of fans\n")
            _scale2 = _scale * np.random.uniform(0.667, 1.5)  # length in pixels
            _aspect2 = _aspect * np.random.uniform(0.75, 1.33)  # length / width
            _azimuth2 = _azimuth + np.random.uniform(-30.0, 30.0)
            delta_dist = (_scale + _scale2) / (
                (_aspect + _aspect2)
                * np.random.triangular(0.5, 0.85, 1.5)
                * np.random.choice([-1.0, 1.0])
            )  # length in pixels
            delta_x, delta_y = self.rotate_point(delta_dist, 0.0, 360.0 - _azimuth2)
            _factor2 = _factor + np.random.uniform(0.8, 1.25)
            _asymmetry_factor2 = _asymmetry_factor + np.random.uniform(-0.25, 0.25)
            _smoothing_size2 = _smoothing_size + np.random.uniform(-0.25, 0.25)

            fan_parameters_2 = (
                f"{_scale2:4.2f}, {_aspect2:4.2f}, {_azimuth2:4.2f}, {_factor2:4.2f},"
                f"{_asymmetry_factor2:4.2f}, {_smoothing_size2:4.2f}"
            )

            if self.cfg.verbose:
                print(
                    f"Fan 1 parameters: {fan_parameters}\nFan 2 parameters: {fan_parameters_2}"
                )
            fan_parameters += f"\n{fan_parameters_2}"
            zi2 = self._generate_fan_thickness_map(
                scale=_scale2,
                aspect=_aspect2,
                rotate_angle=_azimuth2,
                dip=3.5,
                entry_point_factor=_factor2,
                asymmetry_factor=_asymmetry_factor2,
                smoothing_size=_smoothing_size2,
            )

            zi2_padded = np.zeros((zi.shape[0] * 2, zi.shape[1] * 2), "float")
            zi2_padded[
                zi.shape[0] // 2 : zi.shape[0] + zi.shape[0] // 2,
                zi.shape[1] // 2 : zi.shape[1] + zi.shape[1] // 2,
            ] += zi2
            zi2_padded = np.roll(zi2_padded, int(delta_x + 0.5), axis=0)
            zi2_padded = np.roll(zi2_padded, int(delta_y + 0.5), axis=1)
            zi_delta = (
                zi2_padded[
                    zi.shape[0] // 2 : zi.shape[0] + zi.shape[0] // 2,
                    zi.shape[1] // 2 : zi.shape[1] + zi.shape[1] // 2,
                ]
                - zi
            )
            zi_delta[zi_delta < 0.0] = 0.0
            zi3 = zi + zi_delta
            zi = zi3

        # Plot the fan
        if self.cfg.qc_plots:

            plt = import_matplotlib()
            plt.figure(1)
            plt.clf()
            plt.title(fan_parameters)
            plt.imshow(zi.T)
            plt.colorbar()
            plt.savefig(
                os.path.join(
                    self.cfg.work_subfolder, f"random_fan_layer{layer_number}.png"
                )
            )

        self.fan_thicknesses.append(zi)
        return zi

    def _generate_fan_thickness_map(
        self,
        scale=2.5,
        aspect=1.5,
        rotate_angle=0.0,
        zero_at_corners=True,
        dip=3.5,
        entry_point_factor=1.0,
        asymmetry_factor=0.5,
        smoothing_size=2.5,
    ):
        ############################################################################
        # generate a 2D array representing a depth (or TWT) structure map.
        # - grid_size controls output size in (x,y), units are samples
        # - dip_range controls slope of dipping plane upon which random residual points are placed
        # - num_points controls number of randomly positioned points are used for residual grid
        # - asymetry_factor: range [-1, 1.]. 0. does nothing, sign determines left/right prominence
        ############################################################################

        grid_size = self.cfg.cube_shape[:2]
        # define output grid (padded by 15% in each direction)
        xi = np.linspace(-0.15, 1.15, int(float(grid_size[0]) * 1.3))
        yi = np.linspace(-0.15, 1.15, int(float(grid_size[1]) * 1.3))

        # start with a gently dipping plane similar to that found on passive shelf margins (< 1 degree dip)
        dipping_plane = self._fit_plane_strike_dip(
            360.0 - rotate_angle, dip, grid_size, verbose=False
        ).T

        x = np.array(
            [
                0.5 * (1.0 - np.cos(theta)) * np.sin(theta) * np.cos(phi)
                for phi in np.arange(0.0, np.pi, np.pi / 180.0)
                for theta in np.arange(0.0, np.pi, np.pi / 180.0)
            ]
        )
        y = np.array(
            [
                np.cos(theta)
                for phi in np.arange(0.0, np.pi, np.pi / 180.0)
                for theta in np.arange(0.0, np.pi, np.pi / 180.0)
            ]
        )
        z = np.array(
            [
                0.5 * (1.0 - np.cos(theta)) * np.sin(theta) * np.sin(phi)
                for phi in np.arange(0.0, np.pi, np.pi / 180.0)
                for theta in np.arange(0.0, np.pi, np.pi / 180.0)
            ]
        )
        x -= x.min()
        y -= y.min()
        if self.cfg.verbose:
            print("number of xyz points = ", x.size)
            print("x min/mean/max = ", x.min(), x.mean(), x.max(), x.max() - x.min())
        x *= (scale / aspect) / (x.max() - x.min())
        y *= scale / (y.max() - y.min())

        # scale back to range of [0,1]
        x /= grid_size[0]
        y /= grid_size[1]

        if self.cfg.verbose:
            print("x min/mean/max = ", x.min(), x.mean(), x.max(), x.max() - x.min())
            print("y min/mean/max = ", y.min(), y.mean(), y.max(), y.max() - y.min())

        # add asymmetry
        # - small rotation, stretch, undo rotation
        if asymmetry_factor != 0.0:
            x, y = self.rotate_point(x, y, asymmetry_factor * 45.0)
            x *= 2.0 + np.abs(asymmetry_factor)
            y *= 2.0 - np.abs(asymmetry_factor)
            x, y = self.rotate_point(x, y, -asymmetry_factor * 45.0)
            x /= 2.0
            y /= 2.0
        if self.cfg.verbose:
            print("x min/mean/max = ", x.min(), x.mean(), x.max(), x.max() - x.min())
            print("y min/mean/max = ", y.min(), y.mean(), y.max(), y.max() - y.min())

        # rotate
        x, y = self.rotate_point(x, y, 360.0 - rotate_angle)

        # center in the grid
        x += 0.5 - x.mean()
        y += 0.5 - y.mean()

        if self.cfg.verbose:
            print("x min/mean/max = ", x.min(), x.mean(), x.max(), x.max() - x.min())
            print("y min/mean/max = ", y.min(), y.mean(), y.max(), y.max() - y.min())
            print("z min/mean/max = ", z.min(), z.mean(), z.max(), z.max() - z.min())

        # decimate the grid randomly
        decimation_factor = 250

        fan_seed_x = np.random.randint(1, high=(2 ** (32 - 1)))
        fan_seed_y = np.random.randint(1, high=(2 ** (32 - 1)))
        fan_seed_z = np.random.randint(1, high=(2 ** (32 - 1)))
        point_indices = np.arange(x.size).astype("int")
        point_indices = point_indices[: x.size // decimation_factor]
        np.random.shuffle(point_indices)
        np.random.seed(fan_seed_x)
        _x = x + np.random.uniform(low=-0.03, high=0.03, size=x.size)
        np.random.seed(fan_seed_y)
        _y = y + np.random.uniform(low=-0.03, high=0.03, size=x.size)
        np.random.seed(fan_seed_z)
        _z = z + np.random.uniform(low=-0.1, high=0.1, size=x.size)

        _x = _x[point_indices]
        _y = _y[point_indices]
        _z = _z[point_indices]

        if self.cfg.qc_plots:
            plt = import_matplotlib()
            plt.clf()
            plt.grid()
            plt.scatter(x * 300, y * 300, c=z)
            plt.xlim([0, 300])
            plt.ylim([0, 300])
            plt.colorbar()

        if zero_at_corners:
            x = np.hstack((x, [-0.15, -0.15, 1.15, 1.15]))
            y = np.hstack((y, [-0.15, 1.15, -0.15, 1.15]))
            z = np.hstack((z, [-0.15, -0.15, -0.15, -0.15]))
            _x = np.hstack((_x, [-0.15, -0.15, 1.15, 1.15]))
            _y = np.hstack((_y, [-0.15, 1.15, -0.15, 1.15]))
            _z = np.hstack((_z, [-0.15, -0.15, -0.15, -0.15]))

        # grid the data.
        zi = griddata(
            (x, y),
            z,
            (xi.reshape(xi.shape[0], 1), yi.reshape(1, yi.shape[0])),
            method="linear",
        )
        _zi = griddata(
            (_x, _y),
            _z,
            (xi.reshape(xi.shape[0], 1), yi.reshape(1, yi.shape[0])),
            method="cubic",
        )

        zi[np.isnan(zi)] = 0.0
        zi[zi <= 0.0] = 0.0
        zi_mask = zi.copy()
        zi_mask[zi_mask <= 0.015] = 0.0
        if self.cfg.verbose:
            print("zi_mask[zi_mask >0.].min() = ", zi_mask[zi_mask > 0.0].min())
            print("zi_mask[zi_mask >0.].mean() = ", zi_mask[zi_mask > 0.0].mean())
            print(
                "np.percentile(zi_mask[zi_mask >0.],.5) = ",
                np.percentile(zi_mask[zi_mask > 0.0], 0.5),
            )

        _zi[np.isnan(_zi)] = 0.0
        zi = _zi + 0.0
        zi[zi_mask <= 0.0] = 0.0
        zi[zi <= 0.0] = 0.0

        zi = gaussian_filter(zi, smoothing_size)
        # For cube_shape 300, 300, use closing size of (10, 10)
        closing_size = (
            int(self.cfg.cube_shape[0] / 30),
            int(self.cfg.cube_shape[1] / 30),
        )
        zi = grey_closing(zi, size=closing_size)

        zi = zi.reshape(xi.shape[0], yi.shape[0])
        xi_min_index = np.argmin((xi - 0.0) ** 2)
        xi_max_index = xi_min_index + grid_size[0]
        yi_min_index = np.argmin((yi - 0.0) ** 2)
        yi_max_index = yi_min_index + grid_size[1]
        zi = zi[xi_min_index:xi_max_index, yi_min_index:yi_max_index]
        # add to the gently sloping plane
        dipping_plane[zi == 0.0] = 0.0
        dipping_plane -= dipping_plane[dipping_plane != 0.0].min()
        dipping_plane[zi <= 0.05] = 0.0
        if dipping_plane.max() > 0:
            dipping_plane /= dipping_plane[dipping_plane != 0.0].max()
        dipping_plane *= zi.max()
        if self.cfg.verbose:
            print(
                "zi[zi > 0.] min/mean/max = ",
                zi[zi > 0.0].min(),
                zi[zi > 0.0].mean(),
                zi[zi > 0.0].max(),
            )
        if dipping_plane.max() > 0:
            if self.cfg.verbose:
                print(
                    "dipping_plane[dipping_plane > 0.] min/mean/max = ",
                    dipping_plane[dipping_plane > 0.0].min(),
                    dipping_plane[dipping_plane > 0.0].mean(),
                    dipping_plane[dipping_plane > 0.0].max(),
                )
        else:
            if self.cfg.verbose:
                print(
                    "dipping_plane min/mean/max = ",
                    dipping_plane.min(),
                    dipping_plane.mean(),
                    dipping_plane.max(),
                )
        zi += entry_point_factor * dipping_plane

        # todo QC size of this smoothing (to smooth edges of fan)
        zi = gaussian_filter(zi, smoothing_size * 2.0)

        return zi

    def insert_fans_into_horizons(self, depth_maps):
        new_depth_maps = depth_maps[:].copy()
        # Create lookup table for fans
        self._generate_fan_lookup_table()
        # generate fans using fan_layers
        for layer in self.fan_layers:
            thickness_map = (
                self._generate_basin_floor_fan(layer) * self.cfg.infill_factor
            )
            new_depth_maps = self.insert_feature_into_horizon_stack(
                thickness_map, layer, new_depth_maps
            )
        # Write fan layers to logfile
        self.cfg.write_to_logfile(
            f"number_fan_episodes: {len(self.fan_layers)}\n"
            f"fans_horizon_list: {str(self.fan_layers)}"
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="number_fan_episodes",
            val=len(self.fan_layers),
        )
        self.cfg.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="fan_horizon_list",
            val=str(self.fan_layers),
        )

        return new_depth_maps

    def fan_qc_plot(self, maps, lyrnum, thickness):
        """
        Plot a cross-section of the inserted basin floor fan

        Display an inline and crossline with the basin floor fan in red in cross-section
        Inlay a map of the fan in each subplot
        """
        plt = import_matplotlib()
        plt.close()
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharey=True)
        axs[0].set_title(f"Layer {lyrnum}")
        axs[1].set_title(f"Layer {lyrnum}")
        #
        max_layer = np.clip(
            lyrnum + 10, 0, maps.shape[2]
        )  # only plot layers below fan if they exist
        for j in range(lyrnum - 5, max_layer, 1):
            if j == lyrnum:  # highlight fan layers in thicker red line
                axs[0].plot(
                    range(self.cfg.cube_shape[0]),
                    maps[int(self.cfg.cube_shape[0] / 2), :, j],
                    "r-",
                    lw=0.5,
                )
                axs[1].plot(
                    range(self.cfg.cube_shape[0]),
                    maps[:, int(self.cfg.cube_shape[1] / 2), j],
                    "r-",
                    lw=0.5,
                )
            else:
                axs[0].plot(
                    range(self.cfg.cube_shape[0]),
                    maps[int(self.cfg.cube_shape[0] / 2), :, j],
                    "k-",
                    lw=0.2,
                )
                axs[1].plot(
                    range(self.cfg.cube_shape[0]),
                    maps[:, int(self.cfg.cube_shape[1] / 2), j],
                    "k-",
                    lw=0.2,
                )
        for inl, ax in enumerate(axs):
            if inl == 0:
                plt.axes([0.8, 0.75, 0.11, 0.11])
                plt.axvline(
                    x=int(self.cfg.cube_shape[0] / 2), color="grey", linestyle="--"
                )
            else:
                plt.axes([0.8, 0.33, 0.11, 0.11])
                plt.axhline(
                    y=int(self.cfg.cube_shape[1] / 2), color="grey", linestyle="--"
                )
            plt.imshow(thickness.T)
            plt.xticks([])
            plt.yticks([])
            ax.set_ylim(
                top=np.min(maps[..., lyrnum - 5]),
                bottom=np.max(maps[..., max_layer - 1]),
            )  # Reverse Y axis
        fig.savefig(
            os.path.join(self.cfg.work_subfolder, f"BasinFloorFan_layer_{lyrnum}.png")
        )
        plt.close()


class Channel(Horizons):
    def __init__(self):
        self.W = 200.0  # Channel wodth
        self.D = 12.0  # Channel depth
        self.pad = 100  # padding (number of nodepoints along centerline)
        self.deltas = 50.0  # sampling distance along centerline
        self.nit = 2000  # number of iterations
        self.Cf = 0.022  # dimensionless Chezy friction factor
        self.crdist = 1.5 * self.W  # threshold distance at which cutoffs occur
        self.kl = 60.0 / (365 * 24 * 60 * 60.0)  # migration rate constant (m/s)
        self.kv = 1.0e-11  # vertical slope-dependent erosion rate constant (m/s)
        self.dt = 2 * 0.05 * 365 * 24 * 60 * 60.0  # time step (s)
        self.dens = 1000  # density of water (kg/m3)
        self.saved_ts = 20  # which time steps will be saved
        self.n_bends = 30  # approximate number of bends to model
        self.Sl = 0.0  # initial slope (matters more for submarine channels than rivers)
        self.t1 = 500  # time step when incision starts
        self.t2 = 700  # time step when lateral migration starts
        self.t3 = 1400  # time step when aggradation starts
        self.aggr_factor = (
            4e-9  # aggradation factor (m/s, about 0.18 m/year, it kicks in after t3)
        )

        self.h_mud = 0.3  # thickness of overbank deposits for each time step
        self.dx = 20.0  # gridcell size in metres

        # Channel objects
        self.ch = None
        self.chb = None
        self.chb_3d = None

    def generate_channel_parameters(self):
        # select random parameters
        pass

    def create_channel_3d(self):
        # Initialise channel
        self.ch = mp.generate_initial_channel(
            self.W, self.D, self.Sl, self.deltas, self.pad, self.n_bends
        )
        # Create channel belt object
        self.chb = mp.ChannelBelt(
            channels=[self.ch], cutoffs=[], cl_times=[0.0], cutoff_times=[]
        )
        # Migrate channel
        self.chb.migrate(
            self.nit,
            self.saved_ts,
            self.deltas,
            self.pad,
            self.crdist,
            self.Cf,
            self.kl,
            self.kv,
            self.dt,
            self.dens,
            self.t1,
            self.t2,
            self.t3,
            self.aggr_factor,
        )
        end_time = self.chb.cl_times[-1]
        _xmin = 15000
        _xmax = 21000
        self.chb_3d, _, _, _, _ = self.chb.build_3d_model(
            "fluvial",
            h_mud=self.h_mud,
            levee_width=800.0,
            h=12.0,
            w=self.W,
            bth=0.0,
            dcr=10.0,
            dx=self.dx,
            delta_s=self.deltas,
            starttime=self.chb.cl_times[0],
            endtime=end_time,
            xmin=_xmin,
            xmax=_xmax,
            ymin=-3000,
            ymax=3000,
        )

    def insert_channel_into_horizons(self, layer, depth_maps):
        new_depth_maps = self.insert_feature_into_horizon_stack(
            self.chb.strat, layer, depth_maps
        )
        return new_depth_maps

    def insert_channel_facies(self):
        pass


class Facies:
    def __init__(self, parameters, max_layers, onlap_horizon_list, fan_horizon_list):
        self.cfg = parameters
        self.max_layers = max_layers
        self.onlap_horizon_list = onlap_horizon_list
        self.fan_horizon_list = fan_horizon_list
        self.facies = None

    def sand_shale_facies_binomial_dist(self):
        """Randomly select sand or shale facies using binomial distribution using the sand_layer_pct from config file"""
        sand_layer = np.random.binomial(
            1, self.cfg.sand_layer_pct, size=self.max_layers
        )
        # Insert a water-layer code of -1 at the top
        self.facies = np.hstack((np.array((-1.0)), sand_layer))

    def sand_shale_facies_markov(self):
        """Generate a 1D array of facies usinga 2-state Markov process

        Note: the Binomial distribution can be generated by setting the sand_layer_thickness to 1/sand_layer_pct
        """
        mk = MarkovChainFacies(
            self.cfg.sand_layer_pct, self.cfg.sand_layer_thickness, (0, 1)
        )
        # Randomly select initial state
        facies = mk.generate_states(np.random.choice(2, 1)[0], num=self.max_layers)
        self.facies = np.hstack((np.array((-1.0)), facies))

    def set_layers_below_onlaps_to_shale(self):
        """
        Set layers immediately below tilting episode (onlap surface) to shale.

        Set facies array to 0 (shale) above any layer which is marked as being an onlap surface
        """
        onlap_horizon_list_plus_one = np.array(self.onlap_horizon_list)
        onlap_horizon_list_plus_one += 1
        self.facies[onlap_horizon_list_plus_one] = 0.0

    def set_fan_facies(self, depth_maps):
        """Set fan facies as sand layers surrounded by shales

        Set the fan layer to be a sand.
        Set the layers above and below the fan to be shale
        """
        # Find those layers which are directly above (and touching) the fan layer
        layers_with_zero_thickness = set(np.where(np.diff(depth_maps) == 0)[2])
        layers_above_fan = []
        for f in self.fan_horizon_list:
            for i in range(f - 1, 1, -1):
                if i in layers_with_zero_thickness:
                    layers_above_fan.append(i)
                else:
                    break
        sand = list(self.fan_horizon_list)
        shale = layers_above_fan + list(self.fan_horizon_list + 1)
        self.facies[sand] = 1.0
        self.facies[shale] = 0.0


class MarkovChainFacies:
    def __init__(self, sand_fraction, sand_thickness, states=(0, 1)):
        """Initialize the MarkovChain instance.

        Parameters
        ----------
        sand_fraction : float
            Fraction of sand layers in model
        sand_thickness : int
            Thickness of sand layers, given in units of layers
        states : iterable
            An iterable representing the states of the Markov Chain.
        """
        self.sand_fraction = sand_fraction
        self.sand_thickness = sand_thickness
        self.states = states
        self.index_dict = {
            self.states[index]: index for index in range(len(self.states))
        }
        self.state_dict = {
            index: self.states[index] for index in range(len(self.states))
        }
        self._transition_matrix()

    def _transition_matrix(self):
        """TODO Stephs notes go here"""
        beta = 1 / self.sand_thickness
        alpha = self.sand_fraction / (self.sand_thickness * (1 - self.sand_fraction))
        self.transition = np.array([[1 - alpha, alpha], [beta, 1 - beta]])

    def next_state(self, current_state):
        """Returns the state of the random variable at the next instance.

        Parameters
        ----------
        current_state :str
            The current state of the system
        """
        return np.random.choice(
            self.states, p=self.transition[self.index_dict[current_state], :]
        )

    def generate_states(self, current_state, num=100):
        """Generate states of the system with length num

        Parameters
        ----------
        current_state : str
            The state of the current random variable
        num : int, optional
            [description], by default 100
        """
        future_states = []
        for _ in range(num):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return np.array(future_states)


def build_unfaulted_depth_maps(parameters: Parameters):
    """
    Build Unfaulted Depth Maps
    --------------------------
    Generates unfaulted depth maps.

    1. Build stack of horizons.
    2. Generate a random stack of horizons.
    3. Optionally insert basin floor fans
    4. Insert onlap episodes

    Parameters
    ----------
    parameters : str
        The key desired to be accessed

    Returns
    -------
    depth_maps : np.array
        The generated depth maps
    onlap_horizon_list : list
        Onlapping Horizon list
    fan_list : np.array | None
        List of generated fans
    fan_thicknesses : np.array | None
        Generated fan thicknesses
    """
    horizons = RandomHorizonStack(parameters)

    # Insert onlap episodes
    onlaps = Onlaps(
        parameters, horizons.depth_maps, horizons.thicknesses, horizons.max_layers
    )
    onlap_horizon_list = onlaps.insert_tilting_episodes()
    # Insert seafloor
    depth_maps = horizons.insert_seafloor(horizons.depth_maps)

    fan_list = None
    fan_thicknesses = None
    if parameters.basin_floor_fans:
        bff = BasinFloorFans(parameters, horizons.max_layers)
        # Insert Fans
        depth_maps = bff.insert_fans_into_horizons(horizons.depth_maps)
        for layer, thickness in zip(bff.fan_layers, bff.fan_thicknesses):
            bff.fan_qc_plot(depth_maps, layer, thickness)
        fan_list = bff.fan_layers
        fan_thicknesses = bff.fan_thicknesses
    return depth_maps, onlap_horizon_list, fan_list, fan_thicknesses


def create_facies_array(
    parameters: Parameters,
    depth_maps: np.ndarray,
    onlap_horizons: list,
    fan_horizons: np.ndarray = None,
) -> np.ndarray:
    """
    Create Facies Array
    --------------------------
    Generates facies for the model and return an array with the facies.

    Parameters
    ----------
    parameters : datagenerator.Parameters
        Parameter object storing all model parameters.
    depth_maps : np.ndarray
        A numpy array containing the depth maps.
    onlap_horizons : list
        A list of the onlap horizons.
    fan_horizons : np.ndarray
        The fan horizons.

    Returns
    -------
    facies : np.ndarray
        An array that contains the facies for the model.
    """
    facies = Facies(parameters, depth_maps.shape[-1], onlap_horizons, fan_horizons)
    # facies.sand_shale_facies_binomial_dist()
    facies.sand_shale_facies_markov()
    if len(onlap_horizons) > 0:
        facies.set_layers_below_onlaps_to_shale()
    if np.any(fan_horizons):
        facies.set_fan_facies(depth_maps)
    # Write sand layer % to logfile
    sand_pct = facies.facies[facies.facies == 1].size / facies.facies.size
    parameters.write_to_logfile(
        f"Percent of layers containing sand a priori: {parameters.sand_layer_pct:0.2%}",
        mainkey="model_parameters",
        subkey="sand_layer_thickness_a_priori",
        val=parameters.sand_layer_pct,
    )
    parameters.write_to_logfile(
        f"Sand unit thickness (in layers) a priori: {parameters.sand_layer_thickness}",
        mainkey="model_parameters",
        subkey="sand_unit_thickness_a_priori",
        val=parameters.sand_layer_thickness,
    )
    parameters.write_to_logfile(
        f"Percent of layers containing sand a posteriori: {sand_pct:.2%}",
        mainkey="model_parameters",
        subkey="sand_layer_percent_a_posteriori",
        val=sand_pct,
    )
    # Count consecutive occurrences (i.e. facies thickness in layers)
    facies_groups = [(k, sum(1 for i in g)) for k, g in groupby(facies.facies)]
    if parameters.verbose:
        print(f"Facies units: {facies_groups}")
    avg_sand_thickness = np.nan_to_num(
        np.mean([b for a, b in facies_groups if a == 1.0])
    )  # convert nan to 0
    parameters.write_to_logfile(
        f"Average Sand unit thickness (in layers) a posteriori: {avg_sand_thickness:.1f}",
        mainkey="model_parameters",
        subkey="sand_unit_thickness_a_posteriori",
        val=avg_sand_thickness,
    )

    return facies.facies


if __name__ == "__main__":
    par = Parameters(user_config="../config/config_bps.json", test_mode=100)
    par.setup_model()
    # p.cube_shape = (100, 100, 1250)
    zmaps, onlaps, fan_list, fan_thickness = build_unfaulted_depth_maps(par)
    facies = create_facies_array(par, zmaps, onlaps, fan_list)
    print(facies)
