from collections import defaultdict
import datetime
import json
import multiprocessing as mp
import os
import pathlib
import glob
import sqlite3
from subprocess import CalledProcessError
import numpy as np
import zarr
import numcodecs
from typing import Optional, List, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.dataclasses import dataclass
from functools import lru_cache
import subprocess

dir_name = pathlib.Path(__file__).parent
CONFIG_PATH = (dir_name / "../config/config_ht.json").resolve()


class SandLayerFraction(BaseModel):
    """Configuration for sand layer fraction ranges."""

    min: float = Field(..., ge=0.0, le=1.0, description="Minimum sand layer fraction")
    max: float = Field(..., ge=0.0, le=1.0, description="Maximum sand layer fraction")

    @validator("max")
    def max_greater_than_min(cls, v, values):
        if "min" in values and v < values["min"]:
            raise ValueError("max must be greater than min")
        return v


class ModelConfig(BaseModel):
    """Base configuration model for seismic data generation."""

    project: str = Field(..., description="Name of project")
    project_folder: str = Field(..., description="Output directory for models")
    work_folder: str = Field(..., description="Temporary folder for intermediate data")
    cube_shape: Tuple[int, int, int] = Field(
        ..., description="Number of samples in [X, Y, Z]"
    )
    incident_angles: Tuple[float, ...] = Field(
        ..., description="Central angles for output seismic angle-stacks"
    )
    digi: int = Field(..., gt=0, description="Digitization factor")
    infill_factor: int = Field(
        ..., gt=0, description="Infill factor for model generation"
    )
    initial_layer_stdev: Tuple[float, float] = Field(
        ..., description="Initial layer standard deviation range"
    )
    thickness_min: int = Field(..., gt=0, description="Minimum layer thickness")
    thickness_max: int = Field(..., gt=0, description="Maximum layer thickness")
    seabed_min_depth: Union[int, Tuple[int, int]] = Field(
        ..., description="Minimum seabed depth"
    )
    signal_to_noise_ratio_db: Tuple[float, float, float] = Field(
        ..., description="Signal to noise ratio in dB"
    )
    bandwidth_low: Tuple[float, float] = Field(
        ..., description="Low frequency bandwidth range"
    )
    bandwidth_high: Tuple[float, float] = Field(
        ..., description="High frequency bandwidth range"
    )
    bandwidth_ord: int = Field(..., gt=0, description="Bandwidth order")
    dip_factor_max: float = Field(..., gt=0, description="Maximum dip factor")
    min_number_faults: int = Field(..., ge=0, description="Minimum number of faults")
    max_number_faults: int = Field(..., gt=0, description="Maximum number of faults")
    max_column_height: Tuple[float, float] = Field(
        ..., description="Maximum column height range"
    )
    closure_types: List[str] = Field(..., description="Types of closures to generate")
    min_closure_voxels_simple: int = Field(
        ..., gt=0, description="Minimum voxels for simple closures"
    )
    min_closure_voxels_faulted: int = Field(
        ..., gt=0, description="Minimum voxels for faulted closures"
    )
    min_closure_voxels_onlap: int = Field(
        ..., gt=0, description="Minimum voxels for onlap closures"
    )
    sand_layer_thickness: int = Field(..., gt=0, description="Sand layer thickness")
    sand_layer_fraction: SandLayerFraction = Field(
        ..., description="Sand layer fraction configuration"
    )
    extra_qc_plots: bool = Field(False, description="Enable additional QC plots")
    verbose: bool = Field(False, description="Enable verbose output")
    partial_voxels: bool = Field(
        True,
        description="Calculate average properties for voxels spanning multiple layers",
    )
    variable_shale_ng: bool = Field(
        False, description="Enable variable net-to-gross in shale layers"
    )
    basin_floor_fans: bool = Field(False, description="Enable basin floor fan features")
    include_channels: bool = Field(
        False, description="Enable channel features (deprecated)"
    )
    include_salt: bool = Field(False, description="Enable salt bodies")
    write_to_hdf: bool = Field(False, description="Write QC volumes to HDF file")
    broadband_qc_volume: bool = Field(
        False, description="Output broadband seismic data"
    )
    model_qc_volumes: bool = Field(True, description="Save QC volumes to disk")
    multiprocess_bp: bool = Field(
        True, description="Use multiprocessing for bandpass operations"
    )
    pad_samples: int = Field(10, gt=0, description="Number of padding samples")

    @validator("max_number_faults")
    def max_faults_greater_than_min(cls, v, values):
        if "min_number_faults" in values and v < values["min_number_faults"]:
            raise ValueError("max_number_faults must be greater than min_number_faults")
        return v

    @validator("thickness_max")
    def thickness_max_greater_than_min(cls, v, values):
        if "thickness_min" in values and v < values["thickness_min"]:
            raise ValueError("thickness_max must be greater than thickness_min")
        return v

    @validator("bandwidth_high")
    def bandwidth_high_greater_than_low(cls, v, values):
        if "bandwidth_low" in values and v[0] < values["bandwidth_low"][1]:
            raise ValueError("bandwidth_high must be greater than bandwidth_low")
        return v

    @validator("signal_to_noise_ratio_db")
    def validate_snr_db(cls, v):
        if not (v[0] < v[1] < v[2]):
            raise ValueError("signal_to_noise_ratio_db must be in ascending order")
        return v


class RPMScalingFactors(BaseModel):
    """Rock Physics Model scaling factors."""

    layershiftsamples: int = Field(..., gt=0)
    RPshiftsamples: int = Field(..., gt=0)
    shalerho_factor: float = Field(1.0, gt=0)
    shalevp_factor: float = Field(1.0, gt=0)
    shalevs_factor: float = Field(1.0, gt=0)
    sandrho_factor: float = Field(1.0, gt=0)
    sandvp_factor: float = Field(1.0, gt=0)
    sandvs_factor: float = Field(1.0, gt=0)
    nearfactor: float = Field(1.0, gt=0)
    midfactor: float = Field(1.0, gt=0)
    farfactor: float = Field(1.0, gt=0)


class Parameters:
    """
    Modernized parameter object for seismic data generation using Pydantic models.
    Maintains backward compatibility with the original implementation while providing
    better type safety and validation.
    """

    def __init__(
        self,
        user_config: str = CONFIG_PATH,
        test_mode: Optional[int] = None,
        runid: Optional[str] = None,
    ):
        """
        Initialize the Parameters object.

        Parameters
        ----------
        user_config : str, optional
            Path to the JSON configuration file, by default CONFIG_PATH
        test_mode : Optional[int], optional
            Test mode size for reduced model generation, by default None
        runid : Optional[str], optional
            Run identifier for multiple runs, by default None
        """
        self.model_dir_name: str = "seismic"
        self.parameter_file = user_config
        self.test_mode = test_mode
        self.runid = runid
        self.rpm_scaling_factors: Dict[str, Any] = {}
        self.sqldict = defaultdict(dict)

        # Load and validate configuration
        self._config = self._load_config()
        self._setup_from_config()

    @lru_cache()
    def _load_config(self) -> ModelConfig:
        """Load and validate configuration from JSON file."""
        with open(self.parameter_file) as f:
            config_dict = json.load(f)
        return ModelConfig(**config_dict)

    def _setup_from_config(self) -> None:
        """Set up parameters from validated configuration."""
        config = self._config

        # Set basic attributes
        self.project = config.project
        self.project_folder = config.project_folder
        work_folder_path = pathlib.Path(config.work_folder)
        self.work_folder = str(
            work_folder_path if work_folder_path.exists() else pathlib.Path("/tmp")
        )

        # Set model parameters
        self.cube_shape = config.cube_shape
        self.incident_angles = config.incident_angles
        self.digi = config.digi
        self.infill_factor = config.infill_factor
        self.lyr_stdev = config.initial_layer_stdev
        self.thickness_min = config.thickness_min
        self.thickness_max = config.thickness_max
        self.seabed_min_depth = config.seabed_min_depth
        self.snr_db = config.signal_to_noise_ratio_db
        self.bandwidth_low = config.bandwidth_low
        self.bandwidth_high = config.bandwidth_high
        self.bandwidth_ord = config.bandwidth_ord
        self.dip_factor_max = config.dip_factor_max
        self.min_number_faults = config.min_number_faults
        self.max_number_faults = config.max_number_faults
        self.basin_floor_fans = config.basin_floor_fans
        self.pad_samples = config.pad_samples
        self.qc_plots = config.extra_qc_plots
        self.verbose = config.verbose
        self.include_channels = config.include_channels
        self.include_salt = config.include_salt
        self.max_column_height = config.max_column_height
        self.closure_types = config.closure_types
        self.closure_min_voxels_simple = config.min_closure_voxels_simple
        self.closure_min_voxels_faulted = config.min_closure_voxels_faulted
        self.closure_min_voxels_onlap = config.min_closure_voxels_onlap
        self.partial_voxels = config.partial_voxels
        self.variable_shale_ng = config.variable_shale_ng
        self.sand_layer_thickness = config.sand_layer_thickness
        self.sand_layer_pct_min = config.sand_layer_fraction.min
        self.sand_layer_pct_max = config.sand_layer_fraction.max
        self.hdf_store = config.write_to_hdf
        self.broadband_qc_volume = config.broadband_qc_volume
        self.model_qc_volumes = config.model_qc_volumes
        self.multiprocess_bp = config.multiprocess_bp

    def setup_model(self, rpm_factors: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the model with all necessary parameters and directories.

        Parameters
        ----------
        rpm_factors : Optional[Dict[str, Any]], optional
            Rock physics model factors, by default None
        """
        self._set_model_parameters(self.model_dir_name)
        self.make_directories()
        # self.write_key_file()
        self._setup_rpm_scaling_factors(rpm_factors)
        self._write_initial_model_parameters_to_logfile()

    def make_directories(self) -> None:
        """
        Make directories.
        -----------------

        Creates the necessary directories to run the model.

        This function creates the directories on disk
        necessary for the model to run.

        Parameters
        ----------
        self : `Parameters`

        Returns
        -------
        None
        """
        print(f"\nModel folder: {self.work_subfolder}")
        self.sqldict["model_id"] = pathlib.Path(self.work_subfolder).name
        for folder in [self.project_folder, self.work_subfolder, self.temp_folder]:
            folder_path = pathlib.Path(folder)
            if not folder_path.exists():
                print(f"Creating directory: {folder}")
                try:
                    folder_path.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
        try:
            os.system(f"chmod -R 777 {self.work_subfolder}")
        except OSError:
            print(f"Could not chmod {self.work_subfolder}. Continuing...")
            pass

    def write_key_file(self) -> None:
        """
        Write key file
        --------------

        Writes a file that contains important parameters about the cube.

        Method that writes important parameters about the synthetic cube
        such as coordinate transforms and sizes.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Set plausible key file values
        geom_expand = dict()
        geom_expand["3D_NAME"] = "synthetic data for training"
        geom_expand["COORD_METHOD"] = 1
        geom_expand["DATA_TYPE"] = "3D"
        geom_expand["DELTA_BIN_NUM"] = 1
        geom_expand["DELTA_TRACK_NUM"] = 1
        geom_expand["DIGITIZATION"] = 4
        geom_expand["EPSG_CRS"] = 32066
        geom_expand["FIRST_BIN"] = 1000
        geom_expand["FIRST_TRACK"] = 2000
        geom_expand["FORMAT"] = 1
        geom_expand["N_BIN"] = self.cube_shape[1]
        geom_expand["N_SAMP"] = self.cube_shape[2]
        geom_expand["N_TRACK"] = self.cube_shape[0]
        geom_expand["PROJECTION"] = 316
        geom_expand["REAL_DELTA_X"] = 100.0
        geom_expand["REAL_DELTA_Y"] = 100.0
        geom_expand["REAL_GEO_X"] = 1250000.0
        geom_expand["REAL_GEO_Y"] = 10500000.0
        geom_expand["SKEW_ANGLE"] = 0.0
        geom_expand["SUBPOINT_CODE"] = "TTTBBB"
        geom_expand["TIME_OR_DEPTH"] = "TIME"
        geom_expand["TRACK_DIR"] = "H"
        geom_expand["XFORM_TO_WGS84"] = 1241
        geom_expand["ZERO_TIME"] = 0

        # Write the keyfile
        outputkey = (
            pathlib.Path(self.work_subfolder) / f"seismicCube_{self.date_stamp}.key"
        )
        with open(outputkey, "w") as key:
            key.write(
                "{}MESSAGE_FILE\n".format(20 * " ")
            )  # spaces are important here.. Require 20 of them
            key.write("3D_NAME C %s\n" % geom_expand["3D_NAME"])
            key.write("COORD_METHOD I %d\n" % int(geom_expand["COORD_METHOD"]))
            key.write("DATA_TYPE C %s\n" % geom_expand["DATA_TYPE"])
            key.write("DELTA_BIN_NUM I %d\n" % int(geom_expand["DELTA_BIN_NUM"]))
            key.write("DELTA_TRACK_NUM I %d\n" % int(geom_expand["DELTA_TRACK_NUM"]))
            key.write("DIGITIZATION I %d\n" % int(geom_expand["DIGITIZATION"]))
            key.write("EPSG_CRS I %d\n" % int(geom_expand["EPSG_CRS"]))
            key.write("FIRST_BIN I %d\n" % int(geom_expand["FIRST_BIN"]))
            key.write("FIRST_TRACK I %d\n" % int(geom_expand["FIRST_TRACK"]))
            key.write("FORMAT I %d\n" % int(geom_expand["FORMAT"]))
            key.write("N_BIN I %d\n" % int(geom_expand["N_BIN"]))
            key.write("N_SAMP I %d\n" % int(geom_expand["N_SAMP"]))
            key.write("N_TRACK I %d\n" % int(geom_expand["N_TRACK"]))
            key.write("PROJECTION I %d\n" % int(geom_expand["PROJECTION"]))
            key.write("REAL_DELTA_X R %f\n" % float(geom_expand["REAL_DELTA_X"]))
            key.write("REAL_DELTA_Y R %f\n" % float(geom_expand["REAL_DELTA_Y"]))
            key.write("REAL_GEO_X R %f\n" % float(geom_expand["REAL_GEO_X"]))
            key.write("REAL_GEO_Y R %f\n" % float(geom_expand["REAL_GEO_Y"]))
            key.write("SKEW_ANGLE R %f\n" % float(geom_expand["SKEW_ANGLE"]))
            key.write("SUBPOINT_CODE C %s\n" % geom_expand["SUBPOINT_CODE"])
            key.write("TIME_OR_DEPTH C %s\n" % geom_expand["TIME_OR_DEPTH"])
            key.write("TRACK_DIR C %s\n" % geom_expand["TRACK_DIR"])
            key.write("XFORM_TO_WGS84 I %d\n" % int(geom_expand["XFORM_TO_WGS84"]))
            key.write("ZERO_TIME I %d\n" % int(geom_expand["ZERO_TIME"]))
        print(f"\nKeyfile created at {outputkey}")

    def write_to_logfile(self, msg, mainkey=None, subkey=None, val="") -> None:
        """
        write_to_logfile

        Method to write msg to model_parameter file
        (includes newline)

        Parameters
        ----------
        msg : `string`
        Required string object that will be written tom model parameter file.
        mainkey : `string`
        String of the key to be written into de sql dictionary.
        subkey : `string`
        String of the subkey to be written into de sql dictionary.
        val : `string`
        String of the value that should be written into the sql dictionary.

        Returns
        -------
        None
        """
        if msg is not None:
            with open(self.logfile, "a") as f:
                f.write(f"{msg}\n")
        if mainkey is not None:
            self.sqldict[mainkey][subkey] = val
            # for k, v in self.sqldict.items():
            #     print(f"{k}: {v}")

    def write_sqldict_to_logfile(self, logfile=None) -> None:
        """
        write_sqldict_to_logfile

        Write the sql dictionary to the logfile

        Parameters
        ----------
        logfile : `string`
        The path to the log file. By default None

        Returns
        -------
        None
        """
        if logfile is None:
            logfile = self.logfile
        with open(logfile, "a") as f:
            for k, nested in self.sqldict.items():
                print(k, file=f)
                if k == "model_id":
                    print(f"\t{nested}", file=f)
                else:
                    for subkey, value in nested.items():
                        print(f"\t{subkey}: {value}", file=f)
                print(file=f)

    def write_sqldict_to_db(self) -> None:
        """
        write_sqldict_to_db

        Method to write the sqldict to database sqlite file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        model_id = pathlib.Path(self.work_subfolder).name
        model_parameters = self.sqldict["model_parameters"]
        fault_keys = [k for k in self.sqldict.keys() if "fault" in k]
        closure_keys = [k for k in self.sqldict.keys() if "closure" in k]

        db_path = pathlib.Path(self.work_subfolder) / "parameters.db"
        conn = sqlite3.connect(str(db_path))
        # tables = ["model_parameters", "fault_parameters", "closure_parameters"]
        # create tables
        sql = f"CREATE TABLE model_parameters (model_id string primary key, {','.join(model_parameters.keys())})"
        conn.execute(sql)
        # insert model_parameters
        columns = "model_id, " + ", ".join(model_parameters.keys())
        placeholders = ", ".join("?" * (len(model_parameters) + 1))
        sql = f"INSERT INTO model_parameters ({columns}) VALUES ({placeholders})"
        values = tuple([model_id] + [str(x) for x in model_parameters.values()])
        conn.execute(sql, values)
        conn.commit()

        # fault parameters
        if len(fault_keys) > 0:
            f = tuple(self.sqldict[fault_keys[0]].keys())
            sql = f"CREATE TABLE fault_parameters ({','.join(f)})"
            conn.execute(sql)
            columns = ", ".join(self.sqldict[fault_keys[0]].keys())
            placeholders = ", ".join("?" * len(self.sqldict[fault_keys[0]].keys()))
            for f in fault_keys:
                sql = (
                    f"INSERT INTO fault_parameters ({columns}) VALUES ({placeholders})"
                )
                conn.execute(sql, tuple(self.sqldict[f].values()))
                conn.commit()

        if len(closure_keys) > 0:
            c = tuple(self.sqldict[closure_keys[0]].keys())
            sql = f"CREATE TABLE closure_parameters ({','.join(c)})"
            conn.execute(sql)
            columns = ", ".join(self.sqldict[closure_keys[0]].keys())
            placeholders = ", ".join("?" * len(self.sqldict[closure_keys[0]].keys()))
            for c in closure_keys:
                sql = f"INSERT INTO closure_parameters ({columns}) VALUES ({placeholders})"
                conn.execute(sql, tuple(self.sqldict[c].values()))
                conn.commit()

    def _setup_rpm_scaling_factors(
        self, rpm_factors: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set up rock physics model scaling factors with validation."""
        if rpm_factors and not self.test_mode:
            self.rpm_scaling_factors = RPMScalingFactors(**rpm_factors).dict()
        else:
            # Use default RPM factors
            self.rpm_scaling_factors = RPMScalingFactors(
                layershiftsamples=int(np.random.triangular(35, 75, 125)),
                RPshiftsamples=int(np.random.triangular(5, 11, 20)),
            ).dict()

        # Write factors to logfile
        for k, v in self.rpm_scaling_factors.items():
            self.write_to_logfile(
                msg=f"{k}: {v}", mainkey="model_parameters", subkey=k, val=v
            )

    def _set_model_parameters(self, dname: str) -> None:
        """
        Set Model Parameters
        ----------------------------------------

        Method that sets model parameters from user-provided
        config.json file

        Parameters
        ----------
        dname : `str`
        Directory name specified in the configuration file,
        or the default is used

        Returns
        -------
        None
        """
        self.current_dir = pathlib.Path.cwd()
        self.start_time = datetime.datetime.now()
        self.date_stamp = self.year_plus_fraction()

        # Read from input json
        self.parameters_json = self._read_json()
        self._read_user_params()

        # Directories
        model_dir = f"{dname}__{self.date_stamp}"
        self.work_subfolder = pathlib.Path(self.project_folder) / model_dir
        self.temp_folder = (
            pathlib.Path(self.work_folder) / f"temp_folder__{self.date_stamp}"
        )

        if self.runid:
            self.work_subfolder = pathlib.Path(f"{self.work_subfolder}_{self.runid}")
            self.temp_folder = pathlib.Path(f"{self.temp_folder}_{self.runid}")

        # Various model parameters, not in config
        self.num_lyr_lut = self.cube_shape[2] * 2 * self.infill_factor
        # 2500 voxels = 25x25x4m voxels size, 25% porosity and closures > ~40,000 bbl
        # Use the minimum voxel count as initial closure size filter
        self.closure_min_voxels = min(
            self.closure_min_voxels_simple,
            self.closure_min_voxels_faulted,
            self.closure_min_voxels_onlap,
        )
        self.order = self.bandwidth_ord

        if self.test_mode:
            self._set_test_mode(self.test_mode, self.test_mode)

        # Random choices are separated into this method
        self._randomly_chosen_model_parameters()
        # Fault choices
        self._fault_settings()

        # Logfile
        self.logfile = (
            pathlib.Path(self.work_subfolder)
            / f"model_parameters_{self.date_stamp}.txt"
        )

        # HDF file to store various model data
        self.hdf_master = (
            pathlib.Path(self.work_subfolder) / f"seismicCube__{self.date_stamp}.hdf"
        )

    def _calculate_snr_after_lateral_filter(self, sn_db: float) -> float:
        """
        Calculate Signal:Noise Ratio after lateral filter
        ----------------------------------------

        Method that computes the signal to noise ratio after
        the lateral filter is applied.

        Parameters
        ----------
        sn_db : `float`
            Value of the signal to noise value from the database

        Returns
        -------
        pre_smear_snr : `float`
            Signal to noise ratio after the lateral filter is applied
        """
        snr_of_lateral_filter = 10 * np.log10(self.lateral_filter_size**2)
        pre_smear_snr = sn_db - snr_of_lateral_filter
        return pre_smear_snr

    def _randomly_chosen_model_parameters(self) -> None:
        """
        Randomly Chosen Model Parameters
        ----------------------------------------

        Method that sets all randomly chosen model parameters

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Initial layer standard deviation
        self.initial_layer_stdev = (
            np.random.uniform(self.lyr_stdev[0], high=self.lyr_stdev[1])
            * self.infill_factor
        )

        # lateral filter size, either 1x1, 3x3 or 5x5
        self.lateral_filter_size = int(np.random.uniform(0, 2) + 0.5) * 2 + 1

        # Signal to noise in decibels
        sn_db = triangle_distribution_fix(
            left=self.snr_db[0], mode=self.snr_db[1], right=self.snr_db[2]
        )
        # sn_db = np.random.triangular(left=self.snr_db[0], mode=self.snr_db[1], right=self.snr_db[2])
        # self.sn_db = self._calculate_snr_after_lateral_filter(sn_db)
        self.sn_db = sn_db

        # Percentage of layers that are sand
        self.sand_layer_pct = np.random.uniform(
            low=self.sand_layer_pct_min, high=self.sand_layer_pct_max
        )

        # Minimum shallowest depth of seabed
        if (
            len(self.seabed_min_depth) > 1
        ):  # if low/high value provided, select a value between these
            self.seabed_min_depth = np.random.randint(
                low=self.seabed_min_depth[0], high=self.seabed_min_depth[1]
            )

        # Low/High bandwidth to be used
        self.lowfreq = np.random.uniform(self.bandwidth_low[0], self.bandwidth_low[1])
        self.highfreq = np.random.uniform(
            self.bandwidth_high[0], self.bandwidth_high[1]
        )

        # Choose whether to add coherent noise
        self.add_noise = np.random.choice((0, 1))
        if self.add_noise == 1:
            self.smiley_or_frowny = np.random.choice((0, 1))
            if self.smiley_or_frowny == 1:
                self.fnoise = "random_coherent_frowns"
                print("Coherent frowns will be inserted")
            else:
                self.fnoise = "random_coherent_smiles"
                print("Coherent smiles will be inserted")
        else:
            self.fnoise = "random"
            print("No coherent noise will be inserted")

        # Salt inclusion
        # self.include_salt = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        self.noise_stretch_factor = np.random.uniform(1.15, 1.35)
        if self.include_salt:
            print(
                "Salt will be inserted. noise_stretch_factor = {}".format(
                    np.around(self.noise_stretch_factor, 2)
                )
            )
        else:
            print("Salt will be NOT be inserted.")

    def _read_json(self) -> dict:
        # TODO Move this to a separate function in utlis?
        """
        Read JSON file
        ----------------------------------------

        Reads a json file on disk and loads it as
        dictionary

        Parameters
        ----------
        None

        Returns
        -------
        config : `dict`
            Dictionary with the configuration options
        """
        with open(self.parameter_file) as f:
            config: dict = json.load(f)
        return config

    def _read_user_params(self) -> None:
        """
        Read User Params
        ----------------------------------------

        Takes the read in dictionary of JSON configuration
        and reads each parameter and inserts it into the
        attributes.

        In the end it prints a summary of the parameters
        to the console.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        d = self._read_json()
        self.project = d["project"]
        self.project_folder = d["project_folder"]
        wfolder = d["work_folder"]
        if not os.path.exists(wfolder):
            wfolder = "/tmp"  # In case work_folder does not exist, use /tmp
        self.work_folder = wfolder
        # read parameters into Parameter class attributes
        self.cube_shape = tuple(d["cube_shape"])
        self.incident_angles = tuple(d["incident_angles"])
        self.digi = d["digi"]
        self.infill_factor = d["infill_factor"]
        self.lyr_stdev = d["initial_layer_stdev"]
        self.thickness_min = d["thickness_min"]
        self.thickness_max = d["thickness_max"]
        self.seabed_min_depth = d["seabed_min_depth"]
        self.snr_db = d["signal_to_noise_ratio_db"]
        # self.random_depth_perturb = d['random_depth_perturb_range']
        self.bandwidth_low = d["bandwidth_low"]
        self.bandwidth_high = d["bandwidth_high"]
        self.bandwidth_ord = d["bandwidth_ord"]
        self.dip_factor_max = d["dip_factor_max"]
        self.min_number_faults = d["min_number_faults"]
        self.max_number_faults = d["max_number_faults"]
        self.basin_floor_fans = d["basin_floor_fans"]
        self.pad_samples = d["pad_samples"]
        self.qc_plots = d["extra_qc_plots"]
        self.verbose = d["verbose"]
        self.include_channels = d["include_channels"]
        self.include_salt = d["include_salt"]
        self.max_column_height = d["max_column_height"]
        self.closure_types = d["closure_types"]
        self.closure_min_voxels_simple = d["min_closure_voxels_simple"]
        self.closure_min_voxels_faulted = d["min_closure_voxels_faulted"]
        self.closure_min_voxels_onlap = d["min_closure_voxels_onlap"]
        self.partial_voxels = d["partial_voxels"]
        self.variable_shale_ng = d["variable_shale_ng"]
        self.sand_layer_thickness = d["sand_layer_thickness"]
        self.sand_layer_pct_min = d["sand_layer_fraction"]["min"]
        self.sand_layer_pct_max = d["sand_layer_fraction"]["max"]
        self.hdf_store = d["write_to_hdf"]
        self.broadband_qc_volume = d["broadband_qc_volume"]
        self.model_qc_volumes = d["model_qc_volumes"]
        self.multiprocess_bp = d["multiprocess_bp"]

        # print em
        self.__repr__()

    def _set_test_mode(self, size_x: int = 50, size_y: int = 50) -> None:
        """
        Set test mode
        -------------

        Sets whether the parameters for testing mode. If no size integer
        is provided is defaults to 50.

        This value is a good minimum because it allows for the 3D model
        to be able to contain faults and other objects inside.

        Parameters
        ----------
        size_x : `int`
        The parameter that sets the size of the model in the x direction
        size_y : `int`
        The parameter that sets the size of the model in the y direction

        Returns
        -------
        None
        """
        # Set output model folder in work_folder location but with same directory name as project_folder
        normpath = pathlib.Path(self.project_folder).name + "_test_mode_"
        new_project_folder = pathlib.Path(self.work_folder) / normpath

        # Put all folders inside project folder for easy deleting
        self.work_folder = str(new_project_folder)
        self.project_folder = str(new_project_folder)
        self.work_subfolder = (
            new_project_folder / pathlib.Path(self.work_subfolder).name
        )

        if self.runid:
            # Append runid if provided
            self.temp_folder = pathlib.Path(
                f"{self.temp_folder}_{self.runid}__{self.date_stamp}"
            )
        else:
            self.temp_folder = (
                pathlib.Path(self.work_folder) / f"temp_folder__{self.date_stamp}"
            )

        # Set smaller sized model
        self.cube_shape = tuple([size_x, size_y, self.cube_shape[-1]])
        # Print message to user
        print(
            "{0}\nTesting Mode\nOutput Folder: {1}\nCube_Shape: {2}\n{0}".format(
                36 * "-", self.project_folder, self.cube_shape
            )
        )

    def _fault_settings(self) -> None:
        """
        Set Fault Settings
        -------------

        Sets the parameters that will be used to generate faults throughout
        the synthetic model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Fault parameters
        self.low_fault_throw = 5.0 * self.infill_factor
        self.high_fault_throw = 35.0 * self.infill_factor

        # mode & clustering are randomly chosen
        self.mode = np.random.choice([0, 1, 2], 1)[0]
        self.clustering = np.random.choice([0, 1, 2], 1)[0]

        if self.mode == 0:
            # As random as it can be
            self.number_faults = np.random.randint(
                self.min_number_faults, self.max_number_faults
            )
            self.fmode = "random"

        elif self.mode == 1:
            if self.clustering == 0:
                self.fmode = "self_branching"
                # Self Branching. avoid large fault
                self.number_faults = np.random.randint(3, 9)
                self.low_fault_throw = 5.0 * self.infill_factor
                self.high_fault_throw = 15.0 * self.infill_factor
            if self.clustering == 1:
                # Stair case
                self.fmode = "stair_case"
                self.number_faults = np.random.randint(5, self.max_number_faults)
            if self.clustering == 2:
                # Relay ramps
                self.fmode = "relay_ramp"
                self.number_faults = np.random.randint(3, 9)
                self.low_fault_throw = 5.0 * self.infill_factor
                self.high_fault_throw = 15.0 * self.infill_factor
        elif self.mode == 2:
            # Horst and graben
            self.fmode = "horst_and_graben"
            self.number_faults = np.random.randint(3, 7)

        self.fault_param = [
            str(self.mode) + str(self.clustering),
            self.number_faults,
            self.low_fault_throw,
            self.high_fault_throw,
        ]

    def _get_commit_hash(self) -> str:
        """
        Get Commit Hash
        -------------

        Gets the commit hash of the current git repository.

        #TODO Explain what this is for exactly

        Parameters
        ----------
        None

        Returns
        -------
        sha : `str`
            The commit hash of the current git repository
        """

        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
        except CalledProcessError:
            sha = "cwd not a git repository"
        return sha

    def _write_initial_model_parameters_to_logfile(self) -> None:
        """
        Write Initial Model Parameters to Logfile
        ----------------------------------------

        Method that writes the initial parameters set for the model
        to the logfile.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _sha = self._get_commit_hash()
        self.write_to_logfile(
            f"SHA: {_sha}", mainkey="model_parameters", subkey="sha", val=_sha
        )
        self.write_to_logfile(
            f"modeling start time: {self.start_time}",
            mainkey="model_parameters",
            subkey="start_time",
            val=self.start_time,
        )
        self.write_to_logfile(
            f"project_folder: {self.project_folder}",
            mainkey="model_parameters",
            subkey="project_folder",
            val=self.project_folder,
        )
        self.write_to_logfile(
            f"work_subfolder: {self.work_subfolder}",
            mainkey="model_parameters",
            subkey="work_subfolder",
            val=self.work_subfolder,
        )
        self.write_to_logfile(
            f"cube_shape: {self.cube_shape}",
            mainkey="model_parameters",
            subkey="cube_shape",
            val=self.cube_shape,
        )
        self.write_to_logfile(
            f"incident_angles: {self.incident_angles}",
            mainkey="model_parameters",
            subkey="incident_angles",
            val=self.incident_angles,
        )
        self.write_to_logfile(
            f"number_faults: {self.number_faults}",
            mainkey="model_parameters",
            subkey="number_faults",
            val=self.number_faults,
        )
        self.write_to_logfile(
            f"lateral_filter_size: {self.lateral_filter_size}",
            mainkey="model_parameters",
            subkey="lateral_filter_size",
            val=self.lateral_filter_size,
        )
        self.write_to_logfile(
            f"salt_inserted: {self.include_salt}",
            mainkey="model_parameters",
            subkey="salt_inserted",
            val=self.include_salt,
        )
        self.write_to_logfile(
            f"salt noise_stretch_factor: {self.noise_stretch_factor:.2f}",
            mainkey="model_parameters",
            subkey="salt_noise_stretch_factor",
            val=self.noise_stretch_factor,
        )
        self.write_to_logfile(
            f"bandpass_bandlimits: {self.lowfreq:.2f}, {self.highfreq:.2f}"
        )
        self.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="bandpass_bandlimit_low",
            val=self.lowfreq,
        )
        self.write_to_logfile(
            msg=None,
            mainkey="model_parameters",
            subkey="bandpass_bandlimit_high",
            val=self.highfreq,
        )
        self.write_to_logfile(
            f"sn_db: {self.sn_db:.2f}",
            mainkey="model_parameters",
            subkey="sn_db",
            val=self.sn_db,
        )
        self.write_to_logfile(
            f"initial layer depth stdev (flatness of layer): {self.initial_layer_stdev:.2f}",
            mainkey="model_parameters",
            subkey="initial_layer_stdev",
            val=self.initial_layer_stdev,
        )

    @staticmethod
    def year_plus_fraction() -> str:
        # TODO Move this to utils separate module
        """
        Year Plus Fraction
        ----------------------------------------

        Method generates a time stamp in the format of
        year + fraction of year.

        Parameters
        ----------
        None

        Returns
        -------
        fraction of the year : str
            The time stamp in the format of year + fraction of year

        """
        now = datetime.datetime.now()
        year = now.year
        secs_in_year = datetime.timedelta(days=365).total_seconds()
        fraction_of_year = (
            now - datetime.datetime(year, 1, 1, 0, 0)
        ).total_seconds() / secs_in_year
        return format(year + fraction_of_year, "14.8f").replace(" ", "")

    def zarr_setup(self, zarr_name: str) -> None:
        """
        Setup Zarr storage
        ---------------

        This method sets up the Zarr storage structures

        Parameters
        ----------
        zarr_name : str
            The name of the Zarr store to be created

        Returns
        -------
        None
        """
        num_threads = min(8, mp.cpu_count() - 1)
        self.zarr_filename = pathlib.Path(self.temp_folder) / zarr_name
        # Configure compression similar to HDF5 setup
        self.compressor = numcodecs.Blosc(
            cname="blosclz",
            clevel=5,
            shuffle=numcodecs.Blosc.SHUFFLE,
        )
        # Create root group
        self.zarr_store = zarr.open_group(store=str(self.zarr_filename), mode="w")
        # Create ModelData group
        self.zarr_group = self.zarr_store.create_group("ModelData")

    def zarr_init(
        self, dset_name: str, shape: tuple, dtype: str = "float16"
    ) -> zarr.Array:
        """
        Zarr Initialize
        ----------------------------------------

        Method that initializes a Zarr array

        Parameters
        ----------
        dset_name : str
            The name of the dataset to be created
        shape : tuple
            Shape of the array
        dtype : str, optional
            Data type of the array, by default "float64"

        Returns
        -------
        zarr.Array
            The created Zarr array
        """
        return self.zarr_group.create_dataset(
            name=dset_name, shape=shape, dtype=dtype, compressor=self.compressor
        )

    def zarr_node_list(self):
        """Get list of arrays in the ModelData group"""
        return list(self.zarr_group.keys())

    def zarr_remove_node(self, dset_name: str):
        """Remove an array from the ModelData group"""
        try:
            del self.zarr_group[dset_name]
        except KeyError:
            pass

    # Replace old HDF methods with Zarr equivalents for backward compatibility
    hdf_setup = zarr_setup
    hdf_init = zarr_init
    hdf_node_list = zarr_node_list
    hdf_remove_node_list = zarr_remove_node


def triangle_distribution_fix(left, mode, right, random_seed=None):
    """
    Triangle Distribution Fix
    -------------------------

    Draw samples from the triangular distribution over the interval [left, right] with modifications.

    Ensure some values are drawn at the left and right values by enlarging the interval to
    [left - (mode - left), right + (right - mode)]

    Parameters
    ----------
    left: `float`
        lower limit
    mode: `float`
        mode
    right: `float`
        upper limit
    random_seed: `int`
        seed to set numpy's random seed

    Returns
    -------
    sn_db: `float`
        Drawn samples from parameterised triangular distribution
    """
    sn_db = 0
    while sn_db < left or sn_db > right:
        if random_seed:
            np.random.seed(random_seed)
        sn_db = np.random.triangular(left - (mode - left), mode, right + (right - mode))

    return sn_db
