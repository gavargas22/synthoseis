import os
import json
import tempfile
import pytest
from pathlib import Path
import numpy as np
from datagenerator.Parameters import (
    Parameters,
    ModelConfig,
    SandLayerFraction,
    RPMScalingFactors,
    triangle_distribution_fix,
)

# Test data
VALID_CONFIG = {
    "project": "test_project",
    "project_folder": "test_output",
    "work_folder": "test_work",
    "cube_shape": [100, 100, 50],
    "incident_angles": [10.0, 20.0, 30.0],
    "digi": 4,
    "infill_factor": 2,
    "initial_layer_stdev": [0.1, 0.2],
    "thickness_min": 5,
    "thickness_max": 15,
    "seabed_min_depth": 100,
    "signal_to_noise_ratio_db": [10.0, 15.0, 20.0],
    "bandwidth_low": [5.0, 10.0],
    "bandwidth_high": [15.0, 20.0],
    "bandwidth_ord": 2,
    "dip_factor_max": 1.5,
    "min_number_faults": 2,
    "max_number_faults": 5,
    "max_column_height": [100.0, 200.0],
    "closure_types": ["simple", "faulted"],
    "min_closure_voxels_simple": 1000,
    "min_closure_voxels_faulted": 800,
    "min_closure_voxels_onlap": 600,
    "sand_layer_thickness": 10,
    "sand_layer_fraction": {"min": 0.2, "max": 0.4},
    "extra_qc_plots": False,
    "verbose": False,
    "partial_voxels": True,
    "variable_shale_ng": False,
    "basin_floor_fans": False,
    "include_channels": False,
    "include_salt": False,
    "write_to_hdf": False,
    "broadband_qc_volume": False,
    "model_qc_volumes": True,
    "multiprocess_bp": True,
    "pad_samples": 10,
}


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(VALID_CONFIG, f)
        return f.name


@pytest.fixture
def temp_work_dir():
    """Create a temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def params(temp_config_file, temp_work_dir):
    """Create a Parameters instance with temporary config and work directory."""
    # Update config paths to use temp directory
    config = VALID_CONFIG.copy()
    config["project_folder"] = temp_work_dir
    config["work_folder"] = temp_work_dir

    with open(temp_config_file, "w") as f:
        json.dump(config, f)

    return Parameters(user_config=temp_config_file)


class TestSandLayerFraction:
    """Test suite for SandLayerFraction model."""

    def test_valid_fraction(self):
        """Test valid sand layer fraction values."""
        fraction = SandLayerFraction(min=0.2, max=0.4)
        assert fraction.min == 0.2
        assert fraction.max == 0.4

    def test_invalid_fraction_range(self):
        """Test invalid fraction range (max < min)."""
        with pytest.raises(ValueError, match="max must be greater than min"):
            SandLayerFraction(min=0.4, max=0.2)

    def test_invalid_fraction_values(self):
        """Test invalid fraction values outside [0,1] range."""
        with pytest.raises(ValueError):
            SandLayerFraction(min=-0.1, max=0.4)
        with pytest.raises(ValueError):
            SandLayerFraction(min=0.2, max=1.1)


class TestModelConfig:
    """Test suite for ModelConfig model."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ModelConfig(**VALID_CONFIG)
        assert config.project == "test_project"
        assert config.cube_shape == (100, 100, 50)
        assert config.incident_angles == (10.0, 20.0, 30.0)

    def test_invalid_fault_range(self):
        """Test invalid fault range (max < min)."""
        config = VALID_CONFIG.copy()
        config["max_number_faults"] = 1
        with pytest.raises(
            ValueError, match="max_number_faults must be greater than min_number_faults"
        ):
            ModelConfig(**config)

    def test_invalid_thickness_range(self):
        """Test invalid thickness range (max < min)."""
        config = VALID_CONFIG.copy()
        config["thickness_max"] = 3
        with pytest.raises(
            ValueError, match="thickness_max must be greater than thickness_min"
        ):
            ModelConfig(**config)

    def test_invalid_bandwidth_range(self):
        """Test invalid bandwidth range (high < low)."""
        config = VALID_CONFIG.copy()
        config["bandwidth_high"] = [4.0, 5.0]  # Both values less than bandwidth_low[1]
        with pytest.raises(
            ValueError, match="bandwidth_high must be greater than bandwidth_low"
        ):
            ModelConfig(**config)

    def test_invalid_snr_order(self):
        """Test invalid signal-to-noise ratio order."""
        config = VALID_CONFIG.copy()
        config["signal_to_noise_ratio_db"] = [20.0, 15.0, 10.0]  # Not ascending
        with pytest.raises(
            ValueError, match="signal_to_noise_ratio_db must be in ascending order"
        ):
            ModelConfig(**config)


class TestRPMScalingFactors:
    """Test suite for RPMScalingFactors model."""

    def test_valid_factors(self):
        """Test valid RPM scaling factors."""
        factors = RPMScalingFactors(layershiftsamples=50, RPshiftsamples=10)
        assert factors.layershiftsamples == 50
        assert factors.RPshiftsamples == 10
        assert factors.shalerho_factor == 1.0  # Default value

    def test_invalid_factors(self):
        """Test invalid RPM scaling factors."""
        with pytest.raises(ValueError):
            RPMScalingFactors(layershiftsamples=-1, RPshiftsamples=10)
        with pytest.raises(ValueError):
            RPMScalingFactors(layershiftsamples=50, RPshiftsamples=0)


class TestParameters:
    """Test suite for Parameters class."""

    def test_initialization(self, params):
        """Test basic initialization of Parameters."""
        assert params.project == "test_project"
        assert params.cube_shape == (100, 100, 50)
        assert params.incident_angles == (10.0, 20.0, 30.0)

    def test_setup_model(self, params):
        """Test model setup with default RPM factors."""
        params.setup_model()
        assert os.path.exists(params.work_subfolder)
        assert os.path.exists(params.temp_folder)
        assert os.path.exists(params.logfile)

    def test_setup_model_with_rpm_factors(self, params):
        """Test model setup with custom RPM factors."""
        rpm_factors = {
            "layershiftsamples": 75,
            "RPshiftsamples": 15,
            "shalerho_factor": 1.2,
        }
        params.setup_model(rpm_factors=rpm_factors)
        assert params.rpm_scaling_factors["layershiftsamples"] == 75
        assert params.rpm_scaling_factors["RPshiftsamples"] == 15
        assert params.rpm_scaling_factors["shalerho_factor"] == 1.2

    def test_test_mode(self, temp_config_file, temp_work_dir):
        """Test test mode initialization."""
        params = Parameters(user_config=temp_config_file, test_mode=50, runid="test")
        params.setup_model()
        assert params.cube_shape == (50, 50, 50)  # Last dimension preserved
        assert "test_mode" in params.work_subfolder
        assert "test" in params.work_subfolder  # runid included

    def test_write_key_file(self, params):
        """Test key file generation."""
        params.setup_model()
        params.write_key_file()
        key_file = os.path.join(
            params.work_subfolder, f"seismicCube_{params.date_stamp}.key"
        )
        assert os.path.exists(key_file)

        # Verify key file contents
        with open(key_file, "r") as f:
            content = f.read()
            assert "3D_NAME" in content
            assert "CUBE_SHAPE" in content
            assert str(params.cube_shape[0]) in content

    def test_write_to_logfile(self, params):
        """Test logfile writing."""
        params.setup_model()
        test_msg = "Test log message"
        params.write_to_logfile(
            msg=test_msg, mainkey="test_section", subkey="test_key", val="test_value"
        )

        # Verify logfile contents
        with open(params.logfile, "r") as f:
            content = f.read()
            assert test_msg in content
            assert params.sqldict["test_section"]["test_key"] == "test_value"

    def test_write_sqldict_to_db(self, params):
        """Test SQLite database writing."""
        params.setup_model()
        params.write_to_logfile(
            msg="Test model parameter",
            mainkey="model_parameters",
            subkey="test_param",
            val="test_value",
        )
        params.write_sqldict_to_db()

        # Verify database contents
        db_path = os.path.join(params.work_subfolder, "parameters.db")
        assert os.path.exists(db_path)

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT test_param FROM model_parameters")
        result = cursor.fetchone()
        assert result[0] == "test_value"
        conn.close()


def test_triangle_distribution_fix():
    """Test the triangle distribution fix function."""
    # Test with valid parameters
    result = triangle_distribution_fix(left=10, mode=15, right=20)
    assert 10 <= result <= 20

    # Test with fixed random seed
    result1 = triangle_distribution_fix(left=10, mode=15, right=20, random_seed=42)
    result2 = triangle_distribution_fix(left=10, mode=15, right=20, random_seed=42)
    assert result1 == result2  # Should be deterministic with same seed

    # Test edge cases
    result = triangle_distribution_fix(left=10, mode=10, right=10)
    assert result == 10  # Should return exact value when all parameters are equal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
