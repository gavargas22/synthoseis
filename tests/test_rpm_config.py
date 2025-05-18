"""Test suite for RPM configuration."""

import pytest
import numpy as np
from datagenerator.RPMConfig import RPMConfig


@pytest.fixture
def valid_rpm_dict():
    """Fixture providing a valid RPM configuration dictionary."""
    return {
        "layershiftsamples": 50,
        "RPshiftsamples": 10,
        "shalerho_factor": 1.0,
        "shalevp_factor": 1.0,
        "shalevs_factor": 1.0,
        "sandrho_factor": 1.0,
        "sandvp_factor": 1.0,
        "sandvs_factor": 1.0,
        "nearfactor": 1.0,
        "midfactor": 1.0,
        "farfactor": 1.0,
    }


@pytest.fixture
def invalid_rpm_dict():
    """Fixture providing an invalid RPM configuration dictionary."""
    return {
        "layershiftsamples": -1,  # Invalid: must be > 0
        "RPshiftsamples": 0,  # Invalid: must be > 0
        "shalerho_factor": -0.5,  # Invalid: must be > 0
        "shalevp_factor": 0.0,  # Invalid: must be > 0
        "shalevs_factor": 1.0,
        "sandrho_factor": 1.0,
        "sandvp_factor": 1.0,
        "sandvs_factor": 1.0,
        "nearfactor": 1.0,
        "midfactor": 1.0,
        "farfactor": 1.0,
    }


def test_create_from_valid_dict(valid_rpm_dict):
    """Test creating RPMConfig from a valid dictionary."""
    config = RPMConfig.from_dict(valid_rpm_dict)
    assert isinstance(config, RPMConfig)
    assert config.layershiftsamples == 50
    assert config.RPshiftsamples == 10
    assert all(
        getattr(config, f"{factor}_factor") == 1.0
        for factor in ["shalerho", "shalevp", "shalevs", "sandrho", "sandvp", "sandvs"]
    )
    assert all(
        getattr(config, f"{angle}factor") == 1.0 for angle in ["near", "mid", "far"]
    )


def test_create_from_invalid_dict(invalid_rpm_dict):
    """Test that creating RPMConfig from invalid dictionary raises error."""
    with pytest.raises(ValueError):
        RPMConfig.from_dict(invalid_rpm_dict)


def test_create_random():
    """Test creating random RPM configuration."""
    config = RPMConfig.create_random()
    assert isinstance(config, RPMConfig)

    # Check that random values are within expected ranges
    assert 35 <= config.layershiftsamples <= 125
    assert 5 <= config.RPshiftsamples <= 20

    # Check that all factors are positive
    assert all(
        getattr(config, f"{factor}_factor") > 0
        for factor in ["shalerho", "shalevp", "shalevs", "sandrho", "sandvp", "sandvs"]
    )
    assert all(
        getattr(config, f"{angle}factor") > 0 for angle in ["near", "mid", "far"]
    )


def test_to_dict(valid_rpm_dict):
    """Test converting RPMConfig to dictionary."""
    config = RPMConfig.from_dict(valid_rpm_dict)
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict == valid_rpm_dict


def test_validation_rules():
    """Test validation rules for RPM parameters."""
    # Test minimum value validation
    with pytest.raises(ValueError):
        RPMConfig(
            layershiftsamples=0,  # Must be > 0
            RPshiftsamples=10,
            shalerho_factor=1.0,
            shalevp_factor=1.0,
            shalevs_factor=1.0,
            sandrho_factor=1.0,
            sandvp_factor=1.0,
            sandvs_factor=1.0,
            nearfactor=1.0,
            midfactor=1.0,
            farfactor=1.0,
        )

    # Test negative factor validation
    with pytest.raises(ValueError):
        RPMConfig(
            layershiftsamples=50,
            RPshiftsamples=10,
            shalerho_factor=-1.0,  # Must be > 0
            shalevp_factor=1.0,
            shalevs_factor=1.0,
            sandrho_factor=1.0,
            sandvp_factor=1.0,
            sandvs_factor=1.0,
            nearfactor=1.0,
            midfactor=1.0,
            farfactor=1.0,
        )


def test_random_distribution():
    """Test that random values follow expected distribution."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate multiple random configurations
    configs = [RPMConfig.create_random() for _ in range(1000)]

    # Extract values
    layer_shifts = [c.layershiftsamples for c in configs]
    rp_shifts = [c.RPshiftsamples for c in configs]

    # Check mean values are within expected ranges
    assert 70 <= np.mean(layer_shifts) <= 80  # Expected mean around 75
    assert 10 <= np.mean(rp_shifts) <= 12  # Expected mean around 11

    # Check that all values are within bounds
    assert all(35 <= x <= 125 for x in layer_shifts)
    assert all(5 <= x <= 20 for x in rp_shifts)


def test_immutability():
    """Test that RPMConfig instances are immutable."""
    config = RPMConfig.create_random()

    # Attempting to modify attributes should raise error
    with pytest.raises(ValueError):
        config.layershiftsamples = 100

    with pytest.raises(ValueError):
        config.shalevp_factor = 2.0


def test_missing_required_fields():
    """Test that creating RPMConfig without required fields raises error."""
    with pytest.raises(ValueError):
        RPMConfig(
            layershiftsamples=50,
            # Missing RPshiftsamples
            shalerho_factor=1.0,
            shalevp_factor=1.0,
            shalevs_factor=1.0,
            sandrho_factor=1.0,
            sandvp_factor=1.0,
            sandvs_factor=1.0,
            nearfactor=1.0,
            midfactor=1.0,
            farfactor=1.0,
        )
