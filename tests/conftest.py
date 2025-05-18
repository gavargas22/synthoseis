"""Common test configurations and fixtures."""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    np.random.seed(None)  # Reset seed after test


@pytest.fixture
def rpm_config_ranges():
    """Fixture providing valid ranges for RPM parameters."""
    return {
        "layershiftsamples": (35, 75, 125),  # (min, mode, max)
        "RPshiftsamples": (5, 11, 20),  # (min, mode, max)
        "factors": (0.1, 1.0, 2.0),  # (min, default, max)
    }


@pytest.fixture
def sample_rpm_json():
    """Fixture providing a sample RPM configuration in JSON format."""
    return {
        "layershiftsamples": 75,
        "RPshiftsamples": 11,
        "shalerho_factor": 1.2,
        "shalevp_factor": 1.1,
        "shalevs_factor": 1.0,
        "sandrho_factor": 1.0,
        "sandvp_factor": 1.1,
        "sandvs_factor": 1.2,
        "nearfactor": 1.0,
        "midfactor": 1.0,
        "farfactor": 1.0,
    }
