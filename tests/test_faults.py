import pytest
import numpy as np
from datagenerator.faults import (
    Faults,
    FaultParameters,
    FaultGeometryParameters,
    FaultQCParameters,
    FaultError,
    FaultGenerationError,
)
from datagenerator.faults.types import GeomodelProtocol


class MockGeomodel(GeomodelProtocol):
    """Mock geomodel for testing."""

    def __init__(self, shape: tuple[int, int, int]):
        self.shape = shape
        self.geologic_age_store = np.zeros(shape)
        self.property_store = np.zeros(shape)

    def get_property(self, name: str) -> np.ndarray:
        return np.zeros(self.shape)

    def set_property(self, name: str, value: np.ndarray) -> None:
        pass


@pytest.fixture
def fault_parameters():
    """Create test fault parameters."""
    return FaultParameters(
        mode="random",
        throw=100.0,
        dip=45.0,
        strike=90.0,
        length=1000.0,
        width=500.0,
        depth=2000.0,
        sigma=1.0,
        tilt=0.0,
    )


@pytest.fixture
def mock_geomodel():
    """Create a mock geomodel for testing."""
    return MockGeomodel(shape=(100, 100, 50))


@pytest.fixture
def mock_depth_maps():
    """Create mock depth maps for testing."""
    return np.zeros((100, 100, 50))


@pytest.fixture
def mock_onlap_list():
    """Create mock onlap list for testing."""
    return np.zeros((100, 100))


def test_faults_initialization(
    fault_parameters: FaultParameters,
    mock_geomodel: MockGeomodel,
    mock_depth_maps: np.ndarray,
    mock_onlap_list: np.ndarray,
):
    """Test Faults class initialization."""
    faults = Faults(
        parameters=fault_parameters,
        unfaulted_depth_maps=mock_depth_maps,
        onlap_horizon_list=mock_onlap_list,
        geomodels=mock_geomodel,
    )

    assert faults.parameters == fault_parameters
    assert faults.unfaulted_depth_maps.shape == mock_depth_maps.shape
    assert faults.onlap_horizon_list.shape == mock_onlap_list.shape
    assert faults.geomodels == mock_geomodel
    assert faults.faulted_depth_maps is None
    assert faults.faulted_geologic_age is None
    assert faults.fault_traces is None


def test_faults_validation(
    fault_parameters: FaultParameters,
    mock_geomodel: MockGeomodel,
    mock_depth_maps: np.ndarray,
    mock_onlap_list: np.ndarray,
):
    """Test input validation."""
    # Test shape mismatch
    with pytest.raises(FaultError):
        Faults(
            parameters=fault_parameters,
            unfaulted_depth_maps=np.zeros((50, 50, 25)),  # Wrong shape
            onlap_horizon_list=mock_onlap_list,
            geomodels=mock_geomodel,
        )

    # Test invalid parameters
    invalid_params = FaultParameters(
        mode="random",
        throw=-100.0,  # Invalid throw
        dip=45.0,
        strike=90.0,
        length=1000.0,
        width=500.0,
        depth=2000.0,
        sigma=1.0,
        tilt=0.0,
    )

    with pytest.raises(FaultError):
        Faults(
            parameters=invalid_params,
            unfaulted_depth_maps=mock_depth_maps,
            onlap_horizon_list=mock_onlap_list,
            geomodels=mock_geomodel,
        )


def test_faults_application(
    fault_parameters: FaultParameters,
    mock_geomodel: MockGeomodel,
    mock_depth_maps: np.ndarray,
    mock_onlap_list: np.ndarray,
):
    """Test fault application."""
    faults = Faults(
        parameters=fault_parameters,
        unfaulted_depth_maps=mock_depth_maps,
        onlap_horizon_list=mock_onlap_list,
        geomodels=mock_geomodel,
    )

    # Test that faulting hasn't been applied
    with pytest.raises(FaultError):
        faults.build_faulted_property_geomodels(np.zeros((100, 100, 50)))

    # Apply faulting
    faults.apply_faulting_to_geomodels_and_depth_maps()

    # Check that faulting was applied
    assert faults.faulted_depth_maps is not None
    assert faults.faulted_geologic_age is not None
    assert faults.fault_traces is not None
    assert faults.faulted_depth_maps.shape == mock_depth_maps.shape
    assert faults.faulted_geologic_age.shape == mock_geomodel.shape


def test_depth_map_improvement(
    fault_parameters: FaultParameters,
    mock_geomodel: MockGeomodel,
    mock_depth_maps: np.ndarray,
    mock_onlap_list: np.ndarray,
):
    """Test depth map improvement."""
    faults = Faults(
        parameters=fault_parameters,
        unfaulted_depth_maps=mock_depth_maps,
        onlap_horizon_list=mock_onlap_list,
        geomodels=mock_geomodel,
    )

    # Test that faulting hasn't been applied
    with pytest.raises(FaultError):
        faults.improve_depth_maps_post_faulting(
            mock_geomodel.geologic_age_store,
            mock_geomodel.geologic_age_store,
            mock_onlap_list,
        )

    # Apply faulting
    faults.apply_faulting_to_geomodels_and_depth_maps()

    # Improve depth maps
    improved_maps, improved_gaps = faults.improve_depth_maps_post_faulting(
        mock_geomodel.geologic_age_store,
        mock_geomodel.geologic_age_store,
        mock_onlap_list,
    )

    assert improved_maps.shape == mock_depth_maps.shape
    assert improved_gaps.shape == mock_depth_maps.shape
