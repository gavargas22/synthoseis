from typing import Protocol, TypeAlias, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class GeomodelProtocol(Protocol):
    """Protocol defining the required interface for Geomodel."""

    geologic_age_store: np.ndarray
    property_store: dict[str, np.ndarray]
    shape: tuple[int, int, int]

    def get_property(self, property_name: str) -> np.ndarray:
        """Get a property from the geomodel."""
        ...

    def set_property(self, property_name: str, property_data: np.ndarray) -> None:
        """Set a property in the geomodel."""
        ...


class FaultError(Exception):
    """Base exception for fault-related errors."""

    pass


class FaultGenerationError(FaultError):
    """Raised when fault generation fails."""

    pass


class FaultGeometryError(FaultError):
    """Raised when fault geometry calculations fail."""

    pass


class FaultQCError(FaultError):
    """Raised when quality control checks fail."""

    pass


# Type aliases for common numpy array shapes
DepthMap: TypeAlias = np.ndarray  # Shape: (ny, nx)
FaultTrace: TypeAlias = np.ndarray  # Shape: (n_points, 2)
DisplacementVector: TypeAlias = Tuple[
    np.ndarray, np.ndarray, np.ndarray
]  # Shape: ((ny, nx), (ny, nx), (ny, nx))
PropertyModel: TypeAlias = np.ndarray  # Shape: (ny, nx)
