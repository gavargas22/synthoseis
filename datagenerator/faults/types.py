from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class GeomodelProtocol(Protocol):
    """Protocol defining the required interface for Geomodel."""

    geologic_age_store: np.ndarray
    property_store: np.ndarray
    shape: tuple[int, int, int]

    def get_property(self, name: str) -> np.ndarray:
        """Get a property from the geomodel."""
        ...

    def set_property(self, name: str, value: np.ndarray) -> None:
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
DepthMap = np.ndarray  # Shape: (nx, ny, nz)
FaultTrace = np.ndarray  # Shape: (nx, ny)
DisplacementVector = tuple[np.ndarray, np.ndarray, np.ndarray]  # (dx, dy, dz)
PropertyModel = np.ndarray  # Shape: (nx, ny, nz)
