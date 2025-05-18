"""Rock Physics Model configuration and randomization."""

from typing import Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field


class RPMConfig(BaseModel):
    """Configuration for Rock Physics Model parameters."""

    # Layer shift parameters
    layershiftsamples: int = Field(
        ..., gt=0, description="Number of samples to shift layers"
    )
    RPshiftsamples: int = Field(
        ..., gt=0, description="Number of samples to shift rock properties"
    )

    # Shale factors
    shalerho_factor: float = Field(
        1.0, gt=0, description="Shale density scaling factor"
    )
    shalevp_factor: float = Field(
        1.0, gt=0, description="Shale P-wave velocity scaling factor"
    )
    shalevs_factor: float = Field(
        1.0, gt=0, description="Shale S-wave velocity scaling factor"
    )

    # Sand factors
    sandrho_factor: float = Field(1.0, gt=0, description="Sand density scaling factor")
    sandvp_factor: float = Field(
        1.0, gt=0, description="Sand P-wave velocity scaling factor"
    )
    sandvs_factor: float = Field(
        1.0, gt=0, description="Sand S-wave velocity scaling factor"
    )

    # Amplitude scaling factors
    nearfactor: float = Field(1.0, gt=0, description="Near angle stack scaling factor")
    midfactor: float = Field(1.0, gt=0, description="Mid angle stack scaling factor")
    farfactor: float = Field(1.0, gt=0, description="Far angle stack scaling factor")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RPMConfig":
        """Create RPMConfig from a dictionary."""
        return cls(**config)

    @classmethod
    def create_random(cls) -> "RPMConfig":
        """Create RPMConfig with randomized layer shift parameters."""
        return cls(
            layershiftsamples=int(np.random.triangular(35, 75, 125)),
            RPshiftsamples=int(np.random.triangular(5, 11, 20)),
            # Default factors remain at 1.0
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
