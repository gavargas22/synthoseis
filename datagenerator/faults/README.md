# Fault System

A modular system for generating and applying faults to geological models.

## Overview

The fault system provides a flexible and maintainable way to generate and apply faults to geological models. It is designed to be:

- **Modular**: Each component has a single responsibility
- **Type-safe**: Full type hints and validation
- **Testable**: Components can be tested independently
- **Extensible**: New fault types can be added easily
- **Well-documented**: Clear documentation and examples

## Components

### FaultParameters

Manages fault generation parameters using dataclasses:

```python
from datagenerator.faults import FaultParameters

params = FaultParameters(
    mode="random",
    throw=100.0,
    dip=45.0,
    strike=90.0,
    length=1000.0,
    width=500.0,
    depth=2000.0,
    sigma=1.0,
    tilt=0.0
)
```

### FaultGeometry

Handles geometric calculations for faults:

```python
from datagenerator.faults import FaultGeometry, FaultGeometryParameters

geometry_params = FaultGeometryParameters(
    ellipsoid_axes=(1000.0, 500.0, 2000.0),
    origin=(500.0, 500.0, 1000.0),
    rotation_angles=(45.0, 90.0, 0.0),
    displacement_vector=(0.0, 0.0, 100.0)
)

geometry = FaultGeometry(geometry_params)
```

### FaultGenerator

Generates faults based on parameters:

```python
from datagenerator.faults import FaultGenerator

generator = FaultGenerator(params)
fault_traces = generator.generate_faults()
```

### FaultQC

Handles quality control and visualization:

```python
from datagenerator.faults import FaultQC, FaultQCParameters

qc_params = FaultQCParameters(
    plot_fault_planes=True,
    plot_displacement_vectors=True,
    plot_fault_traces=True
)

qc = FaultQC(qc_params)
qc.create_qc_plots(faulted_depth_maps, faulted_geologic_age, fault_traces)
```

### Main Faults Class

Coordinates all components:

```python
from datagenerator.faults import Faults

faults = Faults(
    parameters=params,
    unfaulted_depth_maps=depth_maps,
    onlap_horizon_list=onlap_list,
    geomodels=geomodel
)

# Apply faulting
faults.apply_faulting_to_geomodels_and_depth_maps()

# Build property models
faults.build_faulted_property_geomodels(facies)

# Improve depth maps
improved_maps, improved_gaps = faults.improve_depth_maps_post_faulting(
    unfaulted_geologic_age,
    faulted_geologic_age,
    onlap_clips
)
```

## Error Handling

The system uses custom exceptions for different types of errors:

- `FaultError`: Base exception for all fault-related errors
- `FaultGenerationError`: Raised when fault generation fails
- `FaultGeometryError`: Raised when geometry calculations fail
- `FaultQCError`: Raised when quality control checks fail

Example:

```python
try:
    faults.apply_faulting_to_geomodels_and_depth_maps()
except FaultGenerationError as e:
    print(f"Failed to generate faults: {e}")
except FaultQCError as e:
    print(f"Quality control failed: {e}")
```

## Testing

The system includes comprehensive tests:

```bash
pytest tests/test_faults.py
```

Tests cover:
- Initialization
- Input validation
- Fault application
- Depth map improvement
- Error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here] 