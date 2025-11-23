# Technical Stack

## Core Application

- **Language**: Python >=3.12
- **Packaging**: pyproject.toml (Hatch/Poetry compatible)
- **Numerics**: NumPy >=2.0, SciPy, Numba >=0.62 (JIT compilation)
- **Performance**: Dask (lazy arrays), psutil, multiprocessing
- **Visualization**: Matplotlib (Agg backend), Plotly

## Storage & I/O

- **Primary**: multidimio[distributed] (MDIO/Zarr protocol)
- **Backend**: Zarr (chunked, compressed arrays)
- **Legacy**: HDF5/PyTables (migration target: remove)
- **Metadata**: SQLite (parameters DB), JSON configs

## Domain-Specific

- **Geophysics**: bruges (rock physics, Zoeppritz AVO)
- **Geology**: noise (simplex noise), scikit-image (morphology)
- **Fluvial**: Fortran (gfortran-compiled fluvsim.f90)
- **Progress**: tqdm

## Development

- **Linting**: ruff >=0.10
- **Testing**: pytest >=8.4
- **CLI**: argparse (main.py entrypoint)

## Deployment

- **Runtime**: Pure Python CLI, no server
- **Hosting**: Local/HPC/Cloud (Zarr cloud-compatible)
- **Scaling**: Dask-distributed (parallel generation)
- **Entry Point**: `python main.py -c config.json --num_runs 1000`

## Architecture Decisions

- **Shared State**: Borg pattern (Parameters singleton)
- **Modular**: datagenerator/ modules by geology step
- **Streaming**: MDIO replaces in-memory arrays
- **Config-First**: JSON-driven parameterization
