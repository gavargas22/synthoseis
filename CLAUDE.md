# CLAUDE.md - Synthoseis Development Guide

## Project Overview

Synthoseis is an open-source Python tool for generating pseudo-random synthetic seismic data and associated labels to train deep learning networks. It simulates geological processes (horizons, faults, closures, salt bodies) and computes seismic responses using rock physics models and Zoeppritz equations.

**Version:** 2.5.1
**License:** MIT
**Python:** >=3.12 (runtime uses 3.13 per `.python-version`)
**Paper:** [Synthetic seismic data for training deep learning networks (SEG, 2022)](https://library.seg.org/doi/abs/10.1190/INT-2021-0193.1)

## Repository Structure

```
synthoseis/
├── main.py                      # Entry point - CLI for model generation
├── pyproject.toml               # Package definition, dependencies
├── config/                      # JSON configuration files
│   ├── example.json             # Full-size example config
│   └── test_config.json         # Smaller config (50x50x50) for testing
├── datagenerator/               # Core generation pipeline (~15k LOC)
│   ├── Parameters.py            # Config loading, model setup (Borg pattern)
│   ├── Horizons.py              # Layer/horizon generation
│   ├── Geomodels.py             # 3D geological model construction
│   ├── Faults.py                # Fault generation and displacement
│   ├── Closures.py              # Trap/closure identification (flood-fill)
│   ├── Seismic.py               # Seismic volume (Vp/Vs/Rho, Zoeppritz)
│   ├── Salt.py                  # Salt body generation
│   ├── Augmentation.py          # Geophysical augmentation (stretch, RMO)
│   ├── wavelets.py              # Ricker wavelet, FFT utilities
│   ├── simplexNoise.py          # Perlin noise generation
│   ├── meanderpy.py             # Channel/meander simulation
│   ├── fluvsim.py               # Fortran channel interface
│   ├── fluvsim.f90              # Fortran channel simulation source
│   ├── histogram_equalizer.py   # Histogram equalization
│   └── util.py                  # Plotting, file I/O, math helpers
├── rockphysics/                 # Rock physics models
│   ├── RockPropertyModels.py    # RockProperties, EndMemberMixing classes
│   └── rpm_example.py           # Example depth-trend template
├── synthoseis/                  # Modern package structure
│   └── storage/
│       ├── __init__.py          # Exports StorageClient
│       └── mdio_backend.py      # Zarr v3 storage backend
├── tests/                       # Pytest test suite
│   ├── test_parameters.py       # Parameter + storage init tests
│   └── test_storage_mdio.py     # Zarr storage CRUD tests
├── notebooks/
│   └── synthoseis-quick-start.ipynb
├── scripts/
│   └── start_geocrawlSynthetics.sh
├── docs/                        # Auto-generated HTML API docs
└── img/                         # Gallery images for README
```

## Build and Development Commands

### Package Management

The project uses **uv** as its package manager (see `uv.lock`). Conda (`environment.yml`) is a legacy alternative.

```bash
# Install dependencies with uv
uv sync

# Install with pip (alternative)
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### Running the Application

```bash
# Full model generation
python main.py --config config/example.json --num_runs 1 --run_id my_run

# Test mode (reduced cube size)
python main.py --config config/test_config.json --num_runs 1 --run_id test --test_mode 50
```

CLI arguments:
- `-c, --config_file` (required): Path to JSON config file
- `-n, --num_runs`: Number of models to create (default: 1)
- `-r, --run_id`: Run identifier for output naming
- `-t, --test_mode`: Integer to reduce model size for quick testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_parameters.py
pytest tests/test_storage_mdio.py

# Verbose output
pytest tests/ -v
```

Test coverage is minimal - only storage and parameter initialization are tested. There are no integration tests for the full generation pipeline.

### Linting

```bash
# Lint with ruff (included as a project dependency)
ruff check .

# Auto-fix
ruff check --fix .
```

## Architecture and Key Patterns

### Model Generation Pipeline

The `build_model()` function in `main.py` orchestrates the full pipeline:

1. **Parameters** - Load JSON config, initialize storage
2. **Horizons** - Build unfaulted depth maps, create facies array
3. **Geomodels** - Construct 3D geological age models
4. **Faults** - Generate fault planes, apply displacement
5. **Closures** - Identify traps using flood-fill algorithms
6. **Seismic** - Calculate elastic properties, apply Zoeppritz, generate seismic volumes
7. **Cleanup** - Close storage, remove temp files

### The Borg Pattern (Parameters)

`Parameters` uses the Borg pattern (shared state via `_Borg` base class) so all modules share the same parameter state. This is the central configuration object passed to every module.

### Storage Backend

Storage was migrated from HDF5 to Zarr v3. The `StorageClient` class (`synthoseis/storage/mdio_backend.py`) provides:
- `StorageClient.open(path, mode)` - Open/create a Zarr store
- `create_dataset(name, data=, shape=)` - Create arrays
- `get_dataset(name, lazy=, use_dask=)` - Retrieve arrays (numpy or dask)
- `list_datasets()`, `remove_dataset()`, `close()`

Default chunk size: `(128, 128, 128)`.

### Parallelization

- **Dask** (`dask[distributed]`) for distributed array processing
- **Cubed** for larger-than-memory array operations
- **Multiprocessing** for bandpass filtering (`multiprocess_bp` config flag)
- **Numba** JIT compilation for performance-critical numerical code

### Configuration

Model parameters are defined in JSON files under `config/`. Key parameters:
- `cube_shape`: Output volume dimensions `[X, Y, Z]`
- `incident_angles`: Seismic angle stacks to generate
- `digi`: Vertical sampling rate
- `closure_types`: `["simple", "faulted", "onlap"]`
- `include_salt`, `basin_floor_fans`: Feature toggles

Use `config/test_config.json` (50x50x50 cube) for development and testing.

## Code Conventions

### File Naming

- Core modules use PascalCase filenames: `Parameters.py`, `Horizons.py`, `Faults.py`
- Utility/helper modules use snake_case: `util.py`, `wavelets.py`, `histogram_equalizer.py`
- Test files follow pytest convention: `test_*.py`

### Code Style

- No strict formatter enforced; ruff is used for linting
- Docstrings use NumPy-style format (Parameters, RockPropertyModels)
- Classes use PascalCase; functions/methods use snake_case
- Type hints are used sparingly (mainly in newer code like `StorageClient`)
- Scientific computing conventions: heavy use of numpy, scipy, and array operations

### Dependencies of Note

- **bruges** - Rock physics and seismic analysis (Zoeppritz equations, moduli)
- **scikit-image** - Image processing (morphology, flood-fill for closures)
- **numba** - JIT compilation for numerical loops
- **dask** - Lazy/parallel array processing
- **zarr** - Chunked, compressed array storage (v3)
- **perlin-noise** - Stochastic terrain generation

### Important Implementation Details

- The `Parameters` object manages temporary directories and cleanup. It creates `temp_folder` and `work_subfolder` paths that are cleaned up after model completion.
- Rock property models are dynamically imported based on the `project` field in the config JSON (see `rockphysics/rpm_example.py` as a template).
- The Fortran code (`fluvsim.f90`) is optional and requires `intel-fortran-rt` package. Channel simulation is deprecated (`include_channels` is always false).
- Array data uses float32 by default in the storage layer. Some internal computations use float64 for precision.
- The codebase is actively being modernized: HDF5 was removed in favor of Zarr v3, and dask/cubed parallelization was recently added.

## Git Workflow

- **Main branch:** `master`
- Feature branches use descriptive names (e.g., `feat/gavargas22/mdio`, `claude/migrate-hdf5-storage-*`)
- Commit messages: use conventional prefix style (`feat:`, `fix:`, etc.) for new work
- PRs should include clear descriptions and ensure tests pass
- No CI/CD pipeline is currently configured

## Common Development Tasks

### Adding a New Rock Property Model

1. Copy `rockphysics/rpm_example.py` as a template
2. Define depth trends for Vp, Vs, Rho per facies (shale, brine-sand, oil-sand, gas-sand)
3. Create a new config JSON with `"project": "your_model_name"`
4. The `Parameters` class dynamically loads the model by name

### Modifying the Generation Pipeline

Each stage in `main.py:build_model()` is a distinct module. To modify a stage:
1. Read the corresponding module in `datagenerator/`
2. The `Parameters` object (`p`) is passed to each module constructor
3. Storage is accessed via `p.storage` (a `StorageClient` instance)
4. Intermediate results are stored in the Zarr store under the temp folder

### Running a Quick Test

```bash
python main.py -c config/test_config.json -n 1 -r quick_test -t 32
```

The `-t 32` flag reduces the cube to 32x32 samples, making generation much faster.
