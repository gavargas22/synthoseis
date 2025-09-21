# Synthoseis AI Coding Guidelines

## Architecture Overview
Synthoseis generates synthetic seismic data for ML training. Core architecture:
- **Parameters.py** (Borg pattern): Shared global state for config, storage, and metadata. All modules access `self.cfg` for settings.
- **datagenerator/**: Modular geological model builders (Faults.py for faults, Salt.py for salt bodies, Seismic.py for traces).
- **Storage**: MDIO/zarr for scalable data storage. Data stored as datasets in zarr store.
- **Data Flow**: JSON config → Parameters → Layered model generation → Seismic simulation → MDIO/zarr output.

Why: Modular design allows independent feature development; shared state simplifies cross-module communication.

## Developer Workflows
- **Run model**: `python main.py --config config/example.json --num_runs 1 --run_id test` (use `--test_mode 50` for quick 50x50x50 cubes).
- **Install**: `pip install -e .` or `conda env create -f environment.yml`.
- **Test**: `pytest tests/` (focus on unit tests for individual modules).
- **Debug**: Enable `verbose: true` in config; check intermediate numpy saves in work folder.

## Project Conventions
- **Shared State**: Use Borg pattern in Parameters.py; access via `self.cfg.storage` for storage, `self.cfg.cube_shape` for dimensions.
- **Storage Access**: Read/write arrays via `self.cfg.storage.get_dataset(name)` or `self.cfg.storage.create_dataset(name, data)`.
- **File Structure**: Config in `config/`, outputs in `project_folder/work_subfolder/`, temp data in `temp_folder/`.
- **Error Handling**: Use try/except for file ops; log via `self.cfg.write_to_logfile()`.
- **Imports**: Standard libs first, then local.

## Integration Points
- **Dependencies**: numpy/scipy for arrays, mdio/zarr for storage, matplotlib for plots. Rock physics via custom models in `rockphysics/`.
- **Cross-Component**: Modules communicate via shared Parameters; e.g., Salt.py updates datasets via `self.cfg.storage`.
- **External APIs**: No external services; self-contained data generation.

## Examples
- Add new feature: Create `datagenerator/NewFeature.py`, integrate in main workflow via Parameters.
- Storage usage: Use `self.cfg.storage.create_dataset("dataset", data)` to store, `self.cfg.storage.get_dataset("dataset")` to read.
- Config changes: Update JSON, test with small cube to verify data flow.