# MDIO Conversion Tasks

## Version
1.0 - Created November 22, 2025

## Overview
Tasks derived from MDIO Conversion spec. Total estimate: 20-30 person-days. Assigned to "Dev" (placeholder; adjust as needed).

## Tasks

- [ ] 1. Prep StorageClient & Dependencies
  - [ ] 1.1 Write unit tests for enhanced StorageClient (chunking/lazy/error handling)
  - [ ] 1.2 Update pyproject.toml (multidimio[distributed], dask[distributed])
  - [ ] 1.3 Add chunking param, lazy flag, error handling to StorageClient
  - [ ] 1.4 Verify all tests pass

- [ ] 2. Centralize Parameters.py
  - [ ] 2.1 Write integration tests for Parameters storage flow
  - [ ] 2.2 Remove hdf_setup, self.hdf_master, related methods
  - [ ] 2.3 Enhance storage_setup for chunks/metadata
  - [ ] 2.4 Deprecate hdf_store flag (warnings)
  - [ ] 2.5 Test small model with Zarr only
  - [ ] 2.6 Verify all tests pass

- [ ] 3. Migrate Core datagenerator Modules
  - [ ] 3.1 Write tests for I/O replacement in Seismic/Salt/Geomodels
  - [ ] 3.2 Seismic.py: np.load → get_dataset(lazy=True)
  - [ ] 3.3 Salt.py/Geomodels.py: np.save → create_dataset
  - [ ] 3.4 util.py: Remove write_data_to_hdf
  - [ ] 3.5 Audit/replace Faults/Horizons/Closures I/O
  - [ ] 3.6 End-to-end test; grep no 'h5'/'np.save'
  - [ ] 3.7 Verify all tests pass

- [ ] 4. Migrate Rockphysics & Notebooks
  - [ ] 4.1 Write tests for rockphysics MDIO usage
  - [ ] 4.2 RockPropertyModels.py: HDF → create_dataset
  - [ ] 4.3 Update notebooks (synthoseis-quick-start.ipynb)
  - [ ] 4.4 Test rpm_example.py + notebooks
  - [ ] 4.5 Verify all tests pass

- [ ] 5. Streaming, Testing & Cleanup
  - [ ] 5.1 Write tests for Dask lazy loading/large volumes
  - [ ] 5.2 Add Dask arrays to get_dataset; chunk config in Parameters
  - [ ] 5.3 Test 2000^3 volume (<4GB RAM)
  - [ ] 5.4 Expand test_storage_mdio.py (100% coverage)
  - [ ] 5.5 Update README/docs; final legacy removal
  - [ ] 5.6 Benchmarks; full verification
  - [ ] 5.7 Verify all tests pass

## Execution Notes
- **Total Timeline**: 4-6 weeks sequential; parallelize T3/T4 if multiple devs.
- **Milestones**: After T3 (core migration), test large model.
- **Risks**: Ensure no performance regressions; add benchmarks.

To execute, reply "execute MDIO tasks".
