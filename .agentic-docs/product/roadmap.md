# Product Roadmap

## Phase 0: Already Completed ✅

**Goal**: Full synthetic seismic pipeline
**Success Criteria**: End-to-end model generation with QC

### Features
- [x] 3D geological modeling (horizons/faults/salt/fluvial) `XL`
- [x] AVO seismic simulation (near/mid/far stacks) `XL` 
- [x] Perfect voxel labels (lithology/NTG/faults/closures) `L`
- [x] CLI batch generation (`--num_runs`) `M`
- [x] MDIO storage partial integration `M`
- [x] QC plots/volumes/keyfiles `S`

## Phase 1: MDIO Modernization (Current)

**Goal**: Remove all legacy I/O, enable streaming
**Success Criteria**: Zero np.save/h5py, generate 10GB+ volumes

### Features
- [ ] Complete MDIO migration across all modules `L`
- [ ] Remove HDF5/PyTables/h5py code `M`
- [ ] Streaming geology generation (lazy layers) `L`
- [ ] Distributed generation (Dask cluster) `M`
- [ ] Comprehensive MDIO tests `S`
- [ ] Updated notebooks/docs `XS`

### Dependencies
- multidimio[zarr] stable API

## Phase 2: Major Refactor

**Goal**: Maintainable, readable codebase
**Success Criteria**: 80% test coverage, clear module boundaries

### Features
- [ ] Extract core classes from Borg pattern `L`
- [ ] Domain module separation (geology/seismic/storage) `L`
- [ ] Type hints + mypy `M`
- [ ] Comprehensive tests (80% coverage) `L`
- [ ] Domain-specific abstractions `XL`

## Phase 3: Scale & Polish

**Goal**: Production ML data factory
**Success Criteria**: 1000-model batches, cloud deployment

### Features
- [ ] Cloud runner (AWS Batch/K8s) `L`
- [ ] Model registry/catalog `M`
- [ ] Advanced geology (fractures, geobodies) `XL`
- [ ] REST API for on-demand generation `M`
