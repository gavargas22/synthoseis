# Product Mission

## Pitch

Synthoseis is a synthetic seismic data generator that helps geophysicists and AI engineers create unlimited realistic 3D seismic volumes and labels by simulating complex geology without high memory costs or real surveys.

## Users

### Primary Customers

- **Geophysicists**: Build/test seismic models for exploration workflows
- **AI/ML Engineers**: Generate labeled training data for seismic interpretation models

### User Personas

**Dr. Elena Petrova** (35-50 years old)
- **Role**: Senior Geophysicist
- **Context**: Oil & gas exploration, needs realistic models for AVO analysis
- **Pain Points**: Limited real survey data, high compute costs for simulations
- **Goals**: Rapid prototyping of fault/salt/reservoir scenarios, QC seismic models

**Mark Chen** (28-40 years old) 
- **Role**: ML Engineer
- **Context**: Training seismic fault detection networks
- **Pain Points**: Lack of labeled 3D seismic data, memory limits on large volumes
- **Goals**: Generate 1000s of diverse labeled volumes, scale to TB-scale datasets

## The Problem

### Memory-Constrained Seismic Simulation

Current tools load entire 3D volumes into RAM (50-500GB+), causing OOM errors on standard hardware. This limits model complexity and dataset scale for ML training.

**Our Solution:** MDIO/Zarr streaming enables arbitrarily large volumes with constant memory usage.

### HDF5 I/O Bottlenecks

Legacy HDF5 files are slow for parallel access and cloud deployment, blocking modern ML workflows.

**Our Solution:** Zarr-based MDIO provides distributed, cloud-native I/O with chunked access.

### Sparse Labeled Data

Real seismic surveys rarely have complete labels (faults, salt, horizons), crippling supervised ML.

**Our Solution:** Generate perfectly labeled synthetic volumes at scale.

## Differentiators

### Infinite Scale, Constant Memory

Unlike Landmark/Paradigm simulators requiring 128GB+ RAM, Synthoseis streams via MDIO/Zarr - generate 1TB volumes on a laptop.

### ML-Ready Labels

Every voxel labeled (lithology, faults, salt, closures, NTG) - unlike sparse real data annotations.

### Geophysical Fidelity

Full Zoeppritz AVO modeling + realistic geology (fluvial Fortran sims, stochastic faults) - not simplistic ray-tracing.

## Key Features

### Core Features

- **3D Geological Modeling**: Horizons → faults → salt → fluvial channels → realistic stratigraphy
- **AVO Seismic Simulation**: Near/mid/far angle stacks via Zoeppritz equations
- **Perfect Labels**: Voxel-level masks for faults, salt, horizons, lithology, NTG, closures
- **Streaming I/O**: MDIO/Zarr backend - unlimited volume size, constant memory

### Generation Controls

- **Config-Driven**: JSON params for geology complexity, noise, bandwidth, test mode
- **Stochastic Variety**: Random faults/salt/closures/reservoir properties per run
- **QC Visualization**: 3D closure plots, broadband volumes, keyfiles

### Scalability

- **CLI Batch Mode**: Generate 1000s of models (`--num_runs 1000`)
- **HPC Ready**: Multiprocessing, temp dirs, Fortran performance core
- **Cloud Native**: Zarr stores work with Dask/distributed
