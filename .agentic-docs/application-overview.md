# Synthoseis Application Overview

## What is Synthoseis?

Synthoseis is a Python application that generates synthetic seismic data and associated labels. It's designed to create realistic 3D geological models that mimic real-world subsurface structures, which can then be used to train machine learning models for seismic interpretation tasks. Think of it as a "factory" that produces fake but realistic earthquake data for AI training purposes.

## High-Level Workflow

1. **Setup**: The user provides configuration parameters (like model size, geological features to include).
2. **Model Generation**: The app builds a 3D geological model step-by-step, adding layers like faults, salt bodies, horizons, and rock properties.
3. **Data Storage**: All generated data is stored in a structured format (currently HDF5, being migrated to MDIO/zarr).
4. **Output**: Produces seismic traces, labels, and metadata that can be used for training neural networks.

## Key Components

### Main Entry Point (`main.py`)
- This is where the application starts.
- It reads user configuration, sets up parameters, and orchestrates the entire model generation process.
- Calls functions to create directories, initialize storage, and run the generation pipeline.

### Configuration and Parameters (`datagenerator/Parameters.py`)
- Uses a "Borg" pattern (shared state across instances) to manage global settings.
- Handles:
  - Reading user config from JSON files.
  - Setting up folder structures (project, work, temp directories).
  - Initializing data storage (HDF5 files or MDIO stores).
  - Writing metadata and key files for coordinate systems.
- Acts as the central "control panel" for all model parameters.

### Data Generation Modules (`datagenerator/`)
These modules build different parts of the geological model:

- **Geomodels.py**: Creates the basic 3D geological framework (layers, stratigraphy).
- **Faults.py**: Adds fault structures (breaks in the rock layers).
- **Salt.py**: Inserts salt bodies (irregular shapes that affect seismic waves).
- **Horizons.py**: Defines layer boundaries and surfaces.
- **Seismic.py**: Generates synthetic seismic traces from the model.
- **Parameters.py** (already mentioned): Manages all settings.
- **Util.py**: Utility functions for data handling, file I/O, and helper operations.

### Rock Physics (`rockphysics/`)
- Calculates physical properties of rocks (velocity, density) based on geological inputs.
- Uses models to convert geological descriptions into numerical properties that affect seismic wave propagation.

### Storage and I/O
- **Current**: Uses HDF5 files via PyTables for storing large 3D arrays.
- **Migration**: Moving to MDIO (zarr-based) for better performance and cloud compatibility.
- Handles reading/writing seismic cubes, labels, and intermediate data.

### Other Components
- **Notebooks**: Example Jupyter notebooks showing how to use the app.
- **Tests**: Unit tests to verify functionality.
- **Scripts**: Shell scripts for running on HPC systems.

## Step-by-Step Model Generation Process

1. **Initialization**:
   - Load user config (JSON file with settings like cube size, features to include).
   - Create a `Parameters` object that holds all settings.
   - Set up directories and initialize storage.

2. **Build Geological Framework**:
   - Create basic stratigraphy (rock layers) using `Geomodels`.
   - Add faults using `Faults` (randomly placed breaks in layers).
   - Insert salt bodies using `Salt` (complex 3D shapes).

3. **Add Physical Properties**:
   - Use `RockPropertyModels` to assign velocities, densities to each voxel in the 3D model.

4. **Generate Seismic Data**:
   - Use `Seismic` to simulate how seismic waves would travel through the model.
   - Produce synthetic seismic traces and labels.

5. **Quality Control and Output**:
   - Save intermediate results for debugging.
   - Write final seismic cube and labels to storage.
   - Generate metadata files (key files for coordinates).

## Data Flow

- **Input**: JSON config file, optional test mode settings.
- **Processing**: Parameters → Geological model → Physical properties → Seismic simulation.
- **Storage**: All data stored in HDF5/MDIO as 3D arrays (seismic cube, labels, properties).
- **Output**: Seismic traces, horizon labels, fault masks, salt segmentations, and metadata.

## Configuration

- **User Config**: JSON file (`config/example.json`) with parameters like:
  - Cube dimensions (inline, crossline, time/depth).
  - Geological features (number of faults, salt presence).
  - Rock physics scaling factors.
- **Test Mode**: Reduces model size for quick testing.
- **Run ID**: Allows multiple runs with different settings.

## Outputs

- **Seismic Cube**: 3D array of synthetic seismic traces.
- **Labels**: Masks for faults, salt, horizons.
- **Metadata**: Coordinate systems, binning info.
- **QC Files**: Intermediate numpy arrays for debugging.

## Maintenance Notes

- **Shared State**: `Parameters` uses Borg pattern; changes affect all instances.
- **Storage Migration**: Currently HDF5, moving to MDIO/zarr. Update `Parameters.storage_setup` and replace `h5file` usages.
- **Dependencies**: Requires numpy, scipy, matplotlib, tables (HDF5), mdio/zarr (new).
- **Performance**: Large models need significant memory; uses multiprocessing where possible.
- **Extensibility**: New geological features can be added by creating new modules in `datagenerator/`.

This overview should help maintain functionality during updates like the MDIO migration.