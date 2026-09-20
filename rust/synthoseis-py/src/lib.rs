//! Minimal PyO3 bindings over `synthoseis-io` MDIO create / write / read.
//!
//! Python module name: `synthoseis_mdio` (built via maturin).
//! No algorithm ports — I/O only.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use synthoseis_io::{CreateConfig, Dimension, MdioStore};

fn io_err(e: synthoseis_io::IoError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Thin handle around a Rust [`MdioStore`] on disk.
#[pyclass(name = "MdioStore")]
struct PyMdioStore {
    inner: MdioStore,
}

#[pymethods]
impl PyMdioStore {
    /// Create an empty MDIO-shaped Zarr v2 store at `path`.
    ///
    /// `shape` is `(inline, crossline, samples)` — defaults to `(2, 2, 4)`.
    #[staticmethod]
    #[pyo3(signature = (path, shape=None, chunks=None, digi=4.0, seed=42, units=None, name=None))]
    fn create_empty(
        path: &str,
        shape: Option<(usize, usize, usize)>,
        chunks: Option<(usize, usize, usize)>,
        digi: f64,
        seed: u64,
        units: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let (d0, d1, d2) = shape.unwrap_or((2, 2, 4));
        let config = CreateConfig {
            dimensions: [
                Dimension::sized("inline", d0),
                Dimension::sized("crossline", d1),
                Dimension::sized("time", d2),
            ],
            chunks: chunks.map(|(a, b, c)| [a, b, c]),
            digi,
            seed,
            units: units.unwrap_or_else(|| "ms".into()),
            name: name.unwrap_or_else(|| "synthoseis".into()),
        };
        let inner = MdioStore::create_empty(path, &config).map_err(io_err)?;
        Ok(Self { inner })
    }

    /// Open an existing synthoseis / MDIO store.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let inner = MdioStore::open(path).map_err(io_err)?;
        Ok(Self { inner })
    }

    /// Volume shape `(inline, crossline, samples)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.inner.shape();
        (s[0], s[1], s[2])
    }

    /// Absolute store root path.
    #[getter]
    fn path(&self) -> String {
        self.inner.root().display().to_string()
    }

    /// Write a full volume of float32 samples (C-order, length = product of shape).
    fn write_volume(&self, samples: Vec<f32>) -> PyResult<()> {
        self.inner.write_volume(&samples).map_err(io_err)
    }

    /// Write one chunk given chunk indices and float32 samples for that chunk.
    fn write_chunk(&self, chunk_indices: (usize, usize, usize), samples: Vec<f32>) -> PyResult<()> {
        self.inner
            .write_chunk([chunk_indices.0, chunk_indices.1, chunk_indices.2], &samples)
            .map_err(io_err)
    }

    /// Read the full float32 volume (C-order flat list).
    fn read_volume(&self) -> PyResult<Vec<f32>> {
        self.inner.read_volume().map_err(io_err)
    }

    /// Read `live_mask` as a list of bools (length = inline * crossline).
    fn read_live_mask(&self) -> PyResult<Vec<bool>> {
        let bytes = self.inner.read_live_mask().map_err(io_err)?;
        Ok(bytes.into_iter().map(|b| b != 0).collect())
    }

    fn __repr__(&self) -> String {
        let s = self.inner.shape();
        format!(
            "MdioStore(path={:?}, shape=({}, {}, {}))",
            self.inner.root(),
            s[0],
            s[1],
            s[2]
        )
    }
}

/// Create an empty MDIO store and return an [`MdioStore`] handle.
#[pyfunction]
#[pyo3(signature = (path, shape=None, chunks=None, digi=4.0, seed=42))]
fn create_empty(
    path: &str,
    shape: Option<(usize, usize, usize)>,
    chunks: Option<(usize, usize, usize)>,
    digi: f64,
    seed: u64,
) -> PyResult<PyMdioStore> {
    PyMdioStore::create_empty(path, shape, chunks, digi, seed, None, None)
}

/// Open an existing store.
#[pyfunction]
fn open_store(path: &str) -> PyResult<PyMdioStore> {
    PyMdioStore::open(path)
}

/// Write a full volume to an existing store path (open → write).
#[pyfunction]
fn write_volume(path: &str, samples: Vec<f32>) -> PyResult<()> {
    let store = MdioStore::open(path).map_err(io_err)?;
    store.write_volume(&samples).map_err(io_err)
}

/// Read a full volume from a store path.
#[pyfunction]
fn read_volume(path: &str) -> PyResult<Vec<f32>> {
    let store = MdioStore::open(path).map_err(io_err)?;
    store.read_volume().map_err(io_err)
}

/// Read live_mask from a store path (list of bools).
#[pyfunction]
fn read_live_mask(path: &str) -> PyResult<Vec<bool>> {
    let store = MdioStore::open(path).map_err(io_err)?;
    let bytes = store.read_live_mask().map_err(io_err)?;
    Ok(bytes.into_iter().map(|b| b != 0).collect())
}

#[pymodule]
fn synthoseis_mdio(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMdioStore>()?;
    m.add_function(wrap_pyfunction!(create_empty, m)?)?;
    m.add_function(wrap_pyfunction!(open_store, m)?)?;
    m.add_function(wrap_pyfunction!(write_volume, m)?)?;
    m.add_function(wrap_pyfunction!(read_volume, m)?)?;
    m.add_function(wrap_pyfunction!(read_live_mask, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
