//! MDIO store create / open / read / write.
mod create;
mod rw;

use crate::CreateConfig;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct MdioStore {
    pub(crate) root: PathBuf,
    pub(crate) config: CreateConfig,
}

impl MdioStore {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &CreateConfig {
        &self.config
    }

    pub fn shape(&self) -> [usize; 3] {
        self.config.shape()
    }
}

pub use rw::DeliverableWriter;
