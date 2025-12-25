//! # ModelDB-RS
//!
//! Importer for ModelDB (https://modeldb.science) neural models.
//!
//! ModelDB contains 1800+ models from published papers. Many are in
//! legacy formats (GENESIS, NEURON HOC, old Python).
//!
//! This crate provides importers for:
//! - GENESIS script files
//! - NEURON HOC files
//! - NMODL mechanism files
//! - Legacy Python models

use oldies_core::{Result, OldiesError};
use serde::{Deserialize, Serialize};

/// ModelDB entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// ModelDB ID
    pub id: u32,
    /// Model name
    pub name: String,
    /// Original paper citation
    pub citation: String,
    /// Model type (GENESIS, NEURON, Brian, etc.)
    pub model_type: ModelType,
    /// Keywords
    pub keywords: Vec<String>,
    /// Brain regions
    pub regions: Vec<String>,
    /// Cell types
    pub cell_types: Vec<String>,
}

/// Model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Genesis,
    Neuron,
    Brian,
    Nest,
    Custom,
}

/// Import a model from ModelDB
pub async fn import_model(_id: u32) -> Result<ModelEntry> {
    // TODO: Implement API call to ModelDB
    Err(OldiesError::ModelNotFound("ModelDB import not yet implemented".into()))
}

/// Parse a GENESIS script file
pub fn parse_genesis_script(_content: &str) -> Result<()> {
    // TODO: Implement GENESIS parser
    todo!()
}

/// Parse a NEURON HOC file
pub fn parse_hoc_file(_content: &str) -> Result<()> {
    // TODO: Implement HOC parser
    todo!()
}

/// Parse an NMODL file
pub fn parse_nmodl(_content: &str) -> Result<()> {
    // TODO: Implement NMODL parser
    todo!()
}
