//! # AUTO-RS
//!
//! Revival of AUTO continuation/bifurcation software in Rust.
//!
//! AUTO is a FORTRAN program for continuation and bifurcation problems
//! in ODEs. Originally by Eusebius Doedel.
//!
//! This crate provides the core continuation algorithms used by XPPAUT.

use oldies_core::Result;

/// Continuation parameters
pub struct ContinuationParams {
    /// Parameter to continue
    pub parameter: String,
    /// Starting value
    pub start: f64,
    /// Ending value
    pub end: f64,
    /// Maximum step size
    pub ds_max: f64,
    /// Minimum step size
    pub ds_min: f64,
    /// Initial step size
    pub ds: f64,
}

/// Continuation result
pub struct ContinuationResult {
    /// Parameter values along branch
    pub parameters: Vec<f64>,
    /// State values along branch
    pub states: Vec<Vec<f64>>,
    /// Stability at each point
    pub stable: Vec<bool>,
}

/// Natural parameter continuation
pub fn natural_continuation<F>(
    _rhs: F,
    _initial_state: &[f64],
    _params: ContinuationParams,
) -> Result<ContinuationResult>
where
    F: Fn(&[f64], f64) -> Vec<f64>,
{
    todo!("Natural continuation not yet implemented")
}

/// Pseudo-arclength continuation
pub fn arclength_continuation<F>(
    _rhs: F,
    _initial_state: &[f64],
    _params: ContinuationParams,
) -> Result<ContinuationResult>
where
    F: Fn(&[f64], f64) -> Vec<f64>,
{
    todo!("Arclength continuation not yet implemented")
}
