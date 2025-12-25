//! # OldiesRules Core
//!
//! Shared types and utilities for legacy simulator revival.
//!
//! ## Supported Legacy Simulators
//!
//! | Simulator | Original Language | Era | Status |
//! |-----------|------------------|-----|--------|
//! | GENESIS | SLI + C | 1980s-2014 | Semi-abandoned |
//! | XPPAUT | C + FORTRAN | 1990s | Hobby project |
//! | AUTO | FORTRAN | 1980s | Legacy |
//! | ModelDB | Various | 1996+ | Active but legacy |
//!
//! ## Design Philosophy
//!
//! 1. Preserve numerical equivalence with originals
//! 2. Modern Rust safety and performance
//! 3. HumanBrain integration ready

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Simulator type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Simulator {
    Genesis,
    Xppaut,
    Auto,
    ModelDB,
    Neuron,
    Brian,
}

/// Common errors
#[derive(Debug, Error)]
pub enum OldiesError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Simulation error: {0}")]
    SimulationError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Numerical error: {0}")]
    NumericalError(String),
}

pub type Result<T> = std::result::Result<T, OldiesError>;

/// Time point
pub type Time = f64;

/// Voltage (mV)
pub type Voltage = f64;

/// Current (nA)
pub type Current = f64;

/// Conductance (mS/cm^2)
pub type Conductance = f64;

/// Concentration (mM)
pub type Concentration = f64;

/// State vector for ODE systems
pub type StateVector = Array1<f64>;

/// Time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Time points
    pub time: Vec<Time>,
    /// Values at each time point
    pub values: Vec<f64>,
    /// Variable name
    pub name: String,
    /// Units
    pub units: Option<String>,
}

impl TimeSeries {
    pub fn new(name: &str) -> Self {
        Self {
            time: Vec::new(),
            values: Vec::new(),
            name: name.to_string(),
            units: None,
        }
    }

    pub fn push(&mut self, t: Time, v: f64) {
        self.time.push(t);
        self.values.push(v);
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

/// ODE system trait (for simulators)
pub trait OdeSystem {
    /// System dimension
    fn dimension(&self) -> usize;

    /// Compute derivatives: dy/dt = f(t, y)
    fn derivatives(&self, t: Time, y: &StateVector) -> StateVector;

    /// Optional Jacobian for stiff systems
    fn jacobian(&self, _t: Time, _y: &StateVector) -> Option<Array2<f64>> {
        None
    }
}

/// Simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    /// Start time
    pub t_start: Time,
    /// End time
    pub t_end: Time,
    /// Time step
    pub dt: Time,
    /// Output interval (for recording)
    pub output_dt: Option<Time>,
    /// Solver tolerance
    pub tolerance: f64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            t_start: 0.0,
            t_end: 100.0,
            dt: 0.01,
            output_dt: Some(0.1),
            tolerance: 1e-6,
        }
    }
}

/// Ion channel model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonChannel {
    /// Channel name
    pub name: String,
    /// Maximum conductance (mS/cm^2)
    pub g_max: Conductance,
    /// Reversal potential (mV)
    pub e_rev: Voltage,
    /// Gate variables
    pub gates: Vec<GateVariable>,
}

/// Gate variable (e.g., m, h, n in Hodgkin-Huxley)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateVariable {
    /// Variable name
    pub name: String,
    /// Power (exponent in gating)
    pub power: u32,
    /// Alpha rate function parameters
    pub alpha: RateFunction,
    /// Beta rate function parameters
    pub beta: RateFunction,
}

/// Rate function type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateFunction {
    /// Standard HH form: A*(V+B)/(exp((V+B)/C)-1)
    HodgkinHuxley { a: f64, b: f64, c: f64 },
    /// Exponential: A*exp((V+B)/C)
    Exponential { a: f64, b: f64, c: f64 },
    /// Sigmoid: A/(1+exp((V+B)/C))
    Sigmoid { a: f64, b: f64, c: f64 },
    /// Linear: A*(V+B)
    Linear { a: f64, b: f64 },
    /// Constant
    Constant(f64),
}

impl RateFunction {
    /// Evaluate rate at given voltage
    pub fn eval(&self, v: Voltage) -> f64 {
        match self {
            Self::HodgkinHuxley { a, b, c } => {
                let x = (v + b) / c;
                if x.abs() < 1e-6 {
                    // L'Hopital's rule for x -> 0
                    a * c
                } else {
                    a * (v + b) / (x.exp() - 1.0)
                }
            }
            Self::Exponential { a, b, c } => {
                a * ((v + b) / c).exp()
            }
            Self::Sigmoid { a, b, c } => {
                a / (1.0 + ((v + b) / c).exp())
            }
            Self::Linear { a, b } => {
                a * (v + b)
            }
            Self::Constant(c) => *c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_functions() {
        let hh = RateFunction::HodgkinHuxley { a: 0.1, b: 40.0, c: 10.0 };
        let rate = hh.eval(-65.0);
        assert!(rate > 0.0);

        let exp = RateFunction::Exponential { a: 0.1, b: 65.0, c: 80.0 };
        let rate = exp.eval(-65.0);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_time_series() {
        let mut ts = TimeSeries::new("voltage");
        ts.push(0.0, -65.0);
        ts.push(0.1, -64.0);
        assert_eq!(ts.len(), 2);
    }
}
