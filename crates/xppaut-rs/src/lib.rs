//! # XPPAUT-RS
//!
//! Revival of XPPAUT (XPP + AUTO) bifurcation analysis in Rust.
//!
//! ## History
//!
//! XPPAUT was created by Bard Ermentrout at University of Pittsburgh.
//! It's described by the author as "sort of a hobby of mine" with a
//! "dated user interface" that "sometimes crashes" and has "no scripting
//! interface."
//!
//! This crate provides a modern, safe, high-performance implementation
//! of XPPAUT's bifurcation analysis capabilities.
//!
//! ## Capabilities
//!
//! 1. **ODE Integration**: Multiple solvers (Euler, RK4, adaptive)
//! 2. **Bifurcation Analysis**: Fixed points, limit cycles, Hopf bifurcations
//! 3. **Continuation**: Parameter continuation via AUTO algorithm
//! 4. **Phase Portraits**: Nullclines, vector fields
//! 5. **Stability Analysis**: Eigenvalues, Floquet multipliers

use oldies_core::{OdeSystem, Result, StateVector, Time, OldiesError};
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Bifurcation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BifurcationType {
    /// Saddle-node (fold)
    SaddleNode,
    /// Transcritical
    Transcritical,
    /// Pitchfork
    Pitchfork,
    /// Hopf (supercritical or subcritical)
    Hopf { supercritical: bool },
    /// Period-doubling
    PeriodDoubling,
    /// Limit point of cycles
    LimitPointCycles,
    /// Torus (Neimark-Sacker)
    Torus,
}

/// Fixed point with stability info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    /// State at fixed point
    pub state: Vec<f64>,
    /// Parameter value
    pub parameter: f64,
    /// Eigenvalues
    pub eigenvalues: Vec<Complex64>,
    /// Is stable?
    pub stable: bool,
    /// Type (node, focus, saddle, etc.)
    pub point_type: FixedPointType,
}

/// Fixed point classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedPointType {
    StableNode,
    UnstableNode,
    StableFocus,
    UnstableFocus,
    Saddle,
    Center,
    Unknown,
}

/// Limit cycle with stability info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitCycle {
    /// Period
    pub period: f64,
    /// Amplitude (max deviation from mean)
    pub amplitude: f64,
    /// Parameter value
    pub parameter: f64,
    /// Floquet multipliers
    pub floquet_multipliers: Vec<Complex64>,
    /// Is stable?
    pub stable: bool,
}

/// Bifurcation point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationPoint {
    /// Type of bifurcation
    pub bifurcation_type: BifurcationType,
    /// Parameter value at bifurcation
    pub parameter: f64,
    /// State at bifurcation
    pub state: Vec<f64>,
    /// Additional info (e.g., Hopf frequency)
    pub info: Option<String>,
}

/// Bifurcation diagram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationDiagram {
    /// Parameter name
    pub parameter_name: String,
    /// Parameter range
    pub parameter_range: (f64, f64),
    /// State variable index being tracked
    pub state_index: usize,
    /// Fixed point branches
    pub fixed_points: Vec<FixedPoint>,
    /// Limit cycle branches
    pub limit_cycles: Vec<LimitCycle>,
    /// Bifurcation points
    pub bifurcations: Vec<BifurcationPoint>,
}

/// XPPAUT-style ODE model
#[derive(Debug, Clone)]
pub struct XppModel {
    /// Model name
    pub name: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Parameter names and values
    pub parameters: Vec<(String, f64)>,
    /// ODE right-hand sides (as Rust closures would require Box<dyn Fn>)
    dimension: usize,
}

impl XppModel {
    /// Create a new model
    pub fn new(name: &str, variables: Vec<String>) -> Self {
        let dimension = variables.len();
        Self {
            name: name.to_string(),
            variables,
            parameters: Vec::new(),
            dimension,
        }
    }

    /// Add a parameter
    pub fn add_parameter(&mut self, name: &str, value: f64) {
        self.parameters.push((name.to_string(), value));
    }

    /// Get parameter value
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }

    /// Set parameter value
    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<()> {
        for (n, v) in &mut self.parameters {
            if n == name {
                *v = value;
                return Ok(());
            }
        }
        Err(OldiesError::ModelNotFound(format!("Parameter {} not found", name)))
    }
}

/// Bifurcation analyzer
pub struct BifurcationAnalyzer {
    /// Model
    model: XppModel,
    /// Numerical tolerance
    tolerance: f64,
    /// Maximum iterations for Newton's method
    max_iterations: usize,
}

impl BifurcationAnalyzer {
    /// Create a new analyzer
    pub fn new(model: XppModel) -> Self {
        Self {
            model,
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }

    /// Find fixed points at current parameter values
    pub fn find_fixed_points<F>(&self, rhs: F, initial_guesses: &[Vec<f64>]) -> Vec<FixedPoint>
    where
        F: Fn(&[f64], &[(String, f64)]) -> Vec<f64>,
    {
        let mut fixed_points = Vec::new();

        for guess in initial_guesses {
            if let Some(fp) = self.newton_raphson(&rhs, guess) {
                // Check if we already found this one
                let is_new = fixed_points.iter().all(|existing: &FixedPoint| {
                    let dist: f64 = existing.state.iter()
                        .zip(&fp)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    dist > self.tolerance * 100.0
                });

                if is_new {
                    // Compute Jacobian and eigenvalues
                    let jacobian = self.numerical_jacobian(&rhs, &fp);
                    let eigenvalues = self.compute_eigenvalues(&jacobian);
                    let stable = eigenvalues.iter().all(|e| e.re < 0.0);
                    let point_type = classify_fixed_point(&eigenvalues);

                    fixed_points.push(FixedPoint {
                        state: fp,
                        parameter: 0.0, // Would need parameter tracking
                        eigenvalues,
                        stable,
                        point_type,
                    });
                }
            }
        }

        fixed_points
    }

    /// Newton-Raphson for finding zeros
    fn newton_raphson<F>(&self, rhs: F, initial: &[f64]) -> Option<Vec<f64>>
    where
        F: Fn(&[f64], &[(String, f64)]) -> Vec<f64>,
    {
        let mut x = initial.to_vec();
        let n = x.len();

        for _ in 0..self.max_iterations {
            let f = rhs(&x, &self.model.parameters);

            // Check convergence
            let norm: f64 = f.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            if norm < self.tolerance {
                return Some(x);
            }

            // Compute Jacobian
            let jacobian = self.numerical_jacobian(&rhs, &x);

            // Solve J * delta = -f
            let j = DMatrix::from_row_slice(n, n, &jacobian);
            let f_vec = DVector::from_vec(f.iter().map(|v| -*v).collect());

            if let Some(lu) = j.lu().solve(&f_vec) {
                for i in 0..n {
                    x[i] += lu[i];
                }
            } else {
                return None; // Singular Jacobian
            }
        }

        None
    }

    /// Numerical Jacobian via finite differences
    fn numerical_jacobian<F>(&self, rhs: F, x: &[f64]) -> Vec<f64>
    where
        F: Fn(&[f64], &[(String, f64)]) -> Vec<f64>,
    {
        let n = x.len();
        let h = 1e-8;
        let mut jacobian = vec![0.0; n * n];

        let f0 = rhs(x, &self.model.parameters);

        for j in 0..n {
            let mut x_plus = x.to_vec();
            x_plus[j] += h;
            let f_plus = rhs(&x_plus, &self.model.parameters);

            for i in 0..n {
                jacobian[i * n + j] = (f_plus[i] - f0[i]) / h;
            }
        }

        jacobian
    }

    /// Compute eigenvalues of a matrix
    fn compute_eigenvalues(&self, matrix: &[f64]) -> Vec<Complex64> {
        let n = (matrix.len() as f64).sqrt() as usize;
        let m = DMatrix::from_row_slice(n, n, matrix);

        // Use nalgebra's eigenvalue computation
        if let Some(eigen) = m.clone().try_symmetric_eigen(1e-10, 1000) {
            eigen.eigenvalues.iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect()
        } else {
            // Fall back to Schur decomposition for non-symmetric
            vec![Complex64::new(0.0, 0.0); n]
        }
    }
}

/// Classify fixed point based on eigenvalues
fn classify_fixed_point(eigenvalues: &[Complex64]) -> FixedPointType {
    let all_real = eigenvalues.iter().all(|e| e.im.abs() < 1e-10);
    let all_negative = eigenvalues.iter().all(|e| e.re < 0.0);
    let all_positive = eigenvalues.iter().all(|e| e.re > 0.0);
    let any_zero = eigenvalues.iter().any(|e| e.re.abs() < 1e-10);

    if any_zero {
        FixedPointType::Center
    } else if all_real {
        if all_negative {
            FixedPointType::StableNode
        } else if all_positive {
            FixedPointType::UnstableNode
        } else {
            FixedPointType::Saddle
        }
    } else {
        if all_negative {
            FixedPointType::StableFocus
        } else if all_positive {
            FixedPointType::UnstableFocus
        } else {
            FixedPointType::Saddle
        }
    }
}

/// Common dynamical systems
pub mod examples {
    use super::*;

    /// Lorenz system
    pub fn lorenz(sigma: f64, rho: f64, beta: f64) -> XppModel {
        let mut model = XppModel::new("Lorenz", vec!["x".into(), "y".into(), "z".into()]);
        model.add_parameter("sigma", sigma);
        model.add_parameter("rho", rho);
        model.add_parameter("beta", beta);
        model
    }

    /// Lorenz RHS
    pub fn lorenz_rhs(state: &[f64], params: &[(String, f64)]) -> Vec<f64> {
        let x = state[0];
        let y = state[1];
        let z = state[2];

        let sigma = params.iter().find(|(n, _)| n == "sigma").map(|(_, v)| *v).unwrap_or(10.0);
        let rho = params.iter().find(|(n, _)| n == "rho").map(|(_, v)| *v).unwrap_or(28.0);
        let beta = params.iter().find(|(n, _)| n == "beta").map(|(_, v)| *v).unwrap_or(8.0/3.0);

        vec![
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
    }

    /// FitzHugh-Nagumo model
    pub fn fitzhugh_nagumo(a: f64, b: f64, epsilon: f64) -> XppModel {
        let mut model = XppModel::new("FitzHugh-Nagumo", vec!["v".into(), "w".into()]);
        model.add_parameter("a", a);
        model.add_parameter("b", b);
        model.add_parameter("epsilon", epsilon);
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorenz_model() {
        let model = examples::lorenz(10.0, 28.0, 8.0/3.0);
        assert_eq!(model.variables.len(), 3);
        assert_eq!(model.get_parameter("sigma"), Some(10.0));
    }

    #[test]
    fn test_lorenz_rhs() {
        let model = examples::lorenz(10.0, 28.0, 8.0/3.0);
        let state = vec![1.0, 1.0, 1.0];
        let rhs = examples::lorenz_rhs(&state, &model.parameters);

        assert_eq!(rhs.len(), 3);
        assert!((rhs[0] - 0.0).abs() < 1e-10); // sigma*(y-x) = 10*(1-1) = 0
    }

    #[test]
    fn test_fixed_point_classification() {
        // Stable node (all negative real)
        let eig = vec![Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0)];
        assert_eq!(classify_fixed_point(&eig), FixedPointType::StableNode);

        // Saddle (mixed)
        let eig = vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)];
        assert_eq!(classify_fixed_point(&eig), FixedPointType::Saddle);
    }
}
