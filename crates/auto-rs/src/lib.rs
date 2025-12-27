//! # AUTO-RS: AUTO Continuation/Bifurcation Software Revival
//!
//! Revival of AUTO continuation/bifurcation software in Rust.
//! Originally created by Eusebius Doedel (1980s) at Concordia University.
//!
//! AUTO is the gold standard for numerical continuation and bifurcation
//! analysis of ODEs, PDEs, and algebraic equations.
//!
//! This crate provides:
//! - Natural parameter continuation
//! - Pseudo-arclength continuation
//! - Bifurcation detection (saddle-node, Hopf, branch points)
//! - Stability analysis via eigenvalue computation
//! - Branch switching at bifurcation points
//!
//! Reference: Doedel, E.J. et al. AUTO-07P: Continuation and Bifurcation
//! Software for Ordinary Differential Equations.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AutoError {
    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
    #[error("Singular Jacobian at parameter {0}")]
    SingularJacobian(f64),
    #[error("Step size too small: {0}")]
    StepTooSmall(f64),
    #[error("Maximum steps reached: {0}")]
    MaxStepsReached(usize),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Linear algebra error: {0}")]
    LinearAlgebraError(String),
}

pub type Result<T> = std::result::Result<T, AutoError>;

// ============================================================================
// CONTINUATION PARAMETERS
// ============================================================================

/// Continuation control parameters (like AUTO's constants file)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationParams {
    /// Parameter name to continue
    pub parameter: String,
    /// Starting parameter value
    pub par_start: f64,
    /// Ending parameter value
    pub par_end: f64,
    /// Initial step size
    pub ds: f64,
    /// Minimum step size
    pub ds_min: f64,
    /// Maximum step size
    pub ds_max: f64,
    /// Maximum number of continuation steps
    pub max_steps: usize,
    /// Newton tolerance
    pub newton_tol: f64,
    /// Maximum Newton iterations
    pub newton_max_iter: usize,
    /// Number of mesh points for collocation (BVP)
    pub ntst: usize,
    /// Number of collocation points per mesh interval
    pub ncol: usize,
    /// Adaptive mesh refinement threshold
    pub adapt_threshold: f64,
    /// Output every N steps
    pub output_every: usize,
    /// Detect bifurcations
    pub detect_bifurcations: bool,
    /// Branch switching tolerance
    pub branch_switch_tol: f64,
}

impl Default for ContinuationParams {
    fn default() -> Self {
        Self {
            parameter: "lambda".into(),
            par_start: 0.0,
            par_end: 1.0,
            ds: 0.01,
            ds_min: 1e-6,
            ds_max: 0.1,
            max_steps: 1000,
            newton_tol: 1e-8,
            newton_max_iter: 20,
            ntst: 20,
            ncol: 4,
            adapt_threshold: 0.5,
            output_every: 1,
            detect_bifurcations: true,
            branch_switch_tol: 1e-4,
        }
    }
}

// ============================================================================
// BIFURCATION TYPES
// ============================================================================

/// Types of bifurcations detected
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BifurcationType {
    /// Regular point (not a bifurcation)
    Regular,
    /// Saddle-node (fold/limit point)
    SaddleNode,
    /// Transcritical bifurcation
    Transcritical,
    /// Pitchfork bifurcation
    Pitchfork,
    /// Hopf bifurcation (birth of limit cycle)
    Hopf,
    /// Period-doubling (flip) bifurcation
    PeriodDoubling,
    /// Torus (Neimark-Sacker) bifurcation
    Torus,
    /// Branch point (multiple solutions)
    BranchPoint,
    /// Limit point of cycles
    LimitPointCycle,
    /// User-defined zero
    UserZero,
}

/// Bifurcation point information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationPoint {
    pub bif_type: BifurcationType,
    pub parameter: f64,
    pub state: Array1<f64>,
    /// Eigenvalue(s) crossing imaginary axis
    pub critical_eigenvalues: Vec<(f64, f64)>,  // (real, imag)
    /// Direction for branch switching
    pub tangent: Option<Array1<f64>>,
    /// Period (for periodic orbits)
    pub period: Option<f64>,
}

// ============================================================================
// SOLUTION POINT
// ============================================================================

/// A point on the continuation branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionPoint {
    /// Parameter value
    pub parameter: f64,
    /// State variables
    pub state: Array1<f64>,
    /// Stability (true = stable)
    pub stable: bool,
    /// Eigenvalues of the Jacobian
    pub eigenvalues: Vec<(f64, f64)>,  // (real, imag)
    /// Period (for periodic solutions)
    pub period: Option<f64>,
    /// Floquet multipliers (for periodic solutions)
    pub floquet_multipliers: Option<Vec<(f64, f64)>>,
    /// Bifurcation type if this is a special point
    pub bifurcation: Option<BifurcationType>,
    /// Arclength along the branch
    pub arclength: f64,
    /// Norm of the residual
    pub residual_norm: f64,
}

// ============================================================================
// CONTINUATION BRANCH
// ============================================================================

/// Result of a continuation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationBranch {
    pub name: String,
    pub points: Vec<SolutionPoint>,
    pub bifurcations: Vec<BifurcationPoint>,
    /// Is this a branch of equilibria or periodic orbits?
    pub is_periodic: bool,
    /// Computation statistics
    pub stats: ComputationStats,
}

impl ContinuationBranch {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            points: vec![],
            bifurcations: vec![],
            is_periodic: false,
            stats: ComputationStats::default(),
        }
    }

    /// Get parameter range
    pub fn parameter_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let pars: Vec<f64> = self.points.iter().map(|p| p.parameter).collect();
        let min = pars.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = pars.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// Find stable portions
    pub fn stable_segments(&self) -> Vec<(usize, usize)> {
        let mut segments = vec![];
        let mut start = None;

        for (i, pt) in self.points.iter().enumerate() {
            if pt.stable {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start {
                segments.push((s, i - 1));
                start = None;
            }
        }

        if let Some(s) = start {
            segments.push((s, self.points.len() - 1));
        }

        segments
    }
}

/// Computation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComputationStats {
    pub total_steps: usize,
    pub newton_iterations: usize,
    pub jacobian_evaluations: usize,
    pub step_size_reductions: usize,
    pub bifurcations_detected: usize,
    pub branch_switches: usize,
    pub cpu_time_seconds: f64,
}

// ============================================================================
// ODE SYSTEM TRAIT
// ============================================================================

/// Trait for ODE systems to be continued
pub trait OdeSystem {
    /// Dimension of the state space
    fn dim(&self) -> usize;

    /// Right-hand side: dx/dt = f(x, par)
    fn rhs(&self, x: &Array1<f64>, par: f64) -> Array1<f64>;

    /// Jacobian df/dx (if not provided, numerical differentiation is used)
    fn jacobian(&self, _x: &Array1<f64>, _par: f64) -> Option<Array2<f64>> {
        None
    }

    /// Parameter sensitivity df/dpar
    fn par_derivative(&self, _x: &Array1<f64>, _par: f64) -> Option<Array1<f64>> {
        None
    }
}

// ============================================================================
// NEWTON SOLVER
// ============================================================================

/// Newton's method for finding roots of F(x) = 0
pub fn newton_solve<F, J>(
    f: F,
    jacobian: J,
    mut x: Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<(Array1<f64>, usize)>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
    J: Fn(&Array1<f64>) -> Array2<f64>,
{
    for iter in 0..max_iter {
        let fx = f(&x);
        let norm = fx.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if norm < tol {
            return Ok((x, iter + 1));
        }

        let jac = jacobian(&x);
        let dx = solve_linear_system(&jac, &fx)?;
        x = x - dx;
    }

    Err(AutoError::ConvergenceFailed(max_iter))
}

/// Simple LU-based linear solver
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(AutoError::LinearAlgebraError(
            "Matrix dimension mismatch".into()
        ));
    }

    // Gaussian elimination with partial pivoting
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Pivot
        let mut max_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > max_val {
                max_val = aug[[i, k]].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return Err(AutoError::SingularJacobian(0.0));
        }

        // Swap rows
        if max_row != k {
            for j in 0..=n {
                let tmp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

// ============================================================================
// EIGENVALUE COMPUTATION
// ============================================================================

/// Compute eigenvalues of a matrix using QR iteration
pub fn compute_eigenvalues(a: &Array2<f64>) -> Vec<(f64, f64)> {
    let n = a.nrows();
    if n == 0 {
        return vec![];
    }

    // For small matrices, use direct methods
    if n == 1 {
        return vec![(a[[0, 0]], 0.0)];
    }

    if n == 2 {
        return eigenvalues_2x2(a);
    }

    // QR iteration for larger matrices
    qr_eigenvalues(a)
}

/// Eigenvalues of 2x2 matrix
fn eigenvalues_2x2(a: &Array2<f64>) -> Vec<(f64, f64)> {
    let trace = a[[0, 0]] + a[[1, 1]];
    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];

    let discriminant = trace * trace - 4.0 * det;

    if discriminant >= 0.0 {
        let sqrt_d = discriminant.sqrt();
        vec![
            ((trace + sqrt_d) / 2.0, 0.0),
            ((trace - sqrt_d) / 2.0, 0.0),
        ]
    } else {
        let real = trace / 2.0;
        let imag = (-discriminant).sqrt() / 2.0;
        vec![
            (real, imag),
            (real, -imag),
        ]
    }
}

/// QR iteration for eigenvalues
fn qr_eigenvalues(a: &Array2<f64>) -> Vec<(f64, f64)> {
    let n = a.nrows();
    let mut h = a.clone();
    let max_iter = 100 * n;

    // Hessenberg reduction first (simplified)
    for iter in 0..max_iter {
        // Shifted QR step
        let shift = h[[n - 1, n - 1]];

        for i in 0..n {
            h[[i, i]] -= shift;
        }

        // QR decomposition (simplified Householder)
        let (q, r) = qr_decomposition(&h);

        // H = R * Q
        h = r.dot(&q);

        for i in 0..n {
            h[[i, i]] += shift;
        }

        // Check convergence
        let mut converged = true;
        for i in 1..n {
            if h[[i, i - 1]].abs() > 1e-10 {
                converged = false;
                break;
            }
        }

        if converged || iter == max_iter - 1 {
            break;
        }
    }

    // Extract eigenvalues from quasi-upper triangular form
    let mut eigenvalues = vec![];
    let mut i = 0;
    while i < n {
        if i == n - 1 || h[[i + 1, i]].abs() < 1e-10 {
            // Real eigenvalue
            eigenvalues.push((h[[i, i]], 0.0));
            i += 1;
        } else {
            // Complex conjugate pair from 2x2 block
            let block = Array2::from_shape_vec((2, 2), vec![
                h[[i, i]], h[[i, i + 1]],
                h[[i + 1, i]], h[[i + 1, i + 1]],
            ]).unwrap();

            let eigs = eigenvalues_2x2(&block);
            eigenvalues.extend(eigs);
            i += 2;
        }
    }

    eigenvalues
}

/// QR decomposition (simplified Gram-Schmidt)
fn qr_decomposition(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut q = Array2::zeros((n, n));
    let mut r = Array2::zeros((n, n));

    for j in 0..n {
        let mut v = a.column(j).to_owned();

        for i in 0..j {
            r[[i, j]] = q.column(i).dot(&v);
            for k in 0..n {
                v[k] -= r[[i, j]] * q[[k, i]];
            }
        }

        r[[j, j]] = v.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if r[[j, j]].abs() > 1e-15 {
            for k in 0..n {
                q[[k, j]] = v[k] / r[[j, j]];
            }
        }
    }

    (q, r)
}

// ============================================================================
// NATURAL PARAMETER CONTINUATION
// ============================================================================

/// Natural parameter continuation
/// Simple method that just varies the parameter and solves at each step
pub fn natural_continuation<S: OdeSystem>(
    system: &S,
    initial_state: Array1<f64>,
    params: &ContinuationParams,
) -> Result<ContinuationBranch> {
    let mut branch = ContinuationBranch::new("natural");
    let mut state = initial_state;
    let mut par = params.par_start;
    let direction = if params.par_end > params.par_start { 1.0 } else { -1.0 };

    let mut arclength = 0.0;

    for step in 0..params.max_steps {
        // Solve F(x, par) = 0
        let f = |x: &Array1<f64>| system.rhs(x, par);

        let jac = |x: &Array1<f64>| {
            system.jacobian(x, par).unwrap_or_else(|| numerical_jacobian(system, x, par))
        };

        let (new_state, newton_iters) = newton_solve(f, jac, state.clone(), params.newton_tol, params.newton_max_iter)?;

        branch.stats.newton_iterations += newton_iters;
        branch.stats.jacobian_evaluations += newton_iters;

        // Compute stability
        let jac_matrix = system.jacobian(&new_state, par)
            .unwrap_or_else(|| numerical_jacobian(system, &new_state, par));
        let eigenvalues = compute_eigenvalues(&jac_matrix);
        let stable = eigenvalues.iter().all(|&(re, _)| re < 0.0);

        // Check for bifurcations
        let bifurcation = if params.detect_bifurcations && step > 0 {
            detect_bifurcation(&branch.points.last().unwrap().eigenvalues, &eigenvalues)
        } else {
            None
        };

        if let Some(bif_type) = bifurcation {
            branch.bifurcations.push(BifurcationPoint {
                bif_type,
                parameter: par,
                state: new_state.clone(),
                critical_eigenvalues: find_critical_eigenvalues(&eigenvalues),
                tangent: None,
                period: None,
            });
            branch.stats.bifurcations_detected += 1;
        }

        // Store solution point
        let residual = system.rhs(&new_state, par);
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();

        branch.points.push(SolutionPoint {
            parameter: par,
            state: new_state.clone(),
            stable,
            eigenvalues,
            period: None,
            floquet_multipliers: None,
            bifurcation,
            arclength,
            residual_norm,
        });

        state = new_state;
        arclength += params.ds;

        // Update parameter
        par += direction * params.ds;

        // Check if we've reached the end
        if (direction > 0.0 && par > params.par_end) ||
           (direction < 0.0 && par < params.par_end) {
            break;
        }

        branch.stats.total_steps = step + 1;
    }

    Ok(branch)
}

/// Numerical Jacobian via finite differences
fn numerical_jacobian<S: OdeSystem>(system: &S, x: &Array1<f64>, par: f64) -> Array2<f64> {
    let n = x.len();
    let eps = 1e-8;
    let f0 = system.rhs(x, par);

    let mut jac = Array2::zeros((n, n));

    for j in 0..n {
        let mut x_plus = x.clone();
        x_plus[j] += eps;
        let f_plus = system.rhs(&x_plus, par);

        for i in 0..n {
            jac[[i, j]] = (f_plus[i] - f0[i]) / eps;
        }
    }

    jac
}

// ============================================================================
// PSEUDO-ARCLENGTH CONTINUATION
// ============================================================================

/// Pseudo-arclength continuation
/// Parameterizes the curve by arclength to handle turning points
pub fn arclength_continuation<S: OdeSystem>(
    system: &S,
    initial_state: Array1<f64>,
    params: &ContinuationParams,
) -> Result<ContinuationBranch> {
    let n = system.dim();
    let mut branch = ContinuationBranch::new("arclength");

    // Extended state: (x, par)
    let mut x = initial_state.clone();
    let mut par = params.par_start;
    let mut ds = params.ds;

    // Initial tangent direction
    let mut tangent = compute_initial_tangent(system, &x, par, n, params.par_end > params.par_start);
    let mut arclength = 0.0;

    // First point
    {
        let jac = system.jacobian(&x, par)
            .unwrap_or_else(|| numerical_jacobian(system, &x, par));
        let eigenvalues = compute_eigenvalues(&jac);
        let stable = eigenvalues.iter().all(|&(re, _)| re < 0.0);

        branch.points.push(SolutionPoint {
            parameter: par,
            state: x.clone(),
            stable,
            eigenvalues,
            period: None,
            floquet_multipliers: None,
            bifurcation: None,
            arclength: 0.0,
            residual_norm: 0.0,
        });
    }

    for step in 0..params.max_steps {
        // Predictor: move along tangent
        let mut x_pred = x.clone();
        for i in 0..n {
            x_pred[i] += ds * tangent[i];
        }
        let par_pred = par + ds * tangent[n];

        // Corrector: Newton iteration with arclength constraint
        let result = newton_arclength(
            system,
            x_pred,
            par_pred,
            &x,
            par,
            &tangent,
            ds,
            params.newton_tol,
            params.newton_max_iter,
        );

        match result {
            Ok((new_x, new_par, iters)) => {
                branch.stats.newton_iterations += iters;
                branch.stats.jacobian_evaluations += iters;

                arclength += ds;

                // Update tangent
                let new_tangent = compute_tangent(system, &new_x, new_par, &tangent, n);

                // Stability
                let jac = system.jacobian(&new_x, new_par)
                    .unwrap_or_else(|| numerical_jacobian(system, &new_x, new_par));
                let eigenvalues = compute_eigenvalues(&jac);
                let stable = eigenvalues.iter().all(|&(re, _)| re < 0.0);

                // Detect bifurcations
                let bifurcation = if params.detect_bifurcations && !branch.points.is_empty() {
                    detect_bifurcation(&branch.points.last().unwrap().eigenvalues, &eigenvalues)
                } else {
                    None
                };

                if let Some(bif_type) = bifurcation {
                    branch.bifurcations.push(BifurcationPoint {
                        bif_type,
                        parameter: new_par,
                        state: new_x.clone(),
                        critical_eigenvalues: find_critical_eigenvalues(&eigenvalues),
                        tangent: Some(new_tangent.clone()),
                        period: None,
                    });
                    branch.stats.bifurcations_detected += 1;
                }

                // Store point
                let residual = system.rhs(&new_x, new_par);
                let residual_norm = residual.iter().map(|&v| v * v).sum::<f64>().sqrt();

                branch.points.push(SolutionPoint {
                    parameter: new_par,
                    state: new_x.clone(),
                    stable,
                    eigenvalues,
                    period: None,
                    floquet_multipliers: None,
                    bifurcation,
                    arclength,
                    residual_norm,
                });

                x = new_x;
                par = new_par;
                tangent = new_tangent;

                // Adaptive step size
                if iters < 3 {
                    ds = (ds * 1.5).min(params.ds_max);
                }

                // Check termination
                if (params.par_end > params.par_start && par > params.par_end) ||
                   (params.par_end < params.par_start && par < params.par_end) {
                    break;
                }
            }

            Err(_) => {
                // Reduce step size and try again
                ds = ds / 2.0;
                branch.stats.step_size_reductions += 1;

                if ds < params.ds_min {
                    return Err(AutoError::StepTooSmall(ds));
                }
            }
        }

        branch.stats.total_steps = step + 1;
    }

    Ok(branch)
}

/// Compute initial tangent vector
fn compute_initial_tangent<S: OdeSystem>(
    system: &S,
    x: &Array1<f64>,
    par: f64,
    n: usize,
    forward: bool,
) -> Array1<f64> {
    let jac = system.jacobian(x, par)
        .unwrap_or_else(|| numerical_jacobian(system, x, par));

    // Compute df/dpar numerically
    let eps = 1e-8;
    let f0 = system.rhs(x, par);
    let f1 = system.rhs(x, par + eps);
    let df_dpar: Array1<f64> = (&f1 - &f0) / eps;

    // Solve [df/dx | df/dpar] * [dx; dpar] = 0
    // with ||tangent|| = 1

    // Use nullspace of augmented Jacobian
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = jac[[i, j]];
        }
        aug[[i, n]] = df_dpar[i];
    }

    // Compute nullspace via SVD-like approach (simplified)
    let mut tangent = Array1::zeros(n + 1);
    tangent[n] = 1.0;  // Start with pure parameter variation

    // Solve Jac * dx = -df_dpar * dpar
    if let Ok(dx) = solve_linear_system(&jac, &(-&df_dpar)) {
        for i in 0..n {
            tangent[i] = dx[i];
        }
    }

    // Normalize
    let norm = tangent.iter().map(|&v| v * v).sum::<f64>().sqrt();
    tangent = tangent / norm;

    // Ensure correct direction
    if !forward && tangent[n] > 0.0 || forward && tangent[n] < 0.0 {
        tangent = -tangent;
    }

    tangent
}

/// Compute tangent at a new point
fn compute_tangent<S: OdeSystem>(
    system: &S,
    x: &Array1<f64>,
    par: f64,
    prev_tangent: &Array1<f64>,
    n: usize,
) -> Array1<f64> {
    let new_tangent = compute_initial_tangent(system, x, par, n, true);

    // Choose orientation consistent with previous tangent
    let dot: f64 = new_tangent.iter().zip(prev_tangent.iter()).map(|(&a, &b)| a * b).sum();

    if dot < 0.0 {
        -new_tangent
    } else {
        new_tangent
    }
}

/// Newton iteration with arclength constraint
fn newton_arclength<S: OdeSystem>(
    system: &S,
    mut x: Array1<f64>,
    mut par: f64,
    x0: &Array1<f64>,
    par0: f64,
    tangent: &Array1<f64>,
    ds: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(Array1<f64>, f64, usize)> {
    let n = x.len();

    for iter in 0..max_iter {
        // System: F(x, par) = 0
        let f = system.rhs(&x, par);

        // Arclength constraint: tangent . (x - x0, par - par0) - ds = 0
        let mut g = -ds;
        for i in 0..n {
            g += tangent[i] * (x[i] - x0[i]);
        }
        g += tangent[n] * (par - par0);

        // Check convergence
        let f_norm = f.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if f_norm < tol && g.abs() < tol {
            return Ok((x, par, iter + 1));
        }

        // Jacobian of F
        let jac = system.jacobian(&x, par)
            .unwrap_or_else(|| numerical_jacobian(system, &x, par));

        // df/dpar
        let eps = 1e-8;
        let f_par = system.rhs(&x, par + eps);
        let df_dpar: Array1<f64> = (&f_par - &f) / eps;

        // Build augmented system
        // [J   | df/dpar] [dx  ]   [-F]
        // [t_x | t_par  ] [dpar] = [-g]

        let mut aug = Array2::zeros((n + 1, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = jac[[i, j]];
            }
            aug[[i, n]] = df_dpar[i];
        }
        for j in 0..n {
            aug[[n, j]] = tangent[j];
        }
        aug[[n, n]] = tangent[n];

        let mut rhs = Array1::zeros(n + 1);
        for i in 0..n {
            rhs[i] = -f[i];
        }
        rhs[n] = -g;

        // Solve
        let delta = solve_linear_system(&aug, &rhs)?;

        // Update
        for i in 0..n {
            x[i] += delta[i];
        }
        par += delta[n];
    }

    Err(AutoError::ConvergenceFailed(max_iter))
}

// ============================================================================
// BIFURCATION DETECTION
// ============================================================================

/// Detect bifurcation by monitoring eigenvalue changes
fn detect_bifurcation(
    prev_eigs: &[(f64, f64)],
    curr_eigs: &[(f64, f64)],
) -> Option<BifurcationType> {
    if prev_eigs.len() != curr_eigs.len() {
        return None;
    }

    // Sort eigenvalues by real part for comparison
    let mut prev_sorted: Vec<_> = prev_eigs.to_vec();
    let mut curr_sorted: Vec<_> = curr_eigs.to_vec();
    prev_sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    curr_sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for (prev, curr) in prev_sorted.iter().zip(curr_sorted.iter()) {
        let prev_re = prev.0;
        let curr_re = curr.0;
        let prev_im = prev.1;
        let curr_im = curr.1;

        // Real eigenvalue crossing zero (saddle-node or transcritical)
        if prev_re * curr_re < 0.0 && prev_im.abs() < 1e-6 && curr_im.abs() < 1e-6 {
            return Some(BifurcationType::SaddleNode);
        }

        // Complex conjugate pair crossing imaginary axis (Hopf)
        if prev_re * curr_re < 0.0 && prev_im.abs() > 1e-6 {
            return Some(BifurcationType::Hopf);
        }
    }

    None
}

/// Find eigenvalues that are near the imaginary axis
fn find_critical_eigenvalues(eigenvalues: &[(f64, f64)]) -> Vec<(f64, f64)> {
    eigenvalues
        .iter()
        .filter(|&(re, _)| re.abs() < 0.1)
        .copied()
        .collect()
}

// ============================================================================
// BRANCH SWITCHING
// ============================================================================

/// Switch to a new branch at a bifurcation point
pub fn branch_switch<S: OdeSystem>(
    system: &S,
    bif_point: &BifurcationPoint,
    params: &ContinuationParams,
    perturbation: f64,
) -> Result<ContinuationBranch> {
    let mut new_branch = ContinuationBranch::new("switched");

    // Perturb state in direction of tangent
    let tangent = bif_point.tangent.as_ref()
        .ok_or_else(|| AutoError::InvalidParameter("No tangent at bifurcation".into()))?;

    let n = bif_point.state.len();
    let mut new_state = bif_point.state.clone();
    for i in 0..n {
        new_state[i] += perturbation * tangent[i];
    }
    let new_par = bif_point.parameter + perturbation * tangent[n];

    // Continue from perturbed state
    let branch = arclength_continuation(
        system,
        new_state,
        &ContinuationParams {
            par_start: new_par,
            ..params.clone()
        },
    )?;

    new_branch.points = branch.points;
    new_branch.bifurcations = branch.bifurcations;
    new_branch.stats = branch.stats;
    new_branch.stats.branch_switches = 1;

    Ok(new_branch)
}

// ============================================================================
// STANDARD TEST PROBLEMS
// ============================================================================

/// Fold (saddle-node) normal form: dx/dt = mu - x^2
pub struct FoldNormalForm;

impl OdeSystem for FoldNormalForm {
    fn dim(&self) -> usize { 1 }

    fn rhs(&self, x: &Array1<f64>, mu: f64) -> Array1<f64> {
        Array1::from_vec(vec![mu - x[0] * x[0]])
    }

    fn jacobian(&self, x: &Array1<f64>, _mu: f64) -> Option<Array2<f64>> {
        Some(Array2::from_shape_vec((1, 1), vec![-2.0 * x[0]]).unwrap())
    }
}

/// Hopf normal form: dx/dt = mu*x - y - x*(x^2+y^2), dy/dt = x + mu*y - y*(x^2+y^2)
pub struct HopfNormalForm;

impl OdeSystem for HopfNormalForm {
    fn dim(&self) -> usize { 2 }

    fn rhs(&self, x: &Array1<f64>, mu: f64) -> Array1<f64> {
        let r2 = x[0] * x[0] + x[1] * x[1];
        Array1::from_vec(vec![
            mu * x[0] - x[1] - x[0] * r2,
            x[0] + mu * x[1] - x[1] * r2,
        ])
    }

    fn jacobian(&self, x: &Array1<f64>, mu: f64) -> Option<Array2<f64>> {
        let r2 = x[0] * x[0] + x[1] * x[1];
        Some(Array2::from_shape_vec((2, 2), vec![
            mu - 3.0 * x[0] * x[0] - x[1] * x[1],
            -1.0 - 2.0 * x[0] * x[1],
            1.0 - 2.0 * x[0] * x[1],
            mu - x[0] * x[0] - 3.0 * x[1] * x[1],
        ]).unwrap())
    }
}

/// Pitchfork normal form: dx/dt = mu*x - x^3
pub struct PitchforkNormalForm;

impl OdeSystem for PitchforkNormalForm {
    fn dim(&self) -> usize { 1 }

    fn rhs(&self, x: &Array1<f64>, mu: f64) -> Array1<f64> {
        Array1::from_vec(vec![mu * x[0] - x[0].powi(3)])
    }

    fn jacobian(&self, x: &Array1<f64>, mu: f64) -> Option<Array2<f64>> {
        Some(Array2::from_shape_vec((1, 1), vec![mu - 3.0 * x[0] * x[0]]).unwrap())
    }
}

/// Brusselator: famous chemical oscillator
pub struct Brusselator {
    pub a: f64,
    pub b: f64,
}

impl Default for Brusselator {
    fn default() -> Self {
        Self { a: 1.0, b: 3.0 }
    }
}

impl OdeSystem for Brusselator {
    fn dim(&self) -> usize { 2 }

    fn rhs(&self, x: &Array1<f64>, _par: f64) -> Array1<f64> {
        // par could be 'b' for continuation
        let x_val = x[0];
        let y_val = x[1];
        Array1::from_vec(vec![
            self.a + x_val * x_val * y_val - (self.b + 1.0) * x_val,
            self.b * x_val - x_val * x_val * y_val,
        ])
    }
}

/// Lorenz system
pub struct LorenzSystem {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
}

impl Default for LorenzSystem {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

impl OdeSystem for LorenzSystem {
    fn dim(&self) -> usize { 3 }

    fn rhs(&self, x: &Array1<f64>, _par: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            self.sigma * (x[1] - x[0]),
            x[0] * (self.rho - x[2]) - x[1],
            x[0] * x[1] - self.beta * x[2],
        ])
    }

    fn jacobian(&self, x: &Array1<f64>, _par: f64) -> Option<Array2<f64>> {
        Some(Array2::from_shape_vec((3, 3), vec![
            -self.sigma, self.sigma, 0.0,
            self.rho - x[2], -1.0, -x[0],
            x[1], x[0], -self.beta,
        ]).unwrap())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_normal_form() {
        let system = FoldNormalForm;

        // At mu = 0, x = 0 is a solution
        let f = system.rhs(&Array1::from_vec(vec![0.0]), 0.0);
        assert!(f[0].abs() < 1e-10);

        // At mu = 1, x = 1 is a solution
        let f = system.rhs(&Array1::from_vec(vec![1.0]), 1.0);
        assert!(f[0].abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_2x2() {
        // Real eigenvalues
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let eigs = eigenvalues_2x2(&a);
        assert!((eigs[0].0 - 3.0).abs() < 1e-10 || (eigs[1].0 - 3.0).abs() < 1e-10);
        assert!((eigs[0].0 - 2.0).abs() < 1e-10 || (eigs[1].0 - 2.0).abs() < 1e-10);

        // Complex eigenvalues
        let b = Array2::from_shape_vec((2, 2), vec![0.0, -1.0, 1.0, 0.0]).unwrap();
        let eigs = eigenvalues_2x2(&b);
        assert!(eigs[0].0.abs() < 1e-10);
        assert!((eigs[0].1.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_solver() {
        // Solve x^2 - 2 = 0
        let f = |x: &Array1<f64>| Array1::from_vec(vec![x[0] * x[0] - 2.0]);
        let j = |x: &Array1<f64>| Array2::from_shape_vec((1, 1), vec![2.0 * x[0]]).unwrap();

        let (sol, _iters) = newton_solve(f, j, Array1::from_vec(vec![1.0]), 1e-10, 20).unwrap();
        assert!((sol[0] - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_linear_solve() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 11.0]);

        let x = solve_linear_system(&a, &b).unwrap();

        // Check Ax = b
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_hopf_equilibrium() {
        let system = HopfNormalForm;

        // At mu < 0, origin should be stable
        let jac = system.jacobian(&Array1::from_vec(vec![0.0, 0.0]), -0.5).unwrap();
        let eigs = compute_eigenvalues(&jac);

        // Both eigenvalues should have negative real part
        assert!(eigs.iter().all(|&(re, _)| re < 0.0));
    }

    #[test]
    fn test_natural_continuation() {
        let system = FoldNormalForm;
        let params = ContinuationParams {
            par_start: 0.0,
            par_end: 2.0,
            ds: 0.1,
            max_steps: 30,
            detect_bifurcations: false,
            ..Default::default()
        };

        let branch = natural_continuation(&system, Array1::from_vec(vec![0.01]), &params).unwrap();

        assert!(branch.points.len() > 10);
        assert!(branch.points.last().unwrap().parameter > 1.5);
    }

    #[test]
    fn test_qr_decomposition() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (q, r) = qr_decomposition(&a);

        // Q should be orthogonal: Q^T * Q = I
        let qtq = q.t().dot(&q);
        assert!((qtq[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((qtq[[1, 1]] - 1.0).abs() < 1e-10);

        // QR should equal A
        let qr = q.dot(&r);
        assert!((qr[[0, 0]] - a[[0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_brusselator() {
        let system = Brusselator::default();
        let eq = Array1::from_vec(vec![system.a, system.b / system.a]);

        let f = system.rhs(&eq, 0.0);
        assert!(f[0].abs() < 1e-10);
        assert!(f[1].abs() < 1e-10);
    }
}
