//! # Brian-RS: Brian Spiking Neural Network Simulator Revival
//!
//! Revival of the Brian simulator (http://briansimulator.org/)
//! Originally created by Romain Brette and Dan Goodman (2007)
//!
//! Brian uses equation-based model definitions with natural mathematical syntax.
//! This crate provides:
//! - Equation parser for differential equations
//! - Multiple neuron models (LIF, AdEx, Izhikevich, HH)
//! - Synapse models (exponential, alpha, STDP)
//! - Network topology and connectivity
//! - Spike monitors and state monitors

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BrianError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
    #[error("Invalid equation: {0}")]
    EquationError(String),
    #[error("Unit mismatch: expected {expected}, got {got}")]
    UnitError { expected: String, got: String },
}

pub type Result<T> = std::result::Result<T, BrianError>;

// ============================================================================
// UNITS SYSTEM (Brian's signature feature)
// ============================================================================

/// Physical units with SI prefixes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Unit {
    // Time
    Second,
    Millisecond,  // ms
    Microsecond,  // us

    // Voltage
    Volt,
    Millivolt,    // mV

    // Current
    Ampere,
    Nanoampere,   // nA
    Picoampere,   // pA

    // Conductance
    Siemens,
    Nanosiemens,  // nS
    Microsiemens, // uS

    // Capacitance
    Farad,
    Picofarad,    // pF

    // Resistance
    Ohm,
    Megaohm,      // MOhm
    Gigaohm,      // GOhm

    // Frequency
    Hertz,

    // Dimensionless
    Dimensionless,
}

impl Unit {
    /// Convert to SI base units
    pub fn to_si_factor(&self) -> f64 {
        match self {
            Unit::Second => 1.0,
            Unit::Millisecond => 1e-3,
            Unit::Microsecond => 1e-6,
            Unit::Volt => 1.0,
            Unit::Millivolt => 1e-3,
            Unit::Ampere => 1.0,
            Unit::Nanoampere => 1e-9,
            Unit::Picoampere => 1e-12,
            Unit::Siemens => 1.0,
            Unit::Nanosiemens => 1e-9,
            Unit::Microsiemens => 1e-6,
            Unit::Farad => 1.0,
            Unit::Picofarad => 1e-12,
            Unit::Ohm => 1.0,
            Unit::Megaohm => 1e6,
            Unit::Gigaohm => 1e9,
            Unit::Hertz => 1.0,
            Unit::Dimensionless => 1.0,
        }
    }
}

/// Quantity with value and unit
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Quantity {
    pub value: f64,
    pub unit: Unit,
}

impl Quantity {
    pub fn new(value: f64, unit: Unit) -> Self {
        Self { value, unit }
    }

    /// Convert to SI base units
    pub fn to_si(&self) -> f64 {
        self.value * self.unit.to_si_factor()
    }
}

// ============================================================================
// EQUATION SYSTEM
// ============================================================================

/// Differential equation: dv/dt = expr
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialEquation {
    pub variable: String,
    pub expression: String,
    pub unit: Unit,
    /// Method: euler, rk2, rk4, exponential_euler
    pub method: IntegrationMethod,
}

/// Algebraic equation: v = expr (computed each timestep)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgebraicEquation {
    pub variable: String,
    pub expression: String,
    pub unit: Unit,
}

/// Threshold condition for spike generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdCondition {
    pub condition: String,  // e.g., "v > v_thresh"
}

/// Reset equations after spike
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetEquations {
    pub equations: Vec<String>,  // e.g., ["v = v_reset", "w += b"]
}

/// Refractory period specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefractorySpec {
    Duration(Quantity),           // Fixed duration
    Condition(String),            // Until condition is met
}

/// Integration methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IntegrationMethod {
    Euler,
    ExponentialEuler,  // For linear ODEs
    RungeKutta2,
    RungeKutta4,
    Heun,
    Milstein,  // For SDEs
    ExactSolution,  // For analytically solvable equations
}

/// Complete neuron equations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronEquations {
    pub differential: Vec<DifferentialEquation>,
    pub algebraic: Vec<AlgebraicEquation>,
    pub threshold: Option<ThresholdCondition>,
    pub reset: Option<ResetEquations>,
    pub refractory: Option<RefractorySpec>,
    pub parameters: HashMap<String, Quantity>,
}

// ============================================================================
// NEURON MODELS
// ============================================================================

/// Leaky Integrate-and-Fire neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFNeuron {
    pub tau_m: f64,      // Membrane time constant (ms)
    pub v_rest: f64,     // Resting potential (mV)
    pub v_reset: f64,    // Reset potential (mV)
    pub v_thresh: f64,   // Spike threshold (mV)
    pub r_m: f64,        // Membrane resistance (MOhm)
    pub tau_ref: f64,    // Refractory period (ms)
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            v_rest: -65.0,
            v_reset: -65.0,
            v_thresh: -50.0,
            r_m: 10.0,
            tau_ref: 2.0,
        }
    }
}

impl LIFNeuron {
    pub fn to_equations(&self) -> NeuronEquations {
        NeuronEquations {
            differential: vec![
                DifferentialEquation {
                    variable: "v".into(),
                    expression: format!(
                        "(({} - v) + {} * I) / {}",
                        self.v_rest, self.r_m, self.tau_m
                    ),
                    unit: Unit::Millivolt,
                    method: IntegrationMethod::ExponentialEuler,
                },
            ],
            algebraic: vec![],
            threshold: Some(ThresholdCondition {
                condition: format!("v > {}", self.v_thresh),
            }),
            reset: Some(ResetEquations {
                equations: vec![format!("v = {}", self.v_reset)],
            }),
            refractory: Some(RefractorySpec::Duration(
                Quantity::new(self.tau_ref, Unit::Millisecond)
            )),
            parameters: HashMap::new(),
        }
    }
}

/// Adaptive Exponential Integrate-and-Fire (AdEx)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdExNeuron {
    pub c_m: f64,        // Membrane capacitance (pF)
    pub g_l: f64,        // Leak conductance (nS)
    pub e_l: f64,        // Leak reversal (mV)
    pub v_t: f64,        // Spike initiation threshold (mV)
    pub delta_t: f64,    // Slope factor (mV)
    pub tau_w: f64,      // Adaptation time constant (ms)
    pub a: f64,          // Subthreshold adaptation (nS)
    pub b: f64,          // Spike-triggered adaptation (pA)
    pub v_reset: f64,    // Reset potential (mV)
    pub v_peak: f64,     // Spike cutoff (mV)
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self {
            c_m: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            v_t: -50.4,
            delta_t: 2.0,
            tau_w: 144.0,
            a: 4.0,
            b: 80.5,
            v_reset: -70.6,
            v_peak: 20.0,
        }
    }
}

impl AdExNeuron {
    pub fn to_equations(&self) -> NeuronEquations {
        NeuronEquations {
            differential: vec![
                DifferentialEquation {
                    variable: "v".into(),
                    expression: format!(
                        "(-{} * (v - {}) + {} * {} * exp((v - {}) / {}) - w + I) / {}",
                        self.g_l, self.e_l, self.g_l, self.delta_t,
                        self.v_t, self.delta_t, self.c_m
                    ),
                    unit: Unit::Millivolt,
                    method: IntegrationMethod::Euler,
                },
                DifferentialEquation {
                    variable: "w".into(),
                    expression: format!(
                        "({} * (v - {}) - w) / {}",
                        self.a, self.e_l, self.tau_w
                    ),
                    unit: Unit::Picoampere,
                    method: IntegrationMethod::Euler,
                },
            ],
            algebraic: vec![],
            threshold: Some(ThresholdCondition {
                condition: format!("v > {}", self.v_peak),
            }),
            reset: Some(ResetEquations {
                equations: vec![
                    format!("v = {}", self.v_reset),
                    format!("w += {}", self.b),
                ],
            }),
            refractory: None,
            parameters: HashMap::new(),
        }
    }
}

/// Izhikevich simple model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichNeuron {
    pub a: f64,  // Recovery time scale
    pub b: f64,  // Recovery sensitivity
    pub c: f64,  // Reset potential (mV)
    pub d: f64,  // Recovery reset
}

impl IzhikevichNeuron {
    /// Regular spiking (RS) - typical excitatory cortical neuron
    pub fn regular_spiking() -> Self {
        Self { a: 0.02, b: 0.2, c: -65.0, d: 8.0 }
    }

    /// Intrinsically bursting (IB)
    pub fn intrinsically_bursting() -> Self {
        Self { a: 0.02, b: 0.2, c: -55.0, d: 4.0 }
    }

    /// Chattering (CH)
    pub fn chattering() -> Self {
        Self { a: 0.02, b: 0.2, c: -50.0, d: 2.0 }
    }

    /// Fast spiking (FS) - inhibitory interneuron
    pub fn fast_spiking() -> Self {
        Self { a: 0.1, b: 0.2, c: -65.0, d: 2.0 }
    }

    /// Low-threshold spiking (LTS)
    pub fn low_threshold_spiking() -> Self {
        Self { a: 0.02, b: 0.25, c: -65.0, d: 2.0 }
    }

    pub fn to_equations(&self) -> NeuronEquations {
        NeuronEquations {
            differential: vec![
                DifferentialEquation {
                    variable: "v".into(),
                    expression: "0.04 * v * v + 5.0 * v + 140.0 - u + I".into(),
                    unit: Unit::Millivolt,
                    method: IntegrationMethod::Euler,
                },
                DifferentialEquation {
                    variable: "u".into(),
                    expression: format!("{} * ({} * v - u)", self.a, self.b),
                    unit: Unit::Dimensionless,
                    method: IntegrationMethod::Euler,
                },
            ],
            algebraic: vec![],
            threshold: Some(ThresholdCondition {
                condition: "v >= 30.0".into(),
            }),
            reset: Some(ResetEquations {
                equations: vec![
                    format!("v = {}", self.c),
                    format!("u += {}", self.d),
                ],
            }),
            refractory: None,
            parameters: HashMap::new(),
        }
    }
}

// ============================================================================
// SYNAPSE MODELS
// ============================================================================

/// Synapse model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynapseModel {
    /// Instantaneous (delta function)
    Delta { weight: f64 },

    /// Exponential decay: g(t) = w * exp(-t/tau)
    Exponential {
        weight: f64,
        tau: f64,  // ms
    },

    /// Alpha function: g(t) = w * (t/tau) * exp(1 - t/tau)
    Alpha {
        weight: f64,
        tau: f64,  // ms
    },

    /// Difference of exponentials
    DualExponential {
        weight: f64,
        tau_rise: f64,  // ms
        tau_decay: f64, // ms
    },

    /// NMDA with voltage-dependent Mg block
    NMDA {
        weight: f64,
        tau_rise: f64,
        tau_decay: f64,
        mg_concentration: f64,  // mM
    },

    /// Short-term plasticity (Tsodyks-Markram)
    STP {
        weight: f64,
        u_se: f64,     // Initial utilization
        tau_rec: f64,  // Recovery time constant (ms)
        tau_fac: f64,  // Facilitation time constant (ms)
    },
}

/// Spike-Timing-Dependent Plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPRule {
    pub tau_pre: f64,   // Pre-synaptic trace time constant (ms)
    pub tau_post: f64,  // Post-synaptic trace time constant (ms)
    pub a_plus: f64,    // LTP amplitude
    pub a_minus: f64,   // LTD amplitude
    pub w_max: f64,     // Maximum weight
    pub w_min: f64,     // Minimum weight
}

impl Default for STDPRule {
    fn default() -> Self {
        Self {
            tau_pre: 20.0,
            tau_post: 20.0,
            a_plus: 0.01,
            a_minus: 0.012,  // Slightly stronger LTD
            w_max: 1.0,
            w_min: 0.0,
        }
    }
}

// ============================================================================
// NEURON GROUP
// ============================================================================

/// A group of neurons sharing the same equations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronGroup {
    pub name: String,
    pub n: usize,
    pub equations: NeuronEquations,
    pub method: IntegrationMethod,
    /// State variables for all neurons
    pub state: HashMap<String, Array1<f64>>,
    /// Last spike time for each neuron (-inf if never spiked)
    pub last_spike: Array1<f64>,
    /// Is neuron currently in refractory period?
    pub refractory_until: Array1<f64>,
}

impl NeuronGroup {
    pub fn new(name: &str, n: usize, equations: NeuronEquations) -> Self {
        let mut state = HashMap::new();

        // Initialize state variables
        for eq in &equations.differential {
            state.insert(eq.variable.clone(), Array1::zeros(n));
        }

        Self {
            name: name.to_string(),
            n,
            equations,
            method: IntegrationMethod::Euler,
            state,
            last_spike: Array1::from_elem(n, f64::NEG_INFINITY),
            refractory_until: Array1::from_elem(n, f64::NEG_INFINITY),
        }
    }

    pub fn set_initial(&mut self, variable: &str, values: Array1<f64>) -> Result<()> {
        if let Some(state) = self.state.get_mut(variable) {
            if values.len() != self.n {
                return Err(BrianError::SimulationError(
                    format!("Expected {} values, got {}", self.n, values.len())
                ));
            }
            *state = values;
            Ok(())
        } else {
            Err(BrianError::SimulationError(
                format!("Unknown variable: {}", variable)
            ))
        }
    }
}

// ============================================================================
// SYNAPSES
// ============================================================================

/// Synapse connections between neuron groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapses {
    pub name: String,
    pub source: String,      // Source NeuronGroup name
    pub target: String,      // Target NeuronGroup name
    pub model: SynapseModel,
    pub plasticity: Option<STDPRule>,
    /// Sparse connectivity: (source_idx, target_idx)
    pub connections: Vec<(usize, usize)>,
    /// Weights (same length as connections)
    pub weights: Vec<f64>,
    /// Delays in ms (same length as connections)
    pub delays: Vec<f64>,
}

impl Synapses {
    pub fn new(name: &str, source: &str, target: &str, model: SynapseModel) -> Self {
        Self {
            name: name.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            model,
            plasticity: None,
            connections: vec![],
            weights: vec![],
            delays: vec![],
        }
    }

    /// Connect all-to-all
    pub fn connect_all_to_all(&mut self, n_source: usize, n_target: usize, weight: f64, delay: f64) {
        for i in 0..n_source {
            for j in 0..n_target {
                self.connections.push((i, j));
                self.weights.push(weight);
                self.delays.push(delay);
            }
        }
    }

    /// Connect with probability p
    pub fn connect_random(&mut self, n_source: usize, n_target: usize, p: f64, weight: f64, delay: f64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        for i in 0..n_source {
            for j in 0..n_target {
                let mut hasher = DefaultHasher::new();
                (i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let r = (hash as f64) / (u64::MAX as f64);

                if r < p {
                    self.connections.push((i, j));
                    self.weights.push(weight);
                    self.delays.push(delay);
                }
            }
        }
    }

    /// One-to-one mapping
    pub fn connect_one_to_one(&mut self, n: usize, weight: f64, delay: f64) {
        for i in 0..n {
            self.connections.push((i, i));
            self.weights.push(weight);
            self.delays.push(delay);
        }
    }
}

// ============================================================================
// INPUT DEVICES
// ============================================================================

/// Poisson spike generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonGroup {
    pub name: String,
    pub n: usize,
    pub rates: Array1<f64>,  // Hz
}

impl PoissonGroup {
    pub fn new(name: &str, n: usize, rate: f64) -> Self {
        Self {
            name: name.to_string(),
            n,
            rates: Array1::from_elem(n, rate),
        }
    }

    pub fn new_heterogeneous(name: &str, rates: Array1<f64>) -> Self {
        let n = rates.len();
        Self {
            name: name.to_string(),
            n,
            rates,
        }
    }
}

/// Spike generator from predetermined spike times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeGeneratorGroup {
    pub name: String,
    pub n: usize,
    /// Spike times: (neuron_idx, time_ms)
    pub spike_times: Vec<(usize, f64)>,
}

impl SpikeGeneratorGroup {
    pub fn new(name: &str, n: usize) -> Self {
        Self {
            name: name.to_string(),
            n,
            spike_times: vec![],
        }
    }

    pub fn add_spikes(&mut self, indices: &[usize], times: &[f64]) {
        for (&i, &t) in indices.iter().zip(times.iter()) {
            self.spike_times.push((i, t));
        }
        // Sort by time
        self.spike_times.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }
}

/// Timed array for time-varying input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedArray {
    pub name: String,
    pub times: Array1<f64>,   // ms
    pub values: Array2<f64>,  // (time_points, neurons)
}

// ============================================================================
// MONITORS
// ============================================================================

/// Record spike times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeMonitor {
    pub source: String,
    /// Recorded spikes: (neuron_idx, time_ms)
    pub spikes: Vec<(usize, f64)>,
    /// Spike counts per neuron
    pub counts: Vec<usize>,
}

impl SpikeMonitor {
    pub fn new(source: &str, n: usize) -> Self {
        Self {
            source: source.to_string(),
            spikes: vec![],
            counts: vec![0; n],
        }
    }

    pub fn record_spike(&mut self, idx: usize, time: f64) {
        self.spikes.push((idx, time));
        if idx < self.counts.len() {
            self.counts[idx] += 1;
        }
    }

    /// Get spike trains for each neuron
    pub fn spike_trains(&self) -> HashMap<usize, Vec<f64>> {
        let mut trains: HashMap<usize, Vec<f64>> = HashMap::new();
        for &(idx, time) in &self.spikes {
            trains.entry(idx).or_default().push(time);
        }
        trains
    }

    /// Calculate firing rate in Hz
    pub fn mean_rate(&self, duration_ms: f64) -> f64 {
        if self.counts.is_empty() || duration_ms <= 0.0 {
            return 0.0;
        }
        let total_spikes: usize = self.counts.iter().sum();
        (total_spikes as f64) / (self.counts.len() as f64) / (duration_ms / 1000.0)
    }
}

/// Record state variable over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMonitor {
    pub source: String,
    pub variables: Vec<String>,
    pub record_indices: Vec<usize>,  // Which neurons to record
    pub dt: f64,                     // Recording timestep (ms)
    /// Recorded values: variable -> (times, values[neuron][time])
    pub data: HashMap<String, (Vec<f64>, Vec<Vec<f64>>)>,
}

impl StateMonitor {
    pub fn new(source: &str, variables: &[&str], indices: &[usize], dt: f64) -> Self {
        let mut data = HashMap::new();
        for var in variables {
            data.insert(var.to_string(), (vec![], vec![vec![]; indices.len()]));
        }

        Self {
            source: source.to_string(),
            variables: variables.iter().map(|s| s.to_string()).collect(),
            record_indices: indices.to_vec(),
            dt,
            data,
        }
    }

    pub fn record(&mut self, variable: &str, time: f64, values: &Array1<f64>) {
        if let Some((times, data)) = self.data.get_mut(variable) {
            if times.is_empty() || time >= times.last().unwrap() + self.dt {
                times.push(time);
                for (i, &idx) in self.record_indices.iter().enumerate() {
                    if idx < values.len() {
                        data[i].push(values[idx]);
                    }
                }
            }
        }
    }
}

/// Population rate monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationRateMonitor {
    pub source: String,
    pub bin_size: f64,  // ms
    pub times: Vec<f64>,
    pub rates: Vec<f64>,  // Hz
}

// ============================================================================
// NETWORK
// ============================================================================

/// Complete Brian network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub neuron_groups: HashMap<String, NeuronGroup>,
    pub synapses: HashMap<String, Synapses>,
    pub poisson_groups: HashMap<String, PoissonGroup>,
    pub spike_generators: HashMap<String, SpikeGeneratorGroup>,
    pub spike_monitors: HashMap<String, SpikeMonitor>,
    pub state_monitors: HashMap<String, StateMonitor>,
    pub dt: f64,  // Timestep in ms
    pub t: f64,   // Current time in ms
}

impl Network {
    pub fn new(dt: f64) -> Self {
        Self {
            neuron_groups: HashMap::new(),
            synapses: HashMap::new(),
            poisson_groups: HashMap::new(),
            spike_generators: HashMap::new(),
            spike_monitors: HashMap::new(),
            state_monitors: HashMap::new(),
            dt,
            t: 0.0,
        }
    }

    pub fn add_neuron_group(&mut self, group: NeuronGroup) {
        self.neuron_groups.insert(group.name.clone(), group);
    }

    pub fn add_synapses(&mut self, synapses: Synapses) {
        self.synapses.insert(synapses.name.clone(), synapses);
    }

    pub fn add_poisson_group(&mut self, group: PoissonGroup) {
        self.poisson_groups.insert(group.name.clone(), group);
    }

    pub fn add_spike_monitor(&mut self, monitor: SpikeMonitor) {
        self.spike_monitors.insert(monitor.source.clone(), monitor);
    }

    pub fn add_state_monitor(&mut self, monitor: StateMonitor) {
        self.state_monitors.insert(
            format!("{}_state", monitor.source),
            monitor
        );
    }

    /// Run simulation for given duration
    pub fn run(&mut self, duration: f64) -> Result<()> {
        let n_steps = (duration / self.dt).ceil() as usize;

        for _ in 0..n_steps {
            self.step()?;
        }

        Ok(())
    }

    /// Single simulation step
    fn step(&mut self) -> Result<()> {
        // Update time
        self.t += self.dt;

        // For now, basic Euler integration (placeholder for full implementation)
        for (_name, group) in &mut self.neuron_groups {
            // Simple integration of state variables would go here
            // This is a skeleton - full implementation would parse and evaluate equations
            let _n = group.n;
        }

        Ok(())
    }
}

// ============================================================================
// BRIAN SCRIPT PARSER (simplified)
// ============================================================================

/// Parse Brian-style equations
pub fn parse_equations(text: &str) -> Result<NeuronEquations> {
    let mut differential = vec![];
    let mut algebraic = vec![];

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Differential equation: dv/dt = expr : unit
        if line.starts_with('d') && line.contains("/dt") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let var_part = parts[0].trim();
                let var = var_part
                    .trim_start_matches('d')
                    .split("/dt")
                    .next()
                    .unwrap_or("")
                    .trim();

                let expr_parts: Vec<&str> = parts[1].split(':').collect();
                let expr = expr_parts[0].trim();

                differential.push(DifferentialEquation {
                    variable: var.to_string(),
                    expression: expr.to_string(),
                    unit: Unit::Dimensionless,
                    method: IntegrationMethod::Euler,
                });
            }
        }
        // Algebraic equation: v = expr : unit
        else if line.contains('=') && !line.contains("/dt") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                let var = parts[0].trim();
                let expr_parts: Vec<&str> = parts[1].split(':').collect();
                let expr = expr_parts[0].trim();

                algebraic.push(AlgebraicEquation {
                    variable: var.to_string(),
                    expression: expr.to_string(),
                    unit: Unit::Dimensionless,
                });
            }
        }
    }

    Ok(NeuronEquations {
        differential,
        algebraic,
        threshold: None,
        reset: None,
        refractory: None,
        parameters: HashMap::new(),
    })
}

// ============================================================================
// STANDARD MODELS
// ============================================================================

/// Create a balanced E/I network (Brunel 2000)
pub fn brunel_network(
    n_exc: usize,
    n_inh: usize,
    g: f64,      // Relative inhibitory strength
    eta: f64,    // External rate relative to threshold
    dt: f64,
) -> Network {
    let mut network = Network::new(dt);

    // LIF parameters
    let lif = LIFNeuron::default();

    // Excitatory neurons
    let mut exc = NeuronGroup::new("E", n_exc, lif.to_equations());
    exc.set_initial("v", Array1::from_elem(n_exc, -65.0)).ok();
    network.add_neuron_group(exc);

    // Inhibitory neurons
    let mut inh = NeuronGroup::new("I", n_inh, lif.to_equations());
    inh.set_initial("v", Array1::from_elem(n_inh, -65.0)).ok();
    network.add_neuron_group(inh);

    // Synapses
    let w_exc = 0.1;  // mV
    let w_inh = -g * w_exc;
    let _delay = 1.5;  // ms

    let p_conn = 0.1;  // Connection probability

    // E -> E
    let mut ee = Synapses::new("EE", "E", "E", SynapseModel::Delta { weight: w_exc });
    ee.connect_random(n_exc, n_exc, p_conn, w_exc, 1.5);
    network.add_synapses(ee);

    // E -> I
    let mut ei = Synapses::new("EI", "E", "I", SynapseModel::Delta { weight: w_exc });
    ei.connect_random(n_exc, n_inh, p_conn, w_exc, 1.5);
    network.add_synapses(ei);

    // I -> E
    let mut ie = Synapses::new("IE", "I", "E", SynapseModel::Delta { weight: w_inh });
    ie.connect_random(n_inh, n_exc, p_conn, w_inh, 1.5);
    network.add_synapses(ie);

    // I -> I
    let mut ii = Synapses::new("II", "I", "I", SynapseModel::Delta { weight: w_inh });
    ii.connect_random(n_inh, n_inh, p_conn, w_inh, 1.5);
    network.add_synapses(ii);

    // External Poisson input
    let nu_thresh = lif.v_thresh / (lif.r_m * lif.tau_m);  // Threshold rate
    let nu_ext = eta * nu_thresh * 1000.0;  // Hz

    network.add_poisson_group(PoissonGroup::new("ext_E", n_exc, nu_ext));
    network.add_poisson_group(PoissonGroup::new("ext_I", n_inh, nu_ext));

    // Monitors
    network.add_spike_monitor(SpikeMonitor::new("E", n_exc));
    network.add_spike_monitor(SpikeMonitor::new("I", n_inh));

    network
}

/// CUBA (Current-based) network from Brian examples
pub fn cuba_network(n: usize, dt: f64) -> Network {
    let n_exc = (0.8 * n as f64) as usize;
    let n_inh = n - n_exc;

    brunel_network(n_exc, n_inh, 5.0, 2.0, dt)
}

/// COBA (Conductance-based) LIF network
pub fn coba_network(_n: usize, dt: f64) -> Network {
    // Simplified implementation
    Network::new(dt)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_equations() {
        let lif = LIFNeuron::default();
        let eqs = lif.to_equations();

        assert_eq!(eqs.differential.len(), 1);
        assert_eq!(eqs.differential[0].variable, "v");
        assert!(eqs.threshold.is_some());
        assert!(eqs.reset.is_some());
    }

    #[test]
    fn test_adex_equations() {
        let adex = AdExNeuron::default();
        let eqs = adex.to_equations();

        assert_eq!(eqs.differential.len(), 2);
        assert_eq!(eqs.differential[0].variable, "v");
        assert_eq!(eqs.differential[1].variable, "w");
    }

    #[test]
    fn test_izhikevich_types() {
        let rs = IzhikevichNeuron::regular_spiking();
        let fs = IzhikevichNeuron::fast_spiking();

        assert!(rs.a < fs.a);  // FS has faster recovery
    }

    #[test]
    fn test_synapse_connectivity() {
        let mut syn = Synapses::new("test", "A", "B", SynapseModel::Delta { weight: 1.0 });
        syn.connect_all_to_all(3, 4, 1.0, 1.0);

        assert_eq!(syn.connections.len(), 12);  // 3 * 4
    }

    #[test]
    fn test_neuron_group() {
        let lif = LIFNeuron::default();
        let mut group = NeuronGroup::new("test", 100, lif.to_equations());

        assert_eq!(group.n, 100);
        assert!(group.state.contains_key("v"));

        group.set_initial("v", Array1::from_elem(100, -70.0)).unwrap();
        assert_eq!(group.state["v"][0], -70.0);
    }

    #[test]
    fn test_spike_monitor() {
        let mut monitor = SpikeMonitor::new("test", 10);
        monitor.record_spike(0, 10.0);
        monitor.record_spike(0, 20.0);
        monitor.record_spike(1, 15.0);

        assert_eq!(monitor.counts[0], 2);
        assert_eq!(monitor.counts[1], 1);
        assert_eq!(monitor.spikes.len(), 3);
    }

    #[test]
    fn test_parse_equations() {
        let text = r#"
            dv/dt = (v_rest - v) / tau : volt
            dw/dt = a * (v - v_rest) : amp
        "#;

        let eqs = parse_equations(text).unwrap();
        assert_eq!(eqs.differential.len(), 2);
    }

    #[test]
    fn test_brunel_network() {
        let net = brunel_network(80, 20, 5.0, 2.0, 0.1);

        assert!(net.neuron_groups.contains_key("E"));
        assert!(net.neuron_groups.contains_key("I"));
        assert_eq!(net.neuron_groups["E"].n, 80);
        assert_eq!(net.neuron_groups["I"].n, 20);
    }

    #[test]
    fn test_stdp_rule() {
        let stdp = STDPRule::default();

        assert!(stdp.a_minus > stdp.a_plus);  // Slight LTD dominance
        assert_eq!(stdp.tau_pre, stdp.tau_post);
    }
}
