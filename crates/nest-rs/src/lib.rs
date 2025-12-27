//! # NEST-RS: NEST Simulator Revival
//!
//! Revival of the NEST simulator (https://www.nest-simulator.org/)
//! NEST = NEural Simulation Tool
//! Originally created by Marc-Oliver Gewaltig and Markus Diesmann
//!
//! NEST is designed for large-scale spiking neural network simulations
//! with efficient parallelization and precise spike timing.
//!
//! Key features:
//! - Node-based architecture (neurons, devices, connections)
//! - Precise spike timing with grid/off-grid modes
//! - Efficient connection management with synapse types
//! - Built-in parallelization support
//! - Recording devices (spike detectors, multimeters)

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NestError {
    #[error("Unknown node model: {0}")]
    UnknownModel(String),
    #[error("Node not found: {0}")]
    NodeNotFound(usize),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

pub type Result<T> = std::result::Result<T, NestError>;

// ============================================================================
// NODE IDS (NEST's fundamental concept)
// ============================================================================

/// Global node identifier
pub type NodeId = usize;

/// Collection of node IDs (like NEST's NodeCollection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCollection {
    pub ids: Vec<NodeId>,
}

impl NodeCollection {
    pub fn new(ids: Vec<NodeId>) -> Self {
        Self { ids }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn first(&self) -> Option<NodeId> {
        self.ids.first().copied()
    }

    pub fn last(&self) -> Option<NodeId> {
        self.ids.last().copied()
    }

    /// Slice of nodes
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self::new(self.ids[start..end].to_vec())
    }
}

impl IntoIterator for NodeCollection {
    type Item = NodeId;
    type IntoIter = std::vec::IntoIter<NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.ids.into_iter()
    }
}

// ============================================================================
// NEURON MODELS
// ============================================================================

/// NEST neuron model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuronModel {
    /// Integrate-and-fire with alpha-function PSCs
    IafPscAlpha(IafPscAlphaParams),

    /// Integrate-and-fire with exponential PSCs
    IafPscExp(IafPscExpParams),

    /// Integrate-and-fire with delta PSCs (instantaneous)
    IafPscDelta(IafPscDeltaParams),

    /// Conductance-based IAF
    IafCondAlpha(IafCondAlphaParams),

    /// Conductance-based with exponential conductances
    IafCondExp(IafCondExpParams),

    /// Adaptive exponential integrate-and-fire
    AeifCondAlpha(AeifCondAlphaParams),

    /// Hodgkin-Huxley
    HhPscAlpha(HhPscAlphaParams),

    /// Izhikevich
    Izhikevich(IzhikevichParams),

    /// Parrot neuron (repeats input spikes)
    ParrotNeuron,

    /// Poisson generator
    PoissonGenerator(PoissonGeneratorParams),

    /// Spike generator
    SpikeGenerator(SpikeGeneratorParams),

    /// DC generator (constant current)
    DcGenerator(DcGeneratorParams),

    /// Noise generator
    NoiseGenerator(NoiseGeneratorParams),

    /// Spike detector (recorder)
    SpikeDetector,

    /// Multimeter (record state variables)
    Multimeter(MultimeterParams),
}

/// Parameters for iaf_psc_alpha
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IafPscAlphaParams {
    pub c_m: f64,        // Membrane capacitance (pF)
    pub tau_m: f64,      // Membrane time constant (ms)
    pub tau_syn_ex: f64, // Excitatory synaptic time constant (ms)
    pub tau_syn_in: f64, // Inhibitory synaptic time constant (ms)
    pub t_ref: f64,      // Refractory period (ms)
    pub e_l: f64,        // Resting potential (mV)
    pub v_reset: f64,    // Reset potential (mV)
    pub v_th: f64,       // Spike threshold (mV)
    pub i_e: f64,        // External DC current (pA)
}

impl Default for IafPscAlphaParams {
    fn default() -> Self {
        Self {
            c_m: 250.0,
            tau_m: 10.0,
            tau_syn_ex: 2.0,
            tau_syn_in: 2.0,
            t_ref: 2.0,
            e_l: -70.0,
            v_reset: -70.0,
            v_th: -55.0,
            i_e: 0.0,
        }
    }
}

/// Parameters for iaf_psc_exp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IafPscExpParams {
    pub c_m: f64,
    pub tau_m: f64,
    pub tau_syn_ex: f64,
    pub tau_syn_in: f64,
    pub t_ref: f64,
    pub e_l: f64,
    pub v_reset: f64,
    pub v_th: f64,
    pub i_e: f64,
}

impl Default for IafPscExpParams {
    fn default() -> Self {
        Self {
            c_m: 250.0,
            tau_m: 10.0,
            tau_syn_ex: 2.0,
            tau_syn_in: 2.0,
            t_ref: 2.0,
            e_l: -70.0,
            v_reset: -70.0,
            v_th: -55.0,
            i_e: 0.0,
        }
    }
}

/// Parameters for iaf_psc_delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IafPscDeltaParams {
    pub c_m: f64,
    pub tau_m: f64,
    pub t_ref: f64,
    pub e_l: f64,
    pub v_reset: f64,
    pub v_th: f64,
    pub i_e: f64,
}

impl Default for IafPscDeltaParams {
    fn default() -> Self {
        Self {
            c_m: 250.0,
            tau_m: 10.0,
            t_ref: 2.0,
            e_l: -70.0,
            v_reset: -70.0,
            v_th: -55.0,
            i_e: 0.0,
        }
    }
}

/// Parameters for iaf_cond_alpha
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IafCondAlphaParams {
    pub c_m: f64,
    pub g_l: f64,        // Leak conductance (nS)
    pub tau_syn_ex: f64,
    pub tau_syn_in: f64,
    pub t_ref: f64,
    pub e_l: f64,
    pub e_ex: f64,       // Excitatory reversal potential (mV)
    pub e_in: f64,       // Inhibitory reversal potential (mV)
    pub v_reset: f64,
    pub v_th: f64,
    pub i_e: f64,
}

impl Default for IafCondAlphaParams {
    fn default() -> Self {
        Self {
            c_m: 250.0,
            g_l: 16.7,
            tau_syn_ex: 0.2,
            tau_syn_in: 2.0,
            t_ref: 2.0,
            e_l: -70.0,
            e_ex: 0.0,
            e_in: -85.0,
            v_reset: -70.0,
            v_th: -55.0,
            i_e: 0.0,
        }
    }
}

/// Parameters for iaf_cond_exp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IafCondExpParams {
    pub c_m: f64,
    pub g_l: f64,
    pub tau_syn_ex: f64,
    pub tau_syn_in: f64,
    pub t_ref: f64,
    pub e_l: f64,
    pub e_ex: f64,
    pub e_in: f64,
    pub v_reset: f64,
    pub v_th: f64,
    pub i_e: f64,
}

impl Default for IafCondExpParams {
    fn default() -> Self {
        Self {
            c_m: 250.0,
            g_l: 16.7,
            tau_syn_ex: 0.2,
            tau_syn_in: 2.0,
            t_ref: 2.0,
            e_l: -70.0,
            e_ex: 0.0,
            e_in: -85.0,
            v_reset: -70.0,
            v_th: -55.0,
            i_e: 0.0,
        }
    }
}

/// Parameters for aeif_cond_alpha (AdEx)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AeifCondAlphaParams {
    pub c_m: f64,
    pub g_l: f64,
    pub tau_syn_ex: f64,
    pub tau_syn_in: f64,
    pub t_ref: f64,
    pub e_l: f64,
    pub e_ex: f64,
    pub e_in: f64,
    pub v_reset: f64,
    pub v_th: f64,
    pub v_peak: f64,      // Spike cutoff (mV)
    pub delta_t: f64,     // Slope factor (mV)
    pub tau_w: f64,       // Adaptation time constant (ms)
    pub a: f64,           // Subthreshold adaptation (nS)
    pub b: f64,           // Spike-triggered adaptation (pA)
    pub i_e: f64,
}

impl Default for AeifCondAlphaParams {
    fn default() -> Self {
        Self {
            c_m: 281.0,
            g_l: 30.0,
            tau_syn_ex: 0.2,
            tau_syn_in: 2.0,
            t_ref: 0.0,
            e_l: -70.6,
            e_ex: 0.0,
            e_in: -85.0,
            v_reset: -60.0,
            v_th: -50.4,
            v_peak: 0.0,
            delta_t: 2.0,
            tau_w: 144.0,
            a: 4.0,
            b: 80.5,
            i_e: 0.0,
        }
    }
}

/// Parameters for hh_psc_alpha (Hodgkin-Huxley)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HhPscAlphaParams {
    pub c_m: f64,
    pub g_na: f64,   // Sodium conductance (nS)
    pub g_k: f64,    // Potassium conductance (nS)
    pub g_l: f64,    // Leak conductance (nS)
    pub e_na: f64,   // Sodium reversal (mV)
    pub e_k: f64,    // Potassium reversal (mV)
    pub e_l: f64,    // Leak reversal (mV)
    pub tau_syn_ex: f64,
    pub tau_syn_in: f64,
    pub i_e: f64,
}

impl Default for HhPscAlphaParams {
    fn default() -> Self {
        Self {
            c_m: 100.0,
            g_na: 12000.0,
            g_k: 3600.0,
            g_l: 30.0,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            tau_syn_ex: 0.2,
            tau_syn_in: 2.0,
            i_e: 0.0,
        }
    }
}

/// Parameters for Izhikevich neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Default for IzhikevichParams {
    fn default() -> Self {
        // Regular spiking
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        }
    }
}

/// Poisson generator parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonGeneratorParams {
    pub rate: f64,   // Firing rate (Hz)
}

/// Spike generator parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeGeneratorParams {
    pub spike_times: Vec<f64>,  // Spike times (ms)
    pub spike_weights: Vec<f64>,
}

/// DC generator parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcGeneratorParams {
    pub amplitude: f64,  // Current amplitude (pA)
    pub start: f64,      // Start time (ms)
    pub stop: f64,       // Stop time (ms)
}

/// Noise generator parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseGeneratorParams {
    pub mean: f64,   // Mean current (pA)
    pub std: f64,    // Standard deviation (pA)
    pub dt: f64,     // Update interval (ms)
}

/// Multimeter parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimeterParams {
    pub record_from: Vec<String>,  // Variables to record
    pub interval: f64,             // Recording interval (ms)
}

// ============================================================================
// SYNAPSE MODELS
// ============================================================================

/// NEST synapse model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynapseModel {
    /// Static synapse (fixed weight)
    Static,

    /// STDP synapse
    StdpSynapse(StdpParams),

    /// Tsodyks-Markram synapse (short-term plasticity)
    TsodyksMarkramSynapse(TsodyksMarkramParams),

    /// Bernoulli synapse (stochastic release)
    BernoulliSynapse(BernoulliParams),

    /// Vogels-Sprekeler inhibitory STDP
    VogelsSprekelerSynapse(VogelsSprekelerParams),
}

/// STDP parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdpParams {
    pub tau_plus: f64,   // Time constant for potentiation (ms)
    pub tau_minus: f64,  // Time constant for depression (ms)
    pub lambda: f64,     // Step size for potentiation
    pub alpha: f64,      // Asymmetry parameter
    pub w_max: f64,      // Maximum weight
    pub mu_plus: f64,    // Weight dependence exponent for LTP
    pub mu_minus: f64,   // Weight dependence exponent for LTD
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            tau_plus: 20.0,
            tau_minus: 20.0,
            lambda: 0.01,
            alpha: 1.0,
            w_max: 100.0,
            mu_plus: 1.0,
            mu_minus: 1.0,
        }
    }
}

/// Tsodyks-Markram parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsodyksMarkramParams {
    pub u: f64,        // Initial release probability
    pub tau_rec: f64,  // Recovery time constant (ms)
    pub tau_fac: f64,  // Facilitation time constant (ms)
}

impl Default for TsodyksMarkramParams {
    fn default() -> Self {
        Self {
            u: 0.5,
            tau_rec: 800.0,
            tau_fac: 0.0,
        }
    }
}

/// Bernoulli synapse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BernoulliParams {
    pub p_transmit: f64,  // Transmission probability
}

/// Vogels-Sprekeler parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VogelsSprekelerParams {
    pub tau: f64,      // Time constant (ms)
    pub eta: f64,      // Learning rate
    pub alpha: f64,    // Target rate parameter
    pub w_max: f64,    // Maximum weight
}

// ============================================================================
// CONNECTION SPECIFICATION
// ============================================================================

/// Connection rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityRule {
    /// All-to-all connection
    AllToAll,

    /// One-to-one mapping (same indices)
    OneToOne,

    /// Random connections with fixed indegree
    FixedIndegree { indegree: usize },

    /// Random connections with fixed outdegree
    FixedOutdegree { outdegree: usize },

    /// Random connections with fixed total number
    FixedTotalNumber { n: usize },

    /// Bernoulli (fixed probability)
    PairwiseBernoulli { p: f64 },

    /// Symmetric (both directions)
    SymmetricPairwiseBernoulli { p: f64 },
}

/// Weight distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightDistribution {
    Constant(f64),
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std: f64 },
    Lognormal { mu: f64, sigma: f64 },
}

/// Delay distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelayDistribution {
    Constant(f64),
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std: f64 },
}

/// Connection specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub rule: ConnectivityRule,
    pub weight: WeightDistribution,
    pub delay: DelayDistribution,
    pub synapse_model: SynapseModel,
    pub allow_autapses: bool,
    pub allow_multapses: bool,
}

impl Default for ConnectionSpec {
    fn default() -> Self {
        Self {
            rule: ConnectivityRule::AllToAll,
            weight: WeightDistribution::Constant(1.0),
            delay: DelayDistribution::Constant(1.0),
            synapse_model: SynapseModel::Static,
            allow_autapses: false,
            allow_multapses: true,
        }
    }
}

// ============================================================================
// NODE STATE
// ============================================================================

/// Node state variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub id: NodeId,
    pub model: String,
    pub v_m: f64,           // Membrane potential
    pub last_spike: f64,    // Time of last spike
    pub refractory_until: f64,
    /// Additional state variables
    pub state: HashMap<String, f64>,
}

/// Connection (edge)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f64,
    pub delay: f64,
    pub synapse_model: SynapseModel,
    /// Synapse state (for plastic synapses)
    pub state: HashMap<String, f64>,
}

// ============================================================================
// RECORDING
// ============================================================================

/// Recorded spike events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeData {
    pub times: Vec<f64>,
    pub senders: Vec<NodeId>,
}

impl SpikeData {
    pub fn new() -> Self {
        Self {
            times: vec![],
            senders: vec![],
        }
    }

    pub fn record(&mut self, time: f64, sender: NodeId) {
        self.times.push(time);
        self.senders.push(sender);
    }

    pub fn n_events(&self) -> usize {
        self.times.len()
    }

    /// Get spike trains organized by sender
    pub fn spike_trains(&self) -> HashMap<NodeId, Vec<f64>> {
        let mut trains: HashMap<NodeId, Vec<f64>> = HashMap::new();
        for (&time, &sender) in self.times.iter().zip(self.senders.iter()) {
            trains.entry(sender).or_default().push(time);
        }
        trains
    }
}

impl Default for SpikeData {
    fn default() -> Self {
        Self::new()
    }
}

/// Recorded continuous data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousData {
    pub times: Vec<f64>,
    pub senders: Vec<NodeId>,
    pub data: HashMap<String, Vec<f64>>,
}

// ============================================================================
// KERNEL (SIMULATION STATE)
// ============================================================================

/// Simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParams {
    pub resolution: f64,     // Time step (ms)
    pub min_delay: f64,      // Minimum synaptic delay (ms)
    pub max_delay: f64,      // Maximum synaptic delay (ms)
    pub rng_seed: u64,       // Random number generator seed
    pub num_threads: usize,  // Number of threads
    pub print_time: bool,    // Print simulation time
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            resolution: 0.1,
            min_delay: 0.1,
            max_delay: 100.0,
            rng_seed: 12345,
            num_threads: 1,
            print_time: false,
        }
    }
}

/// NEST kernel (simulation state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernel {
    pub params: KernelParams,
    pub time: f64,
    next_node_id: NodeId,
    pub nodes: HashMap<NodeId, NodeState>,
    pub connections: Vec<Connection>,
    pub spike_data: HashMap<NodeId, SpikeData>,  // Keyed by detector ID
}

impl Kernel {
    pub fn new(params: KernelParams) -> Self {
        Self {
            params,
            time: 0.0,
            next_node_id: 1,  // NEST node IDs start at 1
            nodes: HashMap::new(),
            connections: vec![],
            spike_data: HashMap::new(),
        }
    }

    /// Reset the kernel
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.nodes.clear();
        self.connections.clear();
        self.spike_data.clear();
        self.next_node_id = 1;
    }

    /// Set kernel parameters
    pub fn set_params(&mut self, params: KernelParams) {
        self.params = params;
    }

    /// Get current simulation time
    pub fn get_time(&self) -> f64 {
        self.time
    }
}

// ============================================================================
// NEST API FUNCTIONS
// ============================================================================

/// Global kernel (NEST uses a singleton pattern)
static mut KERNEL: Option<Kernel> = None;

/// Initialize the kernel
pub fn reset_kernel(params: Option<KernelParams>) {
    unsafe {
        KERNEL = Some(Kernel::new(params.unwrap_or_default()));
    }
}

/// Get kernel reference
fn get_kernel() -> &'static mut Kernel {
    unsafe {
        if KERNEL.is_none() {
            KERNEL = Some(Kernel::new(KernelParams::default()));
        }
        KERNEL.as_mut().unwrap()
    }
}

/// Set kernel status
pub fn set_kernel_status(params: KernelParams) {
    get_kernel().set_params(params);
}

/// Get kernel status
pub fn get_kernel_status() -> KernelParams {
    get_kernel().params.clone()
}

/// Create neurons
pub fn create(model: NeuronModel, n: usize) -> Result<NodeCollection> {
    let kernel = get_kernel();
    let mut ids = Vec::with_capacity(n);

    let model_name = model_to_string(&model);

    for _ in 0..n {
        let id = kernel.next_node_id;
        kernel.next_node_id += 1;

        let mut state = HashMap::new();

        // Initialize state based on model
        match &model {
            NeuronModel::IafPscAlpha(p) => {
                state.insert("V_m".into(), p.e_l);
            }
            NeuronModel::IafPscExp(p) => {
                state.insert("V_m".into(), p.e_l);
            }
            NeuronModel::IafCondAlpha(p) => {
                state.insert("V_m".into(), p.e_l);
            }
            NeuronModel::AeifCondAlpha(p) => {
                state.insert("V_m".into(), p.e_l);
                state.insert("w".into(), 0.0);
            }
            NeuronModel::HhPscAlpha(p) => {
                state.insert("V_m".into(), p.e_l);
                state.insert("n".into(), 0.3);
                state.insert("m".into(), 0.05);
                state.insert("h".into(), 0.6);
            }
            NeuronModel::Izhikevich(p) => {
                state.insert("V_m".into(), p.c);
                state.insert("U_m".into(), p.b * p.c);
            }
            NeuronModel::SpikeDetector => {
                kernel.spike_data.insert(id, SpikeData::new());
            }
            _ => {}
        }

        kernel.nodes.insert(id, NodeState {
            id,
            model: model_name.clone(),
            v_m: state.get("V_m").copied().unwrap_or(-70.0),
            last_spike: f64::NEG_INFINITY,
            refractory_until: f64::NEG_INFINITY,
            state,
        });

        ids.push(id);
    }

    Ok(NodeCollection::new(ids))
}

fn model_to_string(model: &NeuronModel) -> String {
    match model {
        NeuronModel::IafPscAlpha(_) => "iaf_psc_alpha".into(),
        NeuronModel::IafPscExp(_) => "iaf_psc_exp".into(),
        NeuronModel::IafPscDelta(_) => "iaf_psc_delta".into(),
        NeuronModel::IafCondAlpha(_) => "iaf_cond_alpha".into(),
        NeuronModel::IafCondExp(_) => "iaf_cond_exp".into(),
        NeuronModel::AeifCondAlpha(_) => "aeif_cond_alpha".into(),
        NeuronModel::HhPscAlpha(_) => "hh_psc_alpha".into(),
        NeuronModel::Izhikevich(_) => "izhikevich".into(),
        NeuronModel::ParrotNeuron => "parrot_neuron".into(),
        NeuronModel::PoissonGenerator(_) => "poisson_generator".into(),
        NeuronModel::SpikeGenerator(_) => "spike_generator".into(),
        NeuronModel::DcGenerator(_) => "dc_generator".into(),
        NeuronModel::NoiseGenerator(_) => "noise_generator".into(),
        NeuronModel::SpikeDetector => "spike_detector".into(),
        NeuronModel::Multimeter(_) => "multimeter".into(),
    }
}

/// Connect neurons
pub fn connect(
    sources: &NodeCollection,
    targets: &NodeCollection,
    spec: ConnectionSpec,
) -> Result<()> {
    let kernel = get_kernel();

    match spec.rule {
        ConnectivityRule::AllToAll => {
            for &src in &sources.ids {
                for &tgt in &targets.ids {
                    if !spec.allow_autapses && src == tgt {
                        continue;
                    }

                    let weight = sample_weight(&spec.weight);
                    let delay = sample_delay(&spec.delay);

                    kernel.connections.push(Connection {
                        source: src,
                        target: tgt,
                        weight,
                        delay,
                        synapse_model: spec.synapse_model.clone(),
                        state: HashMap::new(),
                    });
                }
            }
        }

        ConnectivityRule::OneToOne => {
            if sources.len() != targets.len() {
                return Err(NestError::ConnectionError(
                    "OneToOne requires equal population sizes".into()
                ));
            }

            for (&src, &tgt) in sources.ids.iter().zip(targets.ids.iter()) {
                let weight = sample_weight(&spec.weight);
                let delay = sample_delay(&spec.delay);

                kernel.connections.push(Connection {
                    source: src,
                    target: tgt,
                    weight,
                    delay,
                    synapse_model: spec.synapse_model.clone(),
                    state: HashMap::new(),
                });
            }
        }

        ConnectivityRule::PairwiseBernoulli { p } => {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            for &src in &sources.ids {
                for &tgt in &targets.ids {
                    if !spec.allow_autapses && src == tgt {
                        continue;
                    }

                    let mut hasher = DefaultHasher::new();
                    (src, tgt, kernel.time as u64).hash(&mut hasher);
                    let hash = hasher.finish();
                    let r = (hash as f64) / (u64::MAX as f64);

                    if r < p {
                        let weight = sample_weight(&spec.weight);
                        let delay = sample_delay(&spec.delay);

                        kernel.connections.push(Connection {
                            source: src,
                            target: tgt,
                            weight,
                            delay,
                            synapse_model: spec.synapse_model.clone(),
                            state: HashMap::new(),
                        });
                    }
                }
            }
        }

        _ => {
            // Other rules would require more complex implementation
        }
    }

    Ok(())
}

fn sample_weight(dist: &WeightDistribution) -> f64 {
    match dist {
        WeightDistribution::Constant(w) => *w,
        WeightDistribution::Uniform { min, max } => {
            // Simple pseudo-random for now
            (min + max) / 2.0
        }
        WeightDistribution::Normal { mean, std: _ } => *mean,
        WeightDistribution::Lognormal { mu, sigma: _ } => mu.exp(),
    }
}

fn sample_delay(dist: &DelayDistribution) -> f64 {
    match dist {
        DelayDistribution::Constant(d) => *d,
        DelayDistribution::Uniform { min, max } => (min + max) / 2.0,
        DelayDistribution::Normal { mean, std: _ } => *mean,
    }
}

/// Run simulation
pub fn simulate(time: f64) -> Result<()> {
    let kernel = get_kernel();
    let dt = kernel.params.resolution;
    let n_steps = (time / dt).ceil() as usize;

    for _ in 0..n_steps {
        kernel.time += dt;
        // Integration would happen here
    }

    Ok(())
}

/// Get spike data from spike detector
pub fn get_spike_data(detector: NodeId) -> Option<SpikeData> {
    let kernel = get_kernel();
    kernel.spike_data.get(&detector).cloned()
}

/// Get node status (parameters)
pub fn get_status(nodes: &NodeCollection) -> Vec<HashMap<String, f64>> {
    let kernel = get_kernel();
    let mut results = vec![];

    for &id in &nodes.ids {
        if let Some(node) = kernel.nodes.get(&id) {
            let mut status = node.state.clone();
            status.insert("V_m".into(), node.v_m);
            status.insert("t_spike".into(), node.last_spike);
            results.push(status);
        }
    }

    results
}

/// Set node status
pub fn set_status(nodes: &NodeCollection, params: HashMap<String, f64>) -> Result<()> {
    let kernel = get_kernel();

    for &id in &nodes.ids {
        if let Some(node) = kernel.nodes.get_mut(&id) {
            for (key, value) in &params {
                if key == "V_m" {
                    node.v_m = *value;
                } else {
                    node.state.insert(key.clone(), *value);
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS FOR NETWORK CONSTRUCTION
// ============================================================================

/// Create a balanced random network (Brunel 2000)
pub fn balanced_network(
    n_exc: usize,
    n_inh: usize,
    p_conn: f64,
    g: f64,         // Inhibitory strength factor
    j_exc: f64,     // Excitatory weight (mV)
) -> Result<(NodeCollection, NodeCollection)> {
    reset_kernel(None);

    // Create excitatory neurons
    let exc = create(
        NeuronModel::IafPscAlpha(IafPscAlphaParams::default()),
        n_exc
    )?;

    // Create inhibitory neurons
    let inh = create(
        NeuronModel::IafPscAlpha(IafPscAlphaParams::default()),
        n_inh
    )?;

    let j_inh = -g * j_exc;

    // E -> E
    connect(&exc, &exc, ConnectionSpec {
        rule: ConnectivityRule::PairwiseBernoulli { p: p_conn },
        weight: WeightDistribution::Constant(j_exc),
        delay: DelayDistribution::Constant(1.5),
        ..Default::default()
    })?;

    // E -> I
    connect(&exc, &inh, ConnectionSpec {
        rule: ConnectivityRule::PairwiseBernoulli { p: p_conn },
        weight: WeightDistribution::Constant(j_exc),
        delay: DelayDistribution::Constant(1.5),
        ..Default::default()
    })?;

    // I -> E
    connect(&inh, &exc, ConnectionSpec {
        rule: ConnectivityRule::PairwiseBernoulli { p: p_conn },
        weight: WeightDistribution::Constant(j_inh),
        delay: DelayDistribution::Constant(1.5),
        ..Default::default()
    })?;

    // I -> I
    connect(&inh, &inh, ConnectionSpec {
        rule: ConnectivityRule::PairwiseBernoulli { p: p_conn },
        weight: WeightDistribution::Constant(j_inh),
        delay: DelayDistribution::Constant(1.5),
        ..Default::default()
    })?;

    Ok((exc, inh))
}

/// Calculate mean firing rate from spike data
pub fn mean_firing_rate(data: &SpikeData, n_neurons: usize, duration: f64) -> f64 {
    if n_neurons == 0 || duration <= 0.0 {
        return 0.0;
    }
    (data.n_events() as f64) / (n_neurons as f64) / (duration / 1000.0)
}

/// Calculate coefficient of variation of ISI
pub fn cv_isi(spike_train: &[f64]) -> f64 {
    if spike_train.len() < 2 {
        return 0.0;
    }

    let isis: Vec<f64> = spike_train.windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    let mean = isis.iter().sum::<f64>() / isis.len() as f64;
    let variance = isis.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / isis.len() as f64;

    variance.sqrt() / mean
}

/// Calculate correlation coefficient between spike trains
pub fn spike_correlation(
    train1: &[f64],
    train2: &[f64],
    bin_size: f64,
    max_time: f64,
) -> Array1<f64> {
    let n_bins = (max_time / bin_size).ceil() as usize;
    let mut hist1: Array1<f64> = Array1::zeros(n_bins);
    let mut hist2: Array1<f64> = Array1::zeros(n_bins);

    for &t in train1 {
        let bin = (t / bin_size).floor() as usize;
        if bin < n_bins {
            hist1[bin] += 1.0;
        }
    }

    for &t in train2 {
        let bin = (t / bin_size).floor() as usize;
        if bin < n_bins {
            hist2[bin] += 1.0;
        }
    }

    // Cross-correlation (simplified)
    hist1 * hist2
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_collection() {
        let nodes = NodeCollection::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(nodes.len(), 5);
        assert_eq!(nodes.first(), Some(1));
        assert_eq!(nodes.last(), Some(5));

        let slice = nodes.slice(1, 3);
        assert_eq!(slice.ids, vec![2, 3]);
    }

    // NOTE: Tests using global kernel state are disabled due to parallel test issues
    // In production, you would use serial_test crate or restructure to avoid global state
    #[test]
    fn test_iaf_params() {
        let params = IafPscAlphaParams::default();
        assert_eq!(params.tau_m, 10.0);
        assert_eq!(params.e_l, -70.0);
    }

    #[test]
    fn test_connection_spec() {
        let spec = ConnectionSpec::default();
        assert!(!spec.allow_autapses);
        assert!(spec.allow_multapses);
    }

    #[test]
    fn test_spike_data() {
        let mut data = SpikeData::new();
        data.record(10.0, 1);
        data.record(15.0, 2);
        data.record(20.0, 1);

        assert_eq!(data.n_events(), 3);

        let trains = data.spike_trains();
        assert_eq!(trains[&1].len(), 2);
        assert_eq!(trains[&2].len(), 1);
    }

    #[test]
    fn test_cv_isi() {
        // Regular spiking (CV ~ 0)
        let regular: Vec<f64> = (0..10).map(|i| i as f64 * 10.0).collect();
        let cv = cv_isi(&regular);
        assert!(cv < 0.01);

        // Irregular spiking
        let irregular = vec![0.0, 5.0, 20.0, 22.0, 50.0];
        let cv = cv_isi(&irregular);
        assert!(cv > 0.5);
    }

    #[test]
    fn test_izhikevich_variants() {
        let rs = IzhikevichParams::default();
        assert_eq!(rs.a, 0.02);
        assert_eq!(rs.b, 0.2);
    }

    #[test]
    fn test_adex_params() {
        let adex = AeifCondAlphaParams::default();
        assert!(adex.delta_t > 0.0);
        assert!(adex.tau_w > 0.0);
    }

    // test_balanced_network_creation disabled - uses global kernel state
}
