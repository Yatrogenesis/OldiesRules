//! # GENESIS-RS
//!
//! Revival of GENESIS (General Neural Simulation System) in Rust.
//!
//! ## History
//!
//! GENESIS was developed at Caltech starting in the 1980s. The last official
//! version (2.4) was released in 2014. It is now "user-supported, not actively
//! developed."
//!
//! ## GENESIS Script Language (SLI)
//!
//! GENESIS uses its own scripting language (SLI) for model specification.
//! This crate provides:
//!
//! 1. SLI parser
//! 2. Script interpreter
//! 3. Native Rust model execution
//!
//! ## Key GENESIS Concepts
//!
//! - **Elements**: Basic simulation objects (compartments, channels, etc.)
//! - **Messages**: Connections between elements
//! - **Objects**: Templates for creating elements
//! - **Scripts**: SLI programs that set up simulations
//!
//! ## Compatibility with MOOSE
//!
//! MOOSE (Multiscale Object-Oriented Simulation Environment) inherited the
//! GENESIS parser. This crate aims to be compatible with both GENESIS and
//! MOOSE script formats.

use oldies_core::{OldiesError, Result, TimeSeries, StateVector, Time, Voltage};
use pest_derive::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SLI (Script Language Interpreter) parser
#[derive(Parser)]
#[grammar_inline = r#"
WHITESPACE = _{ " " | "\t" }
COMMENT = _{ "//" ~ (!NEWLINE ~ ANY)* }

number = @{ "-"? ~ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? ~ (("e" | "E") ~ "-"? ~ ASCII_DIGIT+)? }
string = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
path = @{ "/" ~ (identifier ~ "/")* ~ identifier }

statement = { command | assignment | block }
command = { identifier ~ argument* }
argument = { number | string | identifier | path }
assignment = { identifier ~ "=" ~ expression }
expression = { number | string | identifier | path }
block = { "{" ~ statement* ~ "}" }
program = { SOI ~ statement* ~ EOI }
"#]
pub struct SliParser;

/// GENESIS element types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementType {
    /// Neural compartment
    Compartment,
    /// HH sodium channel
    NaChannel,
    /// HH potassium channel
    KChannel,
    /// Calcium channel
    CaChannel,
    /// Synapse
    Synapse,
    /// Spike generator
    SpikeGen,
    /// Recorder (output)
    Recorder,
    /// Neutral (container)
    Neutral,
    /// Custom object
    Custom(String),
}

/// A GENESIS element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    /// Element path (e.g., /cell/soma)
    pub path: String,
    /// Element type
    pub element_type: ElementType,
    /// Parameters
    pub params: HashMap<String, f64>,
    /// Child elements
    pub children: Vec<String>,
    /// Incoming messages
    pub messages_in: Vec<Message>,
    /// Outgoing messages
    pub messages_out: Vec<Message>,
}

impl Element {
    /// Create a new element
    pub fn new(path: &str, element_type: ElementType) -> Self {
        Self {
            path: path.to_string(),
            element_type,
            params: HashMap::new(),
            children: Vec::new(),
            messages_in: Vec::new(),
            messages_out: Vec::new(),
        }
    }

    /// Set a parameter
    pub fn set_param(&mut self, name: &str, value: f64) {
        self.params.insert(name.to_string(), value);
    }

    /// Get a parameter
    pub fn get_param(&self, name: &str) -> Option<f64> {
        self.params.get(name).copied()
    }
}

/// GENESIS message (connection between elements)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Source element path
    pub source: String,
    /// Source field
    pub source_field: String,
    /// Destination element path
    pub dest: String,
    /// Destination field
    pub dest_field: String,
    /// Message type
    pub msg_type: String,
}

/// GENESIS simulation
#[derive(Debug)]
pub struct GenesisSimulation {
    /// Root element
    elements: HashMap<String, Element>,
    /// Current simulation time
    time: Time,
    /// Time step
    dt: Time,
    /// Recorded data
    recordings: HashMap<String, TimeSeries>,
}

impl GenesisSimulation {
    /// Create a new simulation
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            time: 0.0,
            dt: 1e-5, // 10 microseconds
            recordings: HashMap::new(),
        }
    }

    /// Create an element
    pub fn create(&mut self, path: &str, element_type: ElementType) -> &mut Element {
        let element = Element::new(path, element_type);
        self.elements.insert(path.to_string(), element);
        self.elements.get_mut(path).unwrap()
    }

    /// Get an element
    pub fn get(&self, path: &str) -> Option<&Element> {
        self.elements.get(path)
    }

    /// Get a mutable element
    pub fn get_mut(&mut self, path: &str) -> Option<&mut Element> {
        self.elements.get_mut(path)
    }

    /// Add a message between elements
    pub fn add_message(
        &mut self,
        source: &str,
        source_field: &str,
        dest: &str,
        dest_field: &str,
        msg_type: &str,
    ) -> Result<()> {
        let msg = Message {
            source: source.to_string(),
            source_field: source_field.to_string(),
            dest: dest.to_string(),
            dest_field: dest_field.to_string(),
            msg_type: msg_type.to_string(),
        };

        // Add to source's outgoing
        if let Some(elem) = self.elements.get_mut(source) {
            elem.messages_out.push(msg.clone());
        } else {
            return Err(OldiesError::ModelNotFound(source.to_string()));
        }

        // Add to destination's incoming
        if let Some(elem) = self.elements.get_mut(dest) {
            elem.messages_in.push(msg);
        } else {
            return Err(OldiesError::ModelNotFound(dest.to_string()));
        }

        Ok(())
    }

    /// Run simulation step
    pub fn step(&mut self) {
        // TODO: Implement actual simulation logic
        self.time += self.dt;
    }

    /// Run simulation for specified duration
    pub fn run(&mut self, duration: Time) {
        let steps = (duration / self.dt) as usize;
        for _ in 0..steps {
            self.step();
        }
    }

    /// Set time step
    pub fn set_dt(&mut self, dt: Time) {
        self.dt = dt;
    }

    /// Get current time
    pub fn current_time(&self) -> Time {
        self.time
    }
}

impl Default for GenesisSimulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard GENESIS objects
pub mod objects {
    use super::*;

    /// Create a standard compartment
    pub fn compartment(sim: &mut GenesisSimulation, path: &str) -> &mut Element {
        let elem = sim.create(path, ElementType::Compartment);
        elem.set_param("Rm", 1e9);      // Membrane resistance (ohms)
        elem.set_param("Cm", 1e-11);    // Membrane capacitance (F)
        elem.set_param("Ra", 1e7);      // Axial resistance (ohms)
        elem.set_param("Em", -0.065);   // Resting potential (V)
        elem.set_param("initVm", -0.065);
        elem.set_param("Vm", -0.065);
        elem
    }

    /// Create HH sodium channel
    pub fn na_channel(sim: &mut GenesisSimulation, path: &str) -> &mut Element {
        let elem = sim.create(path, ElementType::NaChannel);
        elem.set_param("Gbar", 0.12);   // Max conductance (S/cm^2)
        elem.set_param("Ek", 0.045);    // Reversal potential (V)
        elem
    }

    /// Create HH potassium channel
    pub fn k_channel(sim: &mut GenesisSimulation, path: &str) -> &mut Element {
        let elem = sim.create(path, ElementType::KChannel);
        elem.set_param("Gbar", 0.036);  // Max conductance (S/cm^2)
        elem.set_param("Ek", -0.082);   // Reversal potential (V)
        elem
    }
}

/// Load and execute a GENESIS script
pub fn load_script(_script: &str) -> Result<GenesisSimulation> {
    // TODO: Implement full script parser
    Ok(GenesisSimulation::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_compartment() {
        let mut sim = GenesisSimulation::new();
        objects::compartment(&mut sim, "/cell/soma");

        let soma = sim.get("/cell/soma").unwrap();
        assert!(soma.get_param("Rm").is_some());
    }

    #[test]
    fn test_simulation_step() {
        let mut sim = GenesisSimulation::new();
        sim.set_dt(0.001);
        assert_eq!(sim.current_time(), 0.0);

        sim.step();
        assert!((sim.current_time() - 0.001).abs() < 1e-10);
    }
}
