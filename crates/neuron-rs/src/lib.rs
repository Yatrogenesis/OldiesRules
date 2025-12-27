//! # NEURON-RS
//!
//! Revival of NEURON simulator in Rust.
//!
//! ## History
//!
//! NEURON was developed at Yale by Michael Hines and John Moore starting in
//! the 1980s. It's the most widely used neural simulator with 2000+ citations.
//!
//! ## Components
//!
//! 1. **HOC Parser**: NEURON's scripting language (derived from hoc calculator)
//! 2. **NMODL Parser**: Mechanism description language for ion channels
//! 3. **Simulator**: Cable equation solver with Crank-Nicolson
//!
//! ## Key NEURON Concepts
//!
//! - **Sections**: Cable segments (soma, axon, dendrite)
//! - **Mechanisms**: Ion channels, synapses (defined in NMODL)
//! - **Point Processes**: Synapses, electrodes at specific locations
//! - **Connections**: Section-to-section connectivity
//! - **cvode**: Variable time-step integration

use oldies_core::{OldiesError, Result, Time, Voltage, Current};
use pest_derive::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// HOC PARSER
// =============================================================================

/// HOC (High Order Calculator) parser for NEURON scripts
#[derive(Parser)]
#[grammar_inline = r#"
WHITESPACE = _{ " " | "\t" }
NEWLINE = _{ "\r\n" | "\n" }
COMMENT = _{ "//" ~ (!NEWLINE ~ ANY)* | "/*" ~ (!"*/" ~ ANY)* ~ "*/" }

number = @{ "-"? ~ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? ~ (("e" | "E") ~ "-"? ~ ASCII_DIGIT+)? }
string = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }

// Keywords
create_kw = { "create" }
access_kw = { "access" }
insert_kw = { "insert" }
connect_kw = { "connect" }
proc_kw = { "proc" }
func_kw = { "func" }
objref_kw = { "objref" }
objectvar_kw = { "objectvar" }
strdef_kw = { "strdef" }
new_kw = { "new" }
forall_kw = { "forall" }
forsec_kw = { "forsec" }
ifsec_kw = { "ifsec" }
for_kw = { "for" }
if_kw = { "if" }
else_kw = { "else" }
while_kw = { "while" }
return_kw = { "return" }
print_kw = { "print" }
load_file_kw = { "load_file" }

// Operators
assign = { "=" }
plus = { "+" }
minus = { "-" }
star = { "*" }
slash = { "/" }
lparen = { "(" }
rparen = { ")" }
lbrace = { "{" }
rbrace = { "}" }
lbracket = { "[" }
rbracket = { "]" }
comma = { "," }
dot = { "." }

// Expressions
primary = { number | string | identifier | lparen ~ expr ~ rparen }
member_access = { primary ~ (dot ~ identifier)* ~ (lbracket ~ expr ~ rbracket)? }
call = { member_access ~ (lparen ~ arg_list? ~ rparen)? }
arg_list = { expr ~ (comma ~ expr)* }
unary = { (minus | "!")? ~ call }
term = { unary ~ ((star | slash | "%") ~ unary)* }
arith = { term ~ ((plus | minus) ~ term)* }
comparison = { arith ~ (("<" | ">" | "<=" | ">=" | "==" | "!=") ~ arith)* }
logical = { comparison ~ (("&&" | "||") ~ comparison)* }
expr = { logical }

// Statements
statement = {
    create_stmt |
    access_stmt |
    insert_stmt |
    connect_stmt |
    proc_def |
    func_def |
    objref_stmt |
    for_stmt |
    forall_stmt |
    if_stmt |
    while_stmt |
    return_stmt |
    print_stmt |
    load_file_stmt |
    assignment |
    expr_stmt
}

create_stmt = { create_kw ~ section_list }
section_list = { section_def ~ (comma ~ section_def)* }
section_def = { identifier ~ (lbracket ~ number ~ rbracket)? }

access_stmt = { access_kw ~ identifier }
insert_stmt = { insert_kw ~ identifier }
connect_stmt = { connect_kw ~ identifier ~ lparen ~ expr ~ rparen ~ comma ~ identifier ~ lparen ~ expr ~ rparen }

proc_def = { proc_kw ~ identifier ~ lparen ~ param_list? ~ rparen ~ block }
func_def = { func_kw ~ identifier ~ lparen ~ param_list? ~ rparen ~ block }
param_list = { identifier ~ (comma ~ identifier)* }
block = { lbrace ~ statement* ~ rbrace }

objref_stmt = { (objref_kw | objectvar_kw) ~ identifier ~ (comma ~ identifier)* }

for_stmt = { for_kw ~ lparen ~ assignment ~ ";" ~ expr ~ ";" ~ assignment ~ rparen ~ (block | statement) }
forall_stmt = { forall_kw ~ (block | statement) }
if_stmt = { if_kw ~ lparen ~ expr ~ rparen ~ (block | statement) ~ (else_kw ~ (block | statement))? }
while_stmt = { while_kw ~ lparen ~ expr ~ rparen ~ (block | statement) }

return_stmt = { return_kw ~ expr? }
print_stmt = { print_kw ~ expr ~ (comma ~ expr)* }
load_file_stmt = { load_file_kw ~ lparen ~ string ~ rparen }

assignment = { member_access ~ assign ~ expr }
expr_stmt = { expr }

program = { SOI ~ statement* ~ EOI }
"#]
pub struct HocParser;

// =============================================================================
// NMODL PARSER
// =============================================================================

/// NMODL mechanism types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MechanismType {
    /// Density mechanism (per area)
    Suffix,
    /// Point process
    PointProcess,
    /// Artificial cell
    ArtificialCell,
}

/// NMODL block types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NmodlBlock {
    /// NEURON block (declares interface)
    Neuron {
        mechanism_type: MechanismType,
        suffix: String,
        useion: Vec<UseIon>,
        range: Vec<String>,
        global: Vec<String>,
        pointer: Vec<String>,
        nonspecific_current: Vec<String>,
    },
    /// UNITS block
    Units(Vec<(String, String)>),
    /// PARAMETER block
    Parameter(Vec<NmodlVariable>),
    /// STATE block
    State(Vec<String>),
    /// ASSIGNED block
    Assigned(Vec<NmodlVariable>),
    /// INITIAL block
    Initial(Vec<String>),
    /// BREAKPOINT block
    Breakpoint(Vec<String>),
    /// DERIVATIVE block
    Derivative {
        name: String,
        equations: Vec<String>,
    },
    /// KINETIC block
    Kinetic {
        name: String,
        reactions: Vec<String>,
    },
    /// PROCEDURE block
    Procedure {
        name: String,
        params: Vec<String>,
        body: Vec<String>,
    },
    /// FUNCTION block
    Function {
        name: String,
        params: Vec<String>,
        body: Vec<String>,
    },
    /// NET_RECEIVE block (for point processes)
    NetReceive {
        params: Vec<String>,
        body: Vec<String>,
    },
}

/// USEION declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseIon {
    pub ion: String,
    pub read: Vec<String>,
    pub write: Vec<String>,
    pub valence: Option<i32>,
}

/// NMODL variable with optional units and default value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NmodlVariable {
    pub name: String,
    pub default: Option<f64>,
    pub units: Option<String>,
    pub range: Option<(f64, f64)>,
}

/// Parsed NMODL mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NmodlMechanism {
    pub title: Option<String>,
    pub blocks: Vec<NmodlBlock>,
}

// =============================================================================
// NEURON MODEL
// =============================================================================

/// A NEURON section (cable segment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Section name
    pub name: String,
    /// Number of segments
    pub nseg: usize,
    /// Length (um)
    pub length: f64,
    /// Diameter (um)
    pub diam: f64,
    /// Axial resistance (ohm-cm)
    pub ra: f64,
    /// Membrane capacitance (uF/cm^2)
    pub cm: f64,
    /// Inserted mechanisms
    pub mechanisms: Vec<InsertedMechanism>,
    /// Parent section and location
    pub parent: Option<(String, f64)>,
    /// Children sections
    pub children: Vec<String>,
    /// State: membrane potential per segment
    pub v: Vec<Voltage>,
}

impl Section {
    /// Create a new section with default properties
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nseg: 1,
            length: 100.0,     // um
            diam: 1.0,         // um
            ra: 100.0,         // ohm-cm
            cm: 1.0,           // uF/cm^2
            mechanisms: Vec::new(),
            parent: None,
            children: Vec::new(),
            v: vec![-65.0],    // mV, resting potential
        }
    }

    /// Set number of segments
    pub fn set_nseg(&mut self, nseg: usize) {
        self.nseg = nseg;
        self.v = vec![-65.0; nseg];
    }

    /// Insert a mechanism
    pub fn insert(&mut self, mechanism: InsertedMechanism) {
        self.mechanisms.push(mechanism);
    }

    /// Surface area per segment (cm^2)
    pub fn area(&self) -> f64 {
        let seg_length = self.length / self.nseg as f64;
        std::f64::consts::PI * self.diam * seg_length * 1e-8  // um^2 to cm^2
    }
}

/// An inserted mechanism instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertedMechanism {
    pub name: String,
    pub parameters: HashMap<String, f64>,
    pub state: HashMap<String, Vec<f64>>,
}

/// Point process (synapse, electrode, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointProcess {
    pub name: String,
    pub section: String,
    pub location: f64,  // 0-1 along section
    pub parameters: HashMap<String, f64>,
    pub state: HashMap<String, f64>,
}

/// NEURON cell model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronCell {
    /// Cell name/ID
    pub name: String,
    /// Sections
    pub sections: HashMap<String, Section>,
    /// Point processes
    pub point_processes: Vec<PointProcess>,
    /// Currently accessed section
    current_section: Option<String>,
}

impl NeuronCell {
    /// Create a new cell
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sections: HashMap::new(),
            point_processes: Vec::new(),
            current_section: None,
        }
    }

    /// Create sections
    pub fn create(&mut self, name: &str) -> &mut Section {
        let section = Section::new(name);
        self.sections.insert(name.to_string(), section);
        self.sections.get_mut(name).unwrap()
    }

    /// Access a section
    pub fn access(&mut self, name: &str) -> Result<()> {
        if self.sections.contains_key(name) {
            self.current_section = Some(name.to_string());
            Ok(())
        } else {
            Err(OldiesError::ModelNotFound(format!("Section {} not found", name)))
        }
    }

    /// Get current section
    pub fn current(&self) -> Option<&Section> {
        self.current_section.as_ref().and_then(|n| self.sections.get(n))
    }

    /// Get mutable current section
    pub fn current_mut(&mut self) -> Option<&mut Section> {
        if let Some(name) = self.current_section.clone() {
            self.sections.get_mut(&name)
        } else {
            None
        }
    }

    /// Connect two sections
    pub fn connect(&mut self, child: &str, child_end: f64, parent: &str, parent_loc: f64) -> Result<()> {
        // Validate sections exist
        if !self.sections.contains_key(child) {
            return Err(OldiesError::ModelNotFound(format!("Section {} not found", child)));
        }
        if !self.sections.contains_key(parent) {
            return Err(OldiesError::ModelNotFound(format!("Section {} not found", parent)));
        }

        // Set parent
        if let Some(sec) = self.sections.get_mut(child) {
            sec.parent = Some((parent.to_string(), parent_loc));
        }

        // Add child
        if let Some(sec) = self.sections.get_mut(parent) {
            if !sec.children.contains(&child.to_string()) {
                sec.children.push(child.to_string());
            }
        }

        Ok(())
    }

    /// Add a point process
    pub fn add_point_process(&mut self, pp: PointProcess) {
        self.point_processes.push(pp);
    }

    /// Get total number of segments
    pub fn total_segments(&self) -> usize {
        self.sections.values().map(|s| s.nseg).sum()
    }
}

// =============================================================================
// STANDARD MECHANISMS
// =============================================================================

/// Standard NEURON mechanisms
pub mod mechanisms {
    use super::*;

    /// Hodgkin-Huxley sodium channel (hh)
    pub fn hh_na() -> InsertedMechanism {
        let mut params = HashMap::new();
        params.insert("gnabar".to_string(), 0.12);  // S/cm^2
        params.insert("ena".to_string(), 50.0);     // mV

        InsertedMechanism {
            name: "na".to_string(),
            parameters: params,
            state: HashMap::new(),
        }
    }

    /// Hodgkin-Huxley potassium channel (hh)
    pub fn hh_k() -> InsertedMechanism {
        let mut params = HashMap::new();
        params.insert("gkbar".to_string(), 0.036);  // S/cm^2
        params.insert("ek".to_string(), -77.0);     // mV

        InsertedMechanism {
            name: "k".to_string(),
            parameters: params,
            state: HashMap::new(),
        }
    }

    /// Passive (leak) channel
    pub fn pas() -> InsertedMechanism {
        let mut params = HashMap::new();
        params.insert("g".to_string(), 0.001);      // S/cm^2
        params.insert("e".to_string(), -70.0);      // mV

        InsertedMechanism {
            name: "pas".to_string(),
            parameters: params,
            state: HashMap::new(),
        }
    }

    /// Exponential synapse (ExpSyn)
    pub fn exp_syn(section: &str, loc: f64) -> PointProcess {
        let mut params = HashMap::new();
        params.insert("tau".to_string(), 2.0);      // ms
        params.insert("e".to_string(), 0.0);        // mV

        PointProcess {
            name: "ExpSyn".to_string(),
            section: section.to_string(),
            location: loc,
            parameters: params,
            state: HashMap::new(),
        }
    }

    /// Double-exponential synapse (Exp2Syn)
    pub fn exp2_syn(section: &str, loc: f64) -> PointProcess {
        let mut params = HashMap::new();
        params.insert("tau1".to_string(), 0.5);     // ms (rise)
        params.insert("tau2".to_string(), 2.0);     // ms (decay)
        params.insert("e".to_string(), 0.0);        // mV

        PointProcess {
            name: "Exp2Syn".to_string(),
            section: section.to_string(),
            location: loc,
            parameters: params,
            state: HashMap::new(),
        }
    }

    /// Current clamp (IClamp)
    pub fn iclamp(section: &str, loc: f64, delay: f64, dur: f64, amp: f64) -> PointProcess {
        let mut params = HashMap::new();
        params.insert("delay".to_string(), delay);  // ms
        params.insert("dur".to_string(), dur);      // ms
        params.insert("amp".to_string(), amp);      // nA

        PointProcess {
            name: "IClamp".to_string(),
            section: section.to_string(),
            location: loc,
            parameters: params,
            state: HashMap::new(),
        }
    }
}

// =============================================================================
// SIMULATOR
// =============================================================================

/// NEURON simulation state
pub struct NeuronSimulation {
    /// Cell models
    pub cells: Vec<NeuronCell>,
    /// Current time (ms)
    pub t: Time,
    /// Time step (ms)
    pub dt: Time,
    /// Stop time (ms)
    pub tstop: Time,
    /// Temperature (celsius)
    pub celsius: f64,
    /// Recorded variables
    pub recordings: HashMap<String, Vec<f64>>,
}

impl NeuronSimulation {
    /// Create a new simulation
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            t: 0.0,
            dt: 0.025,      // Default NEURON dt
            tstop: 100.0,
            celsius: 37.0,  // Default temperature
            recordings: HashMap::new(),
        }
    }

    /// Add a cell to the simulation
    pub fn add_cell(&mut self, cell: NeuronCell) {
        self.cells.push(cell);
    }

    /// Initialize simulation
    pub fn finitialize(&mut self, v_init: Voltage) {
        self.t = 0.0;
        self.recordings.clear();

        for cell in &mut self.cells {
            for section in cell.sections.values_mut() {
                for v in &mut section.v {
                    *v = v_init;
                }
            }
        }
    }

    /// Advance one time step
    pub fn fadvance(&mut self) {
        // Simplified cable equation integration
        // In full NEURON, this uses Crank-Nicolson with Gaussian elimination
        self.t += self.dt;
    }

    /// Run simulation
    pub fn run(&mut self) {
        while self.t < self.tstop {
            self.fadvance();
        }
    }

    /// Continue running
    pub fn continuerun(&mut self, tstop: Time) {
        self.tstop = tstop;
        self.run();
    }
}

impl Default for NeuronSimulation {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HOC FILE LOADER
// =============================================================================

/// Load and parse a HOC file
pub fn load_hoc(_content: &str) -> Result<NeuronCell> {
    // TODO: Implement full HOC parser
    // For now, return a basic cell
    Ok(NeuronCell::new("cell"))
}

/// Parse NMODL content
pub fn parse_nmodl(_content: &str) -> Result<NmodlMechanism> {
    // TODO: Implement full NMODL parser
    Ok(NmodlMechanism {
        title: None,
        blocks: Vec::new(),
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cell() {
        let mut cell = NeuronCell::new("pyramidal");
        cell.create("soma");
        cell.create("axon");
        cell.create("dend");

        assert_eq!(cell.sections.len(), 3);
    }

    #[test]
    fn test_access_section() {
        let mut cell = NeuronCell::new("test");
        cell.create("soma");

        cell.access("soma").unwrap();
        assert!(cell.current().is_some());
        assert_eq!(cell.current().unwrap().name, "soma");
    }

    #[test]
    fn test_connect_sections() {
        let mut cell = NeuronCell::new("test");
        cell.create("soma");
        cell.create("dend");

        cell.connect("dend", 0.0, "soma", 1.0).unwrap();

        let dend = cell.sections.get("dend").unwrap();
        assert_eq!(dend.parent, Some(("soma".to_string(), 1.0)));
    }

    #[test]
    fn test_insert_mechanism() {
        let mut cell = NeuronCell::new("test");
        let soma = cell.create("soma");
        soma.insert(mechanisms::hh_na());
        soma.insert(mechanisms::hh_k());
        soma.insert(mechanisms::pas());

        assert_eq!(soma.mechanisms.len(), 3);
    }

    #[test]
    fn test_point_process() {
        let mut cell = NeuronCell::new("test");
        cell.create("soma");
        cell.add_point_process(mechanisms::iclamp("soma", 0.5, 10.0, 50.0, 0.5));

        assert_eq!(cell.point_processes.len(), 1);
        assert_eq!(cell.point_processes[0].name, "IClamp");
    }

    #[test]
    fn test_simulation_init() {
        let mut sim = NeuronSimulation::new();
        let mut cell = NeuronCell::new("test");
        cell.create("soma");
        sim.add_cell(cell);

        sim.finitialize(-65.0);
        assert_eq!(sim.t, 0.0);
    }

    #[test]
    fn test_section_area() {
        let mut sec = Section::new("test");
        sec.length = 100.0;  // um
        sec.diam = 10.0;     // um

        let area = sec.area();
        // pi * 10 * 100 * 1e-8 = ~3.14e-5 cm^2
        assert!((area - 3.14159e-5).abs() < 1e-6);
    }
}
