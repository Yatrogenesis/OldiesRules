# 🎸 OldiesRules

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18071019.svg)](https://doi.org/10.5281/zenodo.18071019)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Yatrogenesis/OldiesRules?style=social)](https://github.com/Yatrogenesis/OldiesRules)

**Revival of Classic Academic Simulators in Rust**

> *Decades of scientific validation. Modern safety and performance.*

OldiesRules provides modern, safe, high-performance Rust implementations of legendary simulation software that is abandoned, unmaintained, or running on obsolete platforms.

## 🚀 Quick Start

```bash
# Install CLI
cargo install --git https://github.com/Yatrogenesis/OldiesRules oldies-cli

# Install GUI
cargo install --git https://github.com/Yatrogenesis/OldiesRules oldies-gui

# Interactive mode
oldies

# Or run directly
oldies genesis model.g
oldies neuron cell.hoc
```

## ✨ Features

- **7 Classic Simulators** - GENESIS, NEURON, XPPAUT, AUTO, COPASI, Brian, NEST
- **Modern GUI** - Real-time visualization with egui
- **Interactive CLI** - Fuzzy search, progress bars, wizard mode
- **Full Compatibility** - Parse original file formats
- **Cross-Platform** - Windows, macOS, Linux

## 🧠 Supported Simulators

| Simulator | Era | Original | Purpose | Status |
|-----------|-----|----------|---------|--------|
| **GENESIS** | 1988 | SLI + C | Compartmental neural modeling | ✅ Complete |
| **NEURON** | 1984 | HOC + NMODL | Cable equation, ion channels | ✅ Complete |
| **XPPAUT** | 1990s | C + FORTRAN | Bifurcation analysis | ✅ Complete |
| **AUTO** | 1980s | FORTRAN | Continuation algorithms | ✅ Complete |
| **COPASI** | 2004 | C++ + SBML | Biochemical networks | ✅ Complete |
| **Brian** | 2008 | Python | Spiking neural networks | ✅ Complete |
| **NEST** | 2001 | C++ + SLI | Large-scale networks | ✅ Complete |

## 🎯 Interactive Mode

```
$ oldies

🎸 OLDIES RULES - Legacy Simulators Reborn
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

? Select a simulator or action (type to search):
› 🧠 GENESIS - Compartmental modeling (1988)
  ⚡ NEURON - Cable equation (1984)
  📊 XPPAUT - Bifurcation analysis (1990s)
  📈 AUTO - Continuation (1980s)
  🧬 COPASI - Biochemical networks (2004)
  🔥 Brian - Spiking networks (2008)
  🌐 NEST - Large-scale simulation (2001)

? Enter path to model file: model.g

⠹ Parsing SLI script...
⠹ Building element tree...
⠹ Connecting messages...
⠹ Simulating 100ms...

✅ Simulation complete!
   Duration: 1.23s
   Timesteps: 10,000
```

## 🖥️ GUI Application

```bash
oldies-gui
```

![OldiesRules GUI](docs/gui-screenshot.png)

- **Simulator Selection** - Choose from 7 classic simulators
- **Parameter Editor** - Modify model parameters in real-time
- **Live Plotting** - Watch variables evolve during simulation
- **Data Export** - CSV, JSON, HDF5 output formats

## 📦 Crates

| Crate | Description |
|-------|-------------|
| `oldies-core` | Shared types: ODEs, channels, time series |
| `oldies-cli` | Interactive command-line interface |
| `oldies-gui` | Graphical interface with egui |
| `genesis-rs` | GENESIS neural simulator |
| `neuron-rs` | NEURON simulator (HOC/NMODL parser) |
| `xppaut-rs` | XPPAUT bifurcation analysis |
| `auto-rs` | AUTO continuation algorithms |
| `copasi-rs` | COPASI/SBML biochemical networks |
| `brian-rs` | Brian spiking networks |
| `nest-rs` | NEST large-scale simulator |
| `modeldb-rs` | ModelDB model importer |

## 🔬 Examples

### GENESIS - Hodgkin-Huxley Neuron

```rust
use genesis_rs::{GenesisSimulation, objects};

let mut sim = GenesisSimulation::new();

// Create compartment
objects::compartment(&mut sim, "/cell/soma");

// Add ion channels
objects::na_channel(&mut sim, "/cell/soma/Na");
objects::k_channel(&mut sim, "/cell/soma/K");

// Inject current
objects::inject(&mut sim, "/cell/soma", 0.1); // 0.1 nA

// Run 100 ms
sim.run(100.0);
```

### XPPAUT - Lorenz Attractor

```rust
use xppaut_rs::{BifurcationAnalyzer, examples};

let model = examples::lorenz(10.0, 28.0, 8.0/3.0);
let analyzer = BifurcationAnalyzer::new(model);

// Find and classify fixed points
let fixed_points = analyzer.find_fixed_points();
for fp in &fixed_points {
    println!("{:?} - Stable: {}", fp.state, fp.stable);
}

// Detect Hopf bifurcation
let hopf = analyzer.detect_hopf_bifurcation("rho", 0.0..50.0);
```

### NEURON - Ion Channel

```rust
use neuron_rs::{Section, Mechanism};

let mut soma = Section::new("soma");
soma.set_length(20.0);  // μm
soma.set_diameter(20.0);

// Insert Hodgkin-Huxley mechanism
soma.insert(Mechanism::HH);

// Run simulation
let v = soma.record("v");
neuron_rs::finitialize(-65.0);
neuron_rs::continuerun(100.0);
```

### Brian - Spiking Network

```rust
use brian_rs::{NeuronGroup, Synapses, Network, equations};

let eqs = equations!("dv/dt = (I - v) / tau : volt");
let group = NeuronGroup::new(100, eqs);

let synapses = Synapses::new(&group, &group);
synapses.connect(0.1); // 10% connection probability

let mut net = Network::new();
net.add(group);
net.add(synapses);
net.run(1.0); // 1 second
```

## 💡 Why Revive These Simulators?

| Problem | OldiesRules Solution |
|---------|---------------------|
| Research papers unreproducible | Run original models natively |
| Legacy code has buffer overflows | Rust memory safety |
| Single-threaded 1990s code | Modern parallelism with Rayon |
| Cryptic FORTRAN-77 | Clean, documented Rust |
| Abandoned projects | Active maintenance |
| Platform-specific binaries | Cross-platform Rust |

## 🔗 Ecosystem

OldiesRules is part of the **Yatrogenesis** scientific computing suite:

- **[HumanBrain](https://github.com/Yatrogenesis/HumanBrain)** - GPU neural simulation at biological scale
- **[Rosetta](https://github.com/Yatrogenesis/Rosetta)** - Legacy code transpiler
- **[Stochastic-Framework](https://github.com/Yatrogenesis/Stochastic-Framework)** - Pattern detection

### Integration

```rust
use humanbrain::Neuron;
use oldies_rs::modeldb;

// Import ModelDB model
let model = modeldb::import(3670)?; // Traub model

// Use in HumanBrain simulation
let neuron: Neuron = model.into();
```

## 📥 Installation

```bash
# CLI only
cargo install --git https://github.com/Yatrogenesis/OldiesRules oldies-cli

# GUI + CLI
cargo install --git https://github.com/Yatrogenesis/OldiesRules oldies-gui
cargo install --git https://github.com/Yatrogenesis/OldiesRules oldies-cli

# From source
git clone https://github.com/Yatrogenesis/OldiesRules
cd OldiesRules
cargo build --release

# Run
./target/release/oldies          # CLI
./target/release/oldies-gui      # GUI
```

## 📊 Comparison with Original

| Feature | Original | OldiesRules |
|---------|----------|-------------|
| Language | C/FORTRAN/C++ | Rust |
| Memory Safety | ❌ | ✅ |
| Parallel | Limited | Rayon + GPU |
| Cross-platform | Partial | ✅ |
| Modern tooling | ❌ | Cargo, tests, docs |
| File format support | Native only | All formats |

## 🤝 Contributing

We welcome contributions! Areas of interest:

- Additional model file parsers
- More example models
- Performance optimizations
- Documentation improvements

## 📜 License

MIT OR Apache-2.0

---

<p align="center">
  <i>"They don't make 'em like they used to. But we can make 'em better."</i>
</p>
