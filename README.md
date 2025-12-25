# OldiesRules

**Revival of Legacy Academic Simulators in Rust**

OldiesRules provides modern, safe, high-performance Rust implementations of classic simulation software that is abandoned, unmaintained, or running on obsolete platforms.

## Target Simulators

| Simulator | Original | Era | Status | OldiesRules |
|-----------|----------|-----|--------|-------------|
| GENESIS | SLI + C | 1980s-2014 | Semi-abandoned | `genesis-rs` |
| XPPAUT | C + FORTRAN | 1990s | Hobby project | `xppaut-rs` |
| AUTO | FORTRAN | 1980s | Legacy | `auto-rs` |
| ModelDB | Various | 1996+ | Active/Legacy | `modeldb-rs` |

## Why Revive These Simulators?

1. **Scientific Legacy**: Decades of validated research depends on these tools
2. **Reproducibility**: Old papers can't be reproduced without the original software
3. **Education**: Students should learn from classic models
4. **Safety**: Modern Rust prevents the bugs plaguing legacy code
5. **Performance**: Take advantage of modern hardware and parallelism

## Crates

### Core
- `oldies-core`: Shared types (ODE systems, ion channels, time series)
- `oldies-cli`: Command-line interface

### Simulators
- `genesis-rs`: GENESIS neural simulator
- `xppaut-rs`: XPPAUT bifurcation analysis
- `auto-rs`: AUTO continuation algorithms
- `modeldb-rs`: ModelDB model importer

## Usage

```bash
# Run a GENESIS script
oldies genesis model.g

# Bifurcation analysis with XPP
oldies xpp model.ode --parameter I

# Import from ModelDB
oldies import 3670 --output ./imported/

# List supported simulators
oldies list
```

## GENESIS-RS

GENESIS (General Neural Simulation System) was the premier neural simulator before NEURON. Key features:

- SLI script parser
- Element-based model construction
- Message passing between elements
- HH channel objects
- Compartmental models

```rust
use genesis_rs::{GenesisSimulation, objects};

let mut sim = GenesisSimulation::new();

// Create soma compartment
objects::compartment(&mut sim, "/cell/soma");

// Add ion channels
objects::na_channel(&mut sim, "/cell/soma/Na");
objects::k_channel(&mut sim, "/cell/soma/K");

// Run simulation
sim.run(100.0); // 100 ms
```

## XPPAUT-RS

XPPAUT provides bifurcation analysis for dynamical systems:

- Fixed point finding
- Eigenvalue analysis
- Limit cycle detection
- Parameter continuation
- Hopf bifurcation detection

```rust
use xppaut_rs::{BifurcationAnalyzer, examples};

// Create Lorenz system
let model = examples::lorenz(10.0, 28.0, 8.0/3.0);
let analyzer = BifurcationAnalyzer::new(model);

// Find fixed points
let fixed_points = analyzer.find_fixed_points(
    examples::lorenz_rhs,
    &[vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]],
);

for fp in &fixed_points {
    println!("Fixed point at {:?}, stable: {}", fp.state, fp.stable);
}
```

## Integration with HumanBrain

OldiesRules is designed to work with [HumanBrain](https://github.com/Yatrogenesis/HumanBrain):

- Import GENESIS models directly
- Use bifurcation analysis on HumanBrain dynamics
- Access ModelDB's 1800+ neural models
- Validate Rust implementations against legacy code

## Related Projects

- [Rosetta](https://github.com/Yatrogenesis/Rosetta): Transpile legacy code to Rust
- [HumanBrain](https://github.com/Yatrogenesis/HumanBrain): GPU neural simulation
- [Stochastic-Framework](https://github.com/Yatrogenesis/Stochastic-Framework): Pattern detection

## License

MIT OR Apache-2.0
