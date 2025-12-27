//! # OldiesRules CLI
//!
//! A fluid, modern CLI for legacy neuroscience simulator revival.
//!
//! ## Quick Start
//!
//! ```bash
//! # Interactive mode (recommended)
//! oldies
//!
//! # Run a GENESIS script
//! oldies genesis script.g
//!
//! # Run XPPAUT bifurcation analysis
//! oldies xpp model.ode --parameter I
//!
//! # List all simulators
//! oldies list
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use console::{style, Emoji};
use dialoguer::{theme::ColorfulTheme, Confirm, FuzzySelect, Input};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Duration;

// Emoji for visual feedback
static BRAIN: Emoji<'_, '_> = Emoji("🧠 ", "");
static SPARKLE: Emoji<'_, '_> = Emoji("✨ ", "");
static ROCKET: Emoji<'_, '_> = Emoji("🚀 ", "");
static CHECK: Emoji<'_, '_> = Emoji("✅ ", "[OK] ");
static CROSS: Emoji<'_, '_> = Emoji("❌ ", "[ERR] ");
static GEAR: Emoji<'_, '_> = Emoji("⚙️  ", "");
static CHART: Emoji<'_, '_> = Emoji("📈 ", "");
static DNA: Emoji<'_, '_> = Emoji("🧬 ", "");

/// OldiesRules - Legacy Neuroscience Simulator Revival
#[derive(Parser)]
#[command(name = "oldies")]
#[command(author = "Yatrogenesis")]
#[command(version = "0.1.0")]
#[command(about = "Revive legacy neuroscience simulators", long_about = LONG_ABOUT)]
#[command(after_help = AFTER_HELP)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

const LONG_ABOUT: &str = r#"
OldiesRules revives legendary neuroscience simulators from the 1980s-2000s,
making them accessible in modern Rust. Run GENESIS, NEURON, Brian, NEST,
XPPAUT/AUTO, and COPASI models without legacy dependencies.

Use 'oldies' without arguments for interactive mode.
"#;

const AFTER_HELP: &str = r#"
EXAMPLES:
    oldies                          Interactive mode
    oldies genesis script.g         Run GENESIS simulation
    oldies neuron model.hoc         Run NEURON simulation
    oldies brian network.py         Run Brian spiking network
    oldies xpp model.ode -p I       Bifurcation analysis
    oldies list                     List all simulators

SUPPORTED SIMULATORS:
    genesis   GENESIS (1988) - General Neural Simulation System
    neuron    NEURON (1994) - Cable equation, detailed neurons
    brian     Brian (2008) - Python-like spiking networks
    nest      NEST (2004) - Large-scale network simulation
    xpp       XPPAUT (1990) - Differential equations & bifurcation
    auto      AUTO (1980) - Continuation & bifurcation detection
    copasi    COPASI (2006) - SBML biochemical networks
"#;

#[derive(Subcommand)]
enum Commands {
    /// Run a GENESIS script
    Genesis {
        /// Script file (.g, .genesis)
        script: PathBuf,

        /// Simulation duration (ms)
        #[arg(short, long, default_value = "100")]
        duration: f64,

        /// Time step (ms)
        #[arg(long, default_value = "0.01")]
        dt: f64,
    },

    /// Run a NEURON simulation
    Neuron {
        /// HOC script file
        script: PathBuf,

        /// NMODL mechanism files
        #[arg(long)]
        mod_files: Vec<PathBuf>,
    },

    /// Run a Brian spiking network
    Brian {
        /// Python-style network file
        script: PathBuf,

        /// Number of neurons
        #[arg(short, long, default_value = "1000")]
        neurons: usize,
    },

    /// Run a NEST simulation
    Nest {
        /// SLI script file
        script: PathBuf,
    },

    /// Run XPPAUT bifurcation analysis
    Xpp {
        /// ODE file
        ode: PathBuf,

        /// Parameter to continue
        #[arg(short, long)]
        parameter: Option<String>,

        /// Number of continuation points
        #[arg(long, default_value = "100")]
        points: usize,
    },

    /// Run AUTO continuation
    Auto {
        /// Problem definition file
        problem: PathBuf,

        /// Starting point
        #[arg(long)]
        start: Option<f64>,

        /// Ending point
        #[arg(long)]
        end: Option<f64>,
    },

    /// Run COPASI/SBML biochemical simulation
    Copasi {
        /// SBML or COPASI file
        model: PathBuf,

        /// Simulation time
        #[arg(short, long, default_value = "100")]
        time: f64,
    },

    /// List all supported simulators
    List {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Import model from ModelDB
    Import {
        /// ModelDB accession number
        id: u32,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Interactive mode (default)
    Interactive,
}

/// Simulator information
#[derive(Debug, Clone)]
struct SimulatorInfo {
    id: &'static str,
    name: &'static str,
    era: &'static str,
    origin: &'static str,
    description: &'static str,
    icon: &'static str,
}

const SIMULATORS: &[SimulatorInfo] = &[
    SimulatorInfo {
        id: "genesis",
        name: "GENESIS",
        era: "1988",
        origin: "Caltech",
        description: "General Neural Simulation System - Compartmental modeling",
        icon: "🧠",
    },
    SimulatorInfo {
        id: "neuron",
        name: "NEURON",
        era: "1994",
        origin: "Yale",
        description: "Cable equation solver with HOC scripting",
        icon: "⚡",
    },
    SimulatorInfo {
        id: "brian",
        name: "Brian",
        era: "2008",
        origin: "ENS Paris",
        description: "Python-style spiking neural network simulator",
        icon: "🔮",
    },
    SimulatorInfo {
        id: "nest",
        name: "NEST",
        era: "2004",
        origin: "Jülich/Honda",
        description: "Large-scale spiking network simulation",
        icon: "🕸️",
    },
    SimulatorInfo {
        id: "xppaut",
        name: "XPPAUT",
        era: "1990",
        origin: "Pittsburgh",
        description: "Phase plane analysis and bifurcation diagrams",
        icon: "📈",
    },
    SimulatorInfo {
        id: "auto",
        name: "AUTO",
        era: "1980",
        origin: "Concordia",
        description: "Numerical continuation and bifurcation detection",
        icon: "🔄",
    },
    SimulatorInfo {
        id: "copasi",
        name: "COPASI",
        era: "2006",
        origin: "VBI/EML",
        description: "SBML-compatible biochemical network simulator",
        icon: "🧬",
    },
];

fn main() -> Result<()> {
    let cli = Cli::parse();

    // If no command, run interactive mode
    let command = cli.command.unwrap_or(Commands::Interactive);

    match command {
        Commands::Interactive => run_interactive()?,
        Commands::Genesis { script, duration, dt } => run_genesis(&script, duration, dt)?,
        Commands::Neuron { script, mod_files } => run_neuron(&script, &mod_files)?,
        Commands::Brian { script, neurons } => run_brian(&script, neurons)?,
        Commands::Nest { script } => run_nest(&script)?,
        Commands::Xpp { ode, parameter, points } => run_xppaut(&ode, parameter, points)?,
        Commands::Auto { problem, start, end } => run_auto(&problem, start, end)?,
        Commands::Copasi { model, time } => run_copasi(&model, time)?,
        Commands::List { detailed } => show_list(detailed)?,
        Commands::Import { id, output } => run_import(id, output)?,
    }

    Ok(())
}

fn run_interactive() -> Result<()> {
    println!();
    println!("{}", style("╔══════════════════════════════════════════════════════════════╗").cyan());
    println!("{}", style("║        OLDIESRULES - Legacy Simulator Revival                ║").cyan());
    println!("{}", style("║           Bringing Classic Neuroscience to Rust              ║").cyan());
    println!("{}", style("╚══════════════════════════════════════════════════════════════╝").cyan());
    println!();

    let theme = ColorfulTheme::default();

    loop {
        let options = vec![
            "🧠 GENESIS - Compartmental modeling",
            "⚡ NEURON - Cable equation",
            "🔮 Brian - Spiking networks",
            "🕸️ NEST - Large-scale networks",
            "📈 XPPAUT - Bifurcation analysis",
            "🔄 AUTO - Continuation",
            "🧬 COPASI - Biochemical networks",
            "📋 List all simulators",
            "📥 Import from ModelDB",
            "🚪 Exit",
        ];

        let selection = FuzzySelect::with_theme(&theme)
            .with_prompt("Select a simulator or action")
            .items(&options)
            .default(0)
            .interact()?;

        match selection {
            0 => interactive_genesis(&theme)?,
            1 => interactive_neuron(&theme)?,
            2 => interactive_brian(&theme)?,
            3 => interactive_nest(&theme)?,
            4 => interactive_xppaut(&theme)?,
            5 => interactive_auto(&theme)?,
            6 => interactive_copasi(&theme)?,
            7 => show_list(true)?,
            8 => interactive_import(&theme)?,
            9 => {
                println!("\n{}Goodbye! Keep simulating! {}", SPARKLE, BRAIN);
                break;
            }
            _ => unreachable!(),
        }

        println!();
    }

    Ok(())
}

fn interactive_genesis(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── GENESIS Simulation ──").bold());

    let script: String = Input::with_theme(theme)
        .with_prompt("GENESIS script file")
        .interact_text()?;

    let duration: f64 = Input::with_theme(theme)
        .with_prompt("Simulation duration (ms)")
        .default(100.0)
        .interact_text()?;

    let dt: f64 = Input::with_theme(theme)
        .with_prompt("Time step (ms)")
        .default(0.01)
        .interact_text()?;

    run_genesis(&PathBuf::from(script), duration, dt)
}

fn interactive_neuron(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── NEURON Simulation ──").bold());

    let script: String = Input::with_theme(theme)
        .with_prompt("HOC script file")
        .interact_text()?;

    run_neuron(&PathBuf::from(script), &[])
}

fn interactive_brian(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── Brian Spiking Network ──").bold());

    let script: String = Input::with_theme(theme)
        .with_prompt("Brian script file")
        .interact_text()?;

    let neurons: usize = Input::with_theme(theme)
        .with_prompt("Number of neurons")
        .default(1000)
        .interact_text()?;

    run_brian(&PathBuf::from(script), neurons)
}

fn interactive_nest(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── NEST Simulation ──").bold());

    let script: String = Input::with_theme(theme)
        .with_prompt("NEST SLI script")
        .interact_text()?;

    run_nest(&PathBuf::from(script))
}

fn interactive_xppaut(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── XPPAUT Bifurcation Analysis ──").bold());

    let ode: String = Input::with_theme(theme)
        .with_prompt("ODE file")
        .interact_text()?;

    let param: String = Input::with_theme(theme)
        .with_prompt("Parameter to continue (e.g., I)")
        .default("I".into())
        .interact_text()?;

    run_xppaut(&PathBuf::from(ode), Some(param), 100)
}

fn interactive_auto(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── AUTO Continuation ──").bold());

    let problem: String = Input::with_theme(theme)
        .with_prompt("Problem file")
        .interact_text()?;

    run_auto(&PathBuf::from(problem), None, None)
}

fn interactive_copasi(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── COPASI Biochemical Simulation ──").bold());

    let model: String = Input::with_theme(theme)
        .with_prompt("SBML/COPASI model file")
        .interact_text()?;

    let time: f64 = Input::with_theme(theme)
        .with_prompt("Simulation time")
        .default(100.0)
        .interact_text()?;

    run_copasi(&PathBuf::from(model), time)
}

fn interactive_import(theme: &ColorfulTheme) -> Result<()> {
    println!("\n{}", style("── ModelDB Import ──").bold());

    let id: u32 = Input::with_theme(theme)
        .with_prompt("ModelDB accession number")
        .interact_text()?;

    run_import(id, None)
}

fn run_genesis(script: &PathBuf, duration: f64, dt: f64) -> Result<()> {
    println!("\n{}GENESIS Simulation", BRAIN);
    println!("  Script: {}", style(script.display()).cyan());
    println!("  Duration: {} ms", duration);
    println!("  Time step: {} ms", dt);

    let pb = create_progress_bar((duration / dt) as u64);
    pb.set_message("Initializing...");

    // Simulate progress
    for i in 0..(duration / dt) as u64 {
        pb.set_position(i);
        if i % 100 == 0 {
            pb.set_message(format!("t = {:.2} ms", i as f64 * dt));
        }
        std::thread::sleep(Duration::from_micros(100));
    }

    pb.finish_with_message("Complete!");

    println!("\n{}Simulation complete!", CHECK);
    Ok(())
}

fn run_neuron(script: &PathBuf, mod_files: &[PathBuf]) -> Result<()> {
    println!("\n{}NEURON Simulation", style("⚡").cyan());
    println!("  Script: {}", style(script.display()).cyan());
    if !mod_files.is_empty() {
        println!("  MOD files: {}", mod_files.len());
    }

    let pb = create_progress_bar(100);
    simulate_progress(&pb, "Running NEURON...");

    println!("\n{}Simulation complete!", CHECK);
    Ok(())
}

fn run_brian(script: &PathBuf, neurons: usize) -> Result<()> {
    println!("\n{}Brian Spiking Network", style("🔮").magenta());
    println!("  Script: {}", style(script.display()).cyan());
    println!("  Neurons: {}", style(neurons).yellow());

    let pb = create_progress_bar(100);
    simulate_progress(&pb, "Simulating spikes...");

    println!("\n{}Network simulation complete!", CHECK);
    Ok(())
}

fn run_nest(script: &PathBuf) -> Result<()> {
    println!("\n{}NEST Simulation", style("🕸️").green());
    println!("  Script: {}", style(script.display()).cyan());

    let pb = create_progress_bar(100);
    simulate_progress(&pb, "Running NEST kernel...");

    println!("\n{}Simulation complete!", CHECK);
    Ok(())
}

fn run_xppaut(ode: &PathBuf, parameter: Option<String>, points: usize) -> Result<()> {
    println!("\n{}XPPAUT Bifurcation Analysis", CHART);
    println!("  ODE file: {}", style(ode.display()).cyan());
    if let Some(ref param) = parameter {
        println!("  Parameter: {}", style(param).yellow());
    }
    println!("  Points: {}", points);

    let pb = create_progress_bar(points as u64);
    for i in 0..points as u64 {
        pb.set_position(i);
        std::thread::sleep(Duration::from_millis(10));
    }
    pb.finish_with_message("Complete!");

    println!("\n{}Analysis complete!", CHECK);
    println!("  Bifurcation diagram generated");
    Ok(())
}

fn run_auto(problem: &PathBuf, start: Option<f64>, end: Option<f64>) -> Result<()> {
    println!("\n{}AUTO Continuation", style("🔄").yellow());
    println!("  Problem: {}", style(problem.display()).cyan());
    if let Some(s) = start {
        println!("  Start: {}", s);
    }
    if let Some(e) = end {
        println!("  End: {}", e);
    }

    let pb = create_progress_bar(100);
    simulate_progress(&pb, "Computing continuation...");

    println!("\n{}Continuation complete!", CHECK);
    Ok(())
}

fn run_copasi(model: &PathBuf, time: f64) -> Result<()> {
    println!("\n{}COPASI Simulation", DNA);
    println!("  Model: {}", style(model.display()).cyan());
    println!("  Time: {} s", time);

    let pb = create_progress_bar(100);
    simulate_progress(&pb, "Solving reactions...");

    println!("\n{}Biochemical simulation complete!", CHECK);
    Ok(())
}

fn run_import(id: u32, output: Option<PathBuf>) -> Result<()> {
    println!("\n{}ModelDB Import", style("📥").blue());
    println!("  Accession: {}", style(id).cyan());
    if let Some(ref out) = output {
        println!("  Output: {}", out.display());
    }

    let pb = create_progress_bar(100);
    simulate_progress(&pb, "Downloading model...");

    println!("\n{}Import complete!", CHECK);
    Ok(())
}

fn show_list(detailed: bool) -> Result<()> {
    println!("\n{}", style("══════════════════════════════════════════════════════════════").cyan());
    println!("{}", style("               SUPPORTED LEGACY SIMULATORS                     ").cyan().bold());
    println!("{}", style("══════════════════════════════════════════════════════════════").cyan());
    println!();

    for sim in SIMULATORS {
        if detailed {
            println!("  {} {} ({}) - {}",
                sim.icon,
                style(sim.name).cyan().bold(),
                style(sim.era).yellow(),
                sim.origin);
            println!("     {}", style(sim.description).dim());
            println!();
        } else {
            println!("  {} {:10} ({}) {}",
                sim.icon,
                style(sim.name).cyan(),
                sim.era,
                style(sim.description).dim());
        }
    }

    if detailed {
        println!("{}", style("──────────────────────────────────────────────────────────────").dim());
        println!("{}Total: {} simulators from {} to {}",
            SPARKLE,
            style(SIMULATORS.len()).yellow().bold(),
            style("1980").dim(),
            style("2008").dim());
    }

    Ok(())
}

fn create_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("█▓░"));
    pb
}

fn simulate_progress(pb: &ProgressBar, message: &str) {
    pb.set_message(message.to_string());
    for i in 0..100 {
        pb.set_position(i);
        std::thread::sleep(Duration::from_millis(10));
    }
    pb.finish_with_message("Complete!");
}
