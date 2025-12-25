//! # OldiesRules CLI
//!
//! Command-line interface for legacy simulator revival.

use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "oldies")]
#[command(author = "Yatrogenesis")]
#[command(version = "0.1.0")]
#[command(about = "Legacy simulator revival toolkit", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a GENESIS script
    Genesis {
        /// Script file
        script: PathBuf,
    },

    /// Run XPPAUT bifurcation analysis
    Xpp {
        /// ODE file
        ode: PathBuf,
        /// Parameter to continue
        #[arg(short, long)]
        parameter: Option<String>,
    },

    /// Import model from ModelDB
    Import {
        /// ModelDB ID
        id: u32,
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// List supported simulators
    List,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Genesis { script } => {
            println!("{} {}", "Loading GENESIS script:".green().bold(), script.display());
            println!("{}", "GENESIS simulation not yet implemented".yellow());
        }

        Commands::Xpp { ode, parameter } => {
            println!("{} {}", "Loading ODE file:".green().bold(), ode.display());
            if let Some(param) = parameter {
                println!("  Continuing parameter: {}", param.cyan());
            }
            println!("{}", "XPP bifurcation analysis not yet implemented".yellow());
        }

        Commands::Import { id, output } => {
            println!("{} {}", "Importing ModelDB:".green().bold(), id);
            if let Some(dir) = output {
                println!("  Output: {}", dir.display());
            }
            println!("{}", "ModelDB import not yet implemented".yellow());
        }

        Commands::List => {
            println!("{}", "Supported Legacy Simulators:".green().bold());
            println!();
            println!("  {} - General Neural Simulation System", "genesis".cyan());
            println!("  {} - Bifurcation analysis (XPP + AUTO)", "xppaut".cyan());
            println!("  {} - Continuation software", "auto".cyan());
            println!("  {} - Model database importer", "modeldb".cyan());
        }
    }

    Ok(())
}
