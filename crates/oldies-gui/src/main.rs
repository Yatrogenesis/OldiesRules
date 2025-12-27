//! # OldiesRules GUI
//!
//! A modern, fluid graphical interface for legacy neuroscience simulator revival.
//!
//! Features:
//! - Run simulations from GENESIS, NEURON, Brian, NEST
//! - Bifurcation analysis with XPPAUT/AUTO
//! - Real-time visualization of results
//! - Import models from ModelDB

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::path::PathBuf;

/// Simulator types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Simulator {
    #[default]
    Genesis,
    Neuron,
    Brian,
    Nest,
    Xppaut,
    Auto,
    Copasi,
}

impl Simulator {
    fn all() -> &'static [Self] {
        &[
            Self::Genesis,
            Self::Neuron,
            Self::Brian,
            Self::Nest,
            Self::Xppaut,
            Self::Auto,
            Self::Copasi,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Genesis => "GENESIS",
            Self::Neuron => "NEURON",
            Self::Brian => "Brian",
            Self::Nest => "NEST",
            Self::Xppaut => "XPPAUT",
            Self::Auto => "AUTO",
            Self::Copasi => "COPASI",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::Genesis => "General Neural Simulation System (Caltech)",
            Self::Neuron => "NEURON Simulator (Yale) - Cable equation",
            Self::Brian => "Brian Spiking Neural Networks (Python-style)",
            Self::Nest => "NEST Simulator (Large-scale networks)",
            Self::Xppaut => "XPP/AUTO Bifurcation Analysis",
            Self::Auto => "AUTO Continuation & Bifurcation",
            Self::Copasi => "COPASI/SBML Biochemical Networks",
        }
    }

    fn icon(&self) -> &'static str {
        match self {
            Self::Genesis => "🧠",
            Self::Neuron => "⚡",
            Self::Brian => "🔮",
            Self::Nest => "🕸️",
            Self::Xppaut => "📈",
            Self::Auto => "🔄",
            Self::Copasi => "🧬",
        }
    }

    fn era(&self) -> &'static str {
        match self {
            Self::Genesis => "1988",
            Self::Neuron => "1994",
            Self::Brian => "2008",
            Self::Nest => "2004",
            Self::Xppaut => "1990",
            Self::Auto => "1980",
            Self::Copasi => "2006",
        }
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Genesis => &["g", "genesis", "sli"],
            Self::Neuron => &["hoc", "nmodl", "mod"],
            Self::Brian => &["py", "brian"],
            Self::Nest => &["sli", "nest", "py"],
            Self::Xppaut => &["ode", "xpp"],
            Self::Auto => &["f", "auto"],
            Self::Copasi => &["cps", "sbml", "xml"],
        }
    }
}

/// Simulation state
#[derive(Debug, Clone, Default)]
struct SimulationState {
    running: bool,
    progress: f32,
    time: f64,
    dt: f64,
    duration: f64,
}

/// Recorded data point
#[derive(Debug, Clone)]
struct DataPoint {
    time: f64,
    voltage: f64,
}

/// Application state
struct OldiesApp {
    // UI state
    selected_simulator: Simulator,
    show_simulator_panel: bool,
    show_settings: bool,
    dark_mode: bool,
    font_size: f32,

    // File state
    current_file: Option<PathBuf>,
    script_content: String,
    output_log: String,

    // Simulation state
    sim_state: SimulationState,
    recorded_data: VecDeque<DataPoint>,
    max_data_points: usize,

    // Parameters
    sim_duration: f64,
    sim_dt: f64,

    // Status
    status_message: String,
}

impl Default for OldiesApp {
    fn default() -> Self {
        Self {
            selected_simulator: Simulator::default(),
            show_simulator_panel: true,
            show_settings: false,
            dark_mode: true,
            font_size: 14.0,
            current_file: None,
            script_content: String::new(),
            output_log: String::new(),
            sim_state: SimulationState::default(),
            recorded_data: VecDeque::new(),
            max_data_points: 1000,
            sim_duration: 100.0,
            sim_dt: 0.1,
            status_message: "Ready".into(),
        }
    }
}

impl OldiesApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Configure custom fonts if needed
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        cc.egui_ctx.set_style(style);

        Self::default()
    }

    fn load_file(&mut self) {
        let extensions: Vec<&str> = Simulator::all()
            .iter()
            .flat_map(|s| s.file_extensions().iter())
            .copied()
            .collect();

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("All Supported", &extensions)
            .add_filter("GENESIS", &["g", "genesis", "sli"])
            .add_filter("NEURON", &["hoc", "mod"])
            .add_filter("Brian", &["py", "brian"])
            .add_filter("XPPAUT", &["ode", "xpp"])
            .add_filter("COPASI", &["cps", "sbml", "xml"])
            .pick_file()
        {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    self.script_content = content;
                    self.current_file = Some(path.clone());

                    // Auto-detect simulator
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        for sim in Simulator::all() {
                            if sim.file_extensions().contains(&ext) {
                                self.selected_simulator = *sim;
                                break;
                            }
                        }
                    }

                    self.status_message = format!("Loaded: {}", path.display());
                    self.log(&format!("Loaded file: {}", path.display()));
                }
                Err(e) => {
                    self.status_message = format!("Error: {}", e);
                    self.log(&format!("Error loading file: {}", e));
                }
            }
        }
    }

    fn save_file(&mut self) {
        let ext = match self.selected_simulator {
            Simulator::Genesis => "g",
            Simulator::Neuron => "hoc",
            Simulator::Brian => "py",
            Simulator::Nest => "sli",
            Simulator::Xppaut => "ode",
            Simulator::Auto => "f",
            Simulator::Copasi => "cps",
        };

        if let Some(path) = rfd::FileDialog::new()
            .add_filter(self.selected_simulator.name(), &[ext])
            .set_file_name(&format!("model.{}", ext))
            .save_file()
        {
            match std::fs::write(&path, &self.script_content) {
                Ok(_) => {
                    self.current_file = Some(path.clone());
                    self.status_message = format!("Saved: {}", path.display());
                    self.log(&format!("Saved to: {}", path.display()));
                }
                Err(e) => {
                    self.status_message = format!("Error: {}", e);
                }
            }
        }
    }

    fn run_simulation(&mut self) {
        self.sim_state.running = true;
        self.sim_state.time = 0.0;
        self.sim_state.dt = self.sim_dt;
        self.sim_state.duration = self.sim_duration;
        self.sim_state.progress = 0.0;
        self.recorded_data.clear();

        self.log(&format!(
            "Starting {} simulation: duration={:.1}ms, dt={:.3}ms",
            self.selected_simulator.name(),
            self.sim_duration,
            self.sim_dt
        ));

        self.status_message = format!("Running {} simulation...", self.selected_simulator.name());
    }

    fn stop_simulation(&mut self) {
        self.sim_state.running = false;
        self.status_message = "Simulation stopped".into();
        self.log("Simulation stopped by user");
    }

    fn step_simulation(&mut self) {
        if !self.sim_state.running {
            return;
        }

        // Simulate a step
        let t = self.sim_state.time;

        // Generate sample data (Hodgkin-Huxley-like action potential)
        let voltage = self.generate_sample_voltage(t);

        self.recorded_data.push_back(DataPoint {
            time: t,
            voltage,
        });

        // Limit data points
        while self.recorded_data.len() > self.max_data_points {
            self.recorded_data.pop_front();
        }

        self.sim_state.time += self.sim_state.dt;
        self.sim_state.progress = (self.sim_state.time / self.sim_state.duration).min(1.0) as f32;

        if self.sim_state.time >= self.sim_state.duration {
            self.sim_state.running = false;
            self.status_message = "Simulation complete".into();
            self.log(&format!(
                "Simulation complete: {} data points recorded",
                self.recorded_data.len()
            ));
        }
    }

    fn generate_sample_voltage(&self, t: f64) -> f64 {
        // Simulated action potential waveform
        let spike_period = 20.0; // ms
        let phase = (t % spike_period) / spike_period;

        if phase < 0.1 {
            // Rising phase
            -65.0 + 120.0 * (phase / 0.1)
        } else if phase < 0.15 {
            // Peak
            55.0 - 50.0 * ((phase - 0.1) / 0.05)
        } else if phase < 0.3 {
            // Falling phase
            5.0 - 80.0 * ((phase - 0.15) / 0.15)
        } else if phase < 0.5 {
            // Hyperpolarization
            -75.0 + 10.0 * ((phase - 0.3) / 0.2)
        } else {
            // Resting
            -65.0 + (rand_simple(t) - 0.5) * 2.0
        }
    }

    fn log(&mut self, message: &str) {
        use std::fmt::Write;
        let timestamp = format!("[{:.1}s] ", self.sim_state.time / 1000.0);
        writeln!(self.output_log, "{}{}", timestamp, message).ok();
    }

    fn export_data(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .add_filter("JSON", &["json"])
            .set_file_name("simulation_data.csv")
            .save_file()
        {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("csv");

            let content = if ext == "json" {
                // Export as JSON
                let data: Vec<_> = self.recorded_data.iter()
                    .map(|p| serde_json::json!({"time": p.time, "voltage": p.voltage}))
                    .collect();
                serde_json::to_string_pretty(&data).unwrap_or_default()
            } else {
                // Export as CSV
                let mut csv = String::from("time,voltage\n");
                for point in &self.recorded_data {
                    csv.push_str(&format!("{:.4},{:.4}\n", point.time, point.voltage));
                }
                csv
            };

            match std::fs::write(&path, content) {
                Ok(_) => {
                    self.status_message = format!("Exported: {}", path.display());
                    self.log(&format!("Data exported to: {}", path.display()));
                }
                Err(e) => {
                    self.status_message = format!("Export error: {}", e);
                }
            }
        }
    }
}

impl eframe::App for OldiesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Step simulation if running
        if self.sim_state.running {
            for _ in 0..10 {
                self.step_simulation();
            }
            ctx.request_repaint();
        }

        // Apply theme
        if self.dark_mode {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }

        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("📂 Open...").clicked() {
                        self.load_file();
                        ui.close_menu();
                    }
                    if ui.button("💾 Save...").clicked() {
                        self.save_file();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("📊 Export Data...").clicked() {
                        self.export_data();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("🚪 Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Simulation", |ui| {
                    if self.sim_state.running {
                        if ui.button("⏹ Stop").clicked() {
                            self.stop_simulation();
                            ui.close_menu();
                        }
                    } else {
                        if ui.button("▶️ Run").clicked() {
                            self.run_simulation();
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    if ui.button("🗑 Clear Data").clicked() {
                        self.recorded_data.clear();
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_simulator_panel, "Simulator Panel");
                    ui.checkbox(&mut self.show_settings, "Settings");
                    ui.checkbox(&mut self.dark_mode, "Dark Mode");
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("📖 Documentation").clicked() {
                        ui.close_menu();
                    }
                    if ui.button("ℹ️ About").clicked() {
                        ui.close_menu();
                    }
                });

                // Status on the right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.sim_state.running {
                        ui.spinner();
                        ui.label(format!("{:.1}%", self.sim_state.progress * 100.0));
                    }
                    ui.label(&self.status_message);
                });
            });
        });

        // Status bar at bottom
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{} {}",
                    self.selected_simulator.icon(),
                    self.selected_simulator.name()
                ));
                ui.separator();
                if let Some(ref path) = self.current_file {
                    ui.label(format!("📄 {}", path.file_name().unwrap_or_default().to_string_lossy()));
                }
                ui.separator();
                ui.label(format!("⏱ t={:.2}ms", self.sim_state.time));
                ui.separator();
                ui.label(format!("📊 {} points", self.recorded_data.len()));
            });
        });

        // Simulator selection panel (left)
        if self.show_simulator_panel {
            egui::SidePanel::left("simulator_panel")
                .default_width(200.0)
                .show(ctx, |ui| {
                    ui.heading("🔬 Simulators");
                    ui.separator();

                    for sim in Simulator::all() {
                        let selected = self.selected_simulator == *sim;
                        let text = format!("{} {}", sim.icon(), sim.name());

                        if ui.selectable_label(selected, text).clicked() {
                            self.selected_simulator = *sim;
                        }

                        if selected {
                            ui.indent(sim.name(), |ui| {
                                ui.label(egui::RichText::new(sim.description()).small().weak());
                                ui.label(egui::RichText::new(format!("Est. {}", sim.era())).small().weak());
                            });
                        }
                    }

                    ui.separator();
                    ui.heading("⚙️ Parameters");

                    ui.horizontal(|ui| {
                        ui.label("Duration:");
                        ui.add(egui::DragValue::new(&mut self.sim_duration)
                            .suffix(" ms")
                            .speed(1.0)
                            .range(1.0..=10000.0));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Time step:");
                        ui.add(egui::DragValue::new(&mut self.sim_dt)
                            .suffix(" ms")
                            .speed(0.01)
                            .range(0.001..=10.0));
                    });

                    ui.separator();

                    ui.horizontal(|ui| {
                        if self.sim_state.running {
                            if ui.button("⏹ Stop").clicked() {
                                self.stop_simulation();
                            }
                        } else {
                            if ui.button("▶️ Run").clicked() {
                                self.run_simulation();
                            }
                        }

                        if ui.button("🗑 Clear").clicked() {
                            self.recorded_data.clear();
                        }
                    });

                    // Progress bar
                    if self.sim_state.running {
                        ui.add(egui::ProgressBar::new(self.sim_state.progress)
                            .text(format!("{:.1}%", self.sim_state.progress * 100.0)));
                    }
                });
        }

        // Settings panel (right)
        if self.show_settings {
            egui::SidePanel::right("settings_panel")
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.heading("⚙️ Settings");
                    ui.separator();

                    ui.checkbox(&mut self.dark_mode, "Dark mode");

                    ui.horizontal(|ui| {
                        ui.label("Font size:");
                        ui.add(egui::Slider::new(&mut self.font_size, 10.0..=24.0));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Max data points:");
                        ui.add(egui::DragValue::new(&mut self.max_data_points)
                            .speed(100)
                            .range(100..=100000));
                    });

                    ui.separator();

                    if ui.button("Close").clicked() {
                        self.show_settings = false;
                    }
                });
        }

        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            // Split into top (plot) and bottom (editor/log)
            egui::TopBottomPanel::top("plot_panel")
                .resizable(true)
                .default_height(300.0)
                .show_inside(ui, |ui| {
                    ui.heading("📈 Membrane Potential");

                    let points: PlotPoints = self.recorded_data.iter()
                        .map(|p| [p.time, p.voltage])
                        .collect();

                    Plot::new("voltage_plot")
                        .height(ui.available_height() - 30.0)
                        .x_axis_label("Time (ms)")
                        .y_axis_label("Voltage (mV)")
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(points)
                                .name("Vm")
                                .color(egui::Color32::from_rgb(100, 200, 100)));
                        });
                });

            // Editor and Log tabs at bottom
            ui.separator();

            egui::TopBottomPanel::bottom("tabs_panel")
                .resizable(true)
                .default_height(200.0)
                .show_inside(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_label(true, "📝 Script");
                        ui.selectable_label(false, "📋 Output Log");
                    });
                    ui.separator();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add_sized(
                            [ui.available_width(), ui.available_height()],
                            egui::TextEdit::multiline(&mut self.script_content)
                                .font(egui::FontId::monospace(self.font_size))
                                .code_editor()
                        );
                    });
                });
        });
    }
}

/// Simple pseudo-random for demonstration
fn rand_simple(seed: f64) -> f64 {
    let x = (seed * 12345.6789).sin() * 43758.5453;
    x - x.floor()
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([1000.0, 700.0])
            .with_title("OldiesRules - Legacy Simulator Revival"),
        ..Default::default()
    };

    eframe::run_native(
        "OldiesRules",
        native_options,
        Box::new(|cc| Ok(Box::new(OldiesApp::new(cc)))),
    )
}
