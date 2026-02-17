use clap::Parser;
use gororoba_engine::thesis_pipelines::Thesis5Pipeline;
use gororoba_engine::traits::ThesisPipeline;
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Grid size
    #[arg(long, default_value_t = 16)]
    grid_size: usize,

    /// Events per bin
    #[arg(long, default_value_t = 1000)]
    events_per_bin: usize,

    /// Frustration scaling c_f
    #[arg(long, default_value_t = 10.0)]
    c_f: f64,

    /// Associator scaling c_a
    #[arg(long, default_value_t = 1.0)]
    c_a: f64,

    /// Initial state type: "triplet" or "singlet"
    #[arg(long, default_value = "triplet")]
    initial_state: String,

    /// Output file
    #[arg(long, default_value = "data/evidence/spin_decoherence.toml")]
    output: String,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    let pipeline = Thesis5Pipeline {
        grid_size: args.grid_size,
        events_per_bin: args.events_per_bin,
        c_f: args.c_f,
        c_a: args.c_a,
        initial_state: args.initial_state,
        ..Default::default()
    };

    let evidence = pipeline.execute();

    println!("Pipeline executed: {}", evidence.label);
    println!("Passes gate: {}", evidence.passes_gate);
    for msg in &evidence.messages {
        println!(" - {}", msg);
    }

    // Save evidence
    if let Some(parent) = Path::new(&args.output).parent() {
        fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    // Serialize evidence using TOML
    let content = toml::to_string_pretty(&evidence).expect("Failed to serialize evidence");
    fs::write(&args.output, content).expect("Failed to write output file");
}
