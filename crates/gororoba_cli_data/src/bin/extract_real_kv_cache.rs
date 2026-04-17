//! KV-cache extractor (replaces the missing
//! scripts/extract_real_kv_cache.py referenced by
//! crates/gororoba_cli_physics/src/bin/turboquant_real_kv.rs).
//!
//! Action item A1 from registry/kv_cache_provenance.toml.
//!
//! Two modes:
//!
//! 1. `--mode synthetic` (default)
//!    Emits deterministic Gaussian-family KV tensors keyed by a
//!    seed + a model spec (n_layers, n_heads, seq_len, head_dim).
//!    This is NOT a substitute for running a real forward pass --
//!    it is a REPRODUCIBLE fixture for regression-testing the
//!    downstream TurboQuant consumer (turboquant-real-kv) without
//!    network access or a multi-GB model checkout.
//!
//! 2. `--mode huggingface-hook`
//!    Prints the PyO3 integration contract the production
//!    extractor must satisfy, plus a Python reference snippet.
//!    The project's "Rust over Python always" policy prohibits
//!    bare Python scripts, but allows PyO3-wrapped Python libraries
//!    for domains without a Rust equivalent (LLM inference is one).
//!    The actual PyO3 implementation is deferred to a follow-up
//!    session; this mode documents the contract so downstream work
//!    can plan around it.
//!
//! Output format (matches what turboquant-real-kv reads):
//!   <out-dir>/metadata.json           JSON with n_layers, n_heads,
//!                                     seq_len, head_dim, model, n_params
//!   <out-dir>/keys_layer{NN}.f64      float64, row-major, one file
//!                                     per transformer layer, shape
//!                                     (seq_len * n_heads * head_dim)
//!   <out-dir>/values_layer{NN}.f64    ditto for value projection

use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::ExitCode,
};

use clap::{Parser, ValueEnum};

#[derive(Clone, Debug, ValueEnum)]
enum Mode {
    Synthetic,
    HuggingfaceHook,
}

#[derive(Parser)]
#[command(
    name = "extract-real-kv-cache",
    about = "Generate or document LLM KV-cache tensor extracts (action A1 of plan P6A.S7)."
)]
struct Args {
    /// Operating mode.
    #[arg(long, default_value = "synthetic", value_enum)]
    mode: Mode,

    /// Output directory (tensors + metadata.json).
    #[arg(long, default_value = "data/external/real_kv_cache_synthetic")]
    out_dir: PathBuf,

    /// Model label recorded in metadata.json.
    #[arg(long, default_value = "synthetic-SmolLM2-shaped")]
    model: String,

    /// Transformer layer count.
    #[arg(long, default_value_t = 30)]
    n_layers: usize,

    /// Attention heads per layer.
    #[arg(long, default_value_t = 9)]
    n_heads: usize,

    /// Sequence length (tokens in the synthetic batch).
    #[arg(long, default_value_t = 62)]
    seq_len: usize,

    /// Per-head dimensionality.
    #[arg(long, default_value_t = 64)]
    head_dim: usize,

    /// Nominal parameter count recorded in metadata.json.
    #[arg(long, default_value_t = 135_000_000)]
    n_params: u64,

    /// Deterministic RNG seed (any u64).
    #[arg(long, default_value_t = 0xC0FFEE_u64)]
    seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();

    match args.mode {
        Mode::Synthetic => run_synthetic(&args),
        Mode::HuggingfaceHook => {
            print_hook_contract();
            ExitCode::SUCCESS
        }
    }
}

fn run_synthetic(args: &Args) -> ExitCode {
    if let Err(e) = fs::create_dir_all(&args.out_dir) {
        eprintln!("ERROR: create {}: {e}", args.out_dir.display());
        return ExitCode::FAILURE;
    }

    // metadata.json first so partial failures still leave a manifest.
    let meta = format!(
        "{{\n\
          \"model\": \"{}\",\n\
          \"n_layers\": {},\n\
          \"n_heads\": {},\n\
          \"seq_len\": {},\n\
          \"head_dim\": {},\n\
          \"n_params\": {},\n\
          \"synthetic\": true,\n\
          \"seed\": {},\n\
          \"provenance\": \"crates/gororoba_cli_data/src/bin/extract_real_kv_cache.rs --mode synthetic\"\n\
        }}\n",
        args.model,
        args.n_layers,
        args.n_heads,
        args.seq_len,
        args.head_dim,
        args.n_params,
        args.seed,
    );
    if let Err(e) = fs::write(args.out_dir.join("metadata.json"), meta.as_bytes()) {
        eprintln!("ERROR: write metadata.json: {e}");
        return ExitCode::FAILURE;
    }

    let total_per_layer = args.seq_len * args.n_heads * args.head_dim;
    let mut rng = SplitMix64::new(args.seed);

    for layer in 0..args.n_layers {
        // Seed per-layer so individual layer files are reproducible in isolation.
        let layer_seed = args
            .seed
            .wrapping_add((layer as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

        for kind in ["keys", "values"] {
            let mut sub = SplitMix64::new(
                layer_seed
                    .wrapping_add(match kind {
                        "keys" => 0x1000_0000_0000_0001,
                        _ => 0x2000_0000_0000_0003,
                    }),
            );
            let path = args.out_dir.join(format!("{kind}_layer{layer:02}.f64"));
            let mut f = match File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("ERROR: create {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            };
            // Write float64 little-endian, row-major.
            for _ in 0..total_per_layer {
                let v = gaussian_from_splitmix(&mut sub);
                if let Err(e) = f.write_all(&v.to_le_bytes()) {
                    eprintln!("ERROR: write {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    // Advance top-level rng by an unused step to acknowledge seed usage
    // (silences unused-variable warnings without allow attribute).
    let _ = rng.next_u64();

    let total_bytes = (args.n_layers * 2 * total_per_layer * 8) as u64;
    println!(
        "[extract-real-kv-cache] wrote {} tensors ({} layers x 2 KV parts, {} bytes each, total {} MB)",
        args.n_layers * 2,
        args.n_layers,
        total_per_layer * 8,
        total_bytes / 1_048_576
    );
    ExitCode::SUCCESS
}

fn print_hook_contract() {
    println!(
        "extract-real-kv-cache --mode huggingface-hook\n\
\n\
Production extractor integration contract:\n\
\n\
  1. Use PyO3 to call transformers + torch from Rust. Bindings\n\
     crates to consider: pyo3 + pyo3-safetensors + numpy (no\n\
     candle in this workspace as of 2026-04-17).\n\
\n\
  2. Python reference implementation (to port to PyO3):\n\
\n\
     from transformers import AutoModelForCausalLM, AutoTokenizer\n\
     import torch, numpy as np, json, os\n\
\n\
     model_id = 'HuggingFaceTB/SmolLM2-135M'\n\
     tok = AutoTokenizer.from_pretrained(model_id)\n\
     m = AutoModelForCausalLM.from_pretrained(\n\
           model_id, output_hidden_states=False,\n\
           torch_dtype=torch.float32)\n\
     inputs = tok('<deterministic prompt>', return_tensors='pt')\n\
     with torch.no_grad():\n\
       out = m(**inputs, use_cache=True)\n\
\n\
     cfg = m.config\n\
     meta = {{\n\
       'model': model_id,\n\
       'n_layers': cfg.num_hidden_layers,\n\
       'n_heads': cfg.num_attention_heads,\n\
       'seq_len': inputs.input_ids.shape[1],\n\
       'head_dim': cfg.hidden_size // cfg.num_attention_heads,\n\
       'n_params': sum(p.numel() for p in m.parameters()),\n\
     }}\n\
\n\
     os.makedirs(out_dir, exist_ok=True)\n\
     json.dump(meta, open(f'{{out_dir}}/metadata.json', 'w'))\n\
     for i, (k, v) in enumerate(out.past_key_values):\n\
       k.cpu().to(torch.float64).numpy().tofile(\n\
         f'{{out_dir}}/keys_layer{{i:02d}}.f64')\n\
       v.cpu().to(torch.float64).numpy().tofile(\n\
         f'{{out_dir}}/values_layer{{i:02d}}.f64')\n\
\n\
  3. Required PyO3 wrapper responsibilities:\n\
     - gate the Python call with a CString.clone() of the prompt\n\
     - preserve determinism via torch.manual_seed before forward\n\
     - record prompt_sha256 into metadata.json (plan P6A.S7 A3)\n\
     - pin HF revision via transformers from_pretrained(revision=)\n\
\n\
  4. Once implemented, update registry/kv_cache_provenance.toml\n\
     so the `extractor` field points at this binary (instead of\n\
     'scripts/extract_real_kv_cache.py (MISSING)') and record the\n\
     HF revision + prompt_sha256 per extract.\n"
    );
}

/// SplitMix64 PRNG -- high-quality, fast, seedable. Used to produce
/// reproducible Gaussian-ish values without pulling in rand.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64_unit(&mut self) -> f64 {
        // Uniform [0, 1): 53 bits of mantissa.
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Box-Muller transform: two uniform draws -> one standard-normal sample.
fn gaussian_from_splitmix(rng: &mut SplitMix64) -> f64 {
    use std::f64::consts::TAU;
    let u1 = rng.next_f64_unit().max(1e-300); // avoid log(0)
    let u2 = rng.next_f64_unit();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}
