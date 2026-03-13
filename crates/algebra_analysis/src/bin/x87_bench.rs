//! Placeholder execution target that reserves the `x87_bench` workspace name.
//!
//! Real benchmark runs for E-188/E-189 live in `benches/x87_bench.rs` and
//! should be executed with `cargo bench -p algebra_analysis --bench x87_bench`.

fn main() {
    eprintln!(
        "Use `cargo bench -p algebra_analysis --bench x87_bench -- <filter>` for E-188/E-189."
    );
    std::process::exit(2);
}
