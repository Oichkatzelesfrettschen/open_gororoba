mod warp_precision_suite_ops;
mod warp_runner;

use std::error::Error;
use warp_precision_suite_ops::run_bench_compat_args;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    run_bench_compat_args(&std::env::args().collect::<Vec<_>>())
}
