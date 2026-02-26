use gororoba_cli_warp::warp_precision_suite_ops;

use std::error::Error;
use warp_precision_suite_ops::run_matrix_compat_args;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    run_matrix_compat_args(&std::env::args().collect::<Vec<_>>())
}
