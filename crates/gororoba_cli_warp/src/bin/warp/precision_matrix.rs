use gororoba_cli_warp::warp_precision_suite_ops;

use std::error::Error;
use warp_precision_suite_ops::run_matrix_compat_args;

pub fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    run_matrix_compat_args(args)
}
