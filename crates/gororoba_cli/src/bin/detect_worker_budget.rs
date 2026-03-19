//! detect_worker_budget: Pure Rust port of `scripts/detect_worker_budget.sh`.
//! Prints the safe parallel worker count based on host resources.

fn main() {
    println!("{}", verified_core::get_worker_budget());
}
