//! detect_worker_budget: Pure Rust port of `scripts/detect_worker_budget.sh`.
//! Prints the safe parallel worker count based on host resources.

fn main() {
    let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    println!("{}", cores);
}
