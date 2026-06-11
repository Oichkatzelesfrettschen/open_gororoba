#![cfg(feature = "cubecl")]

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use stats_core::ultrametric::{INF_DISTANCE, UltrametricCubeclKernel, minimax_distance_matrix_cpu};

fn seeded_graph(n_nodes: usize, seed: u64) -> Vec<f32> {
    let mut adjacency = vec![INF_DISTANCE; n_nodes * n_nodes];
    for node in 0..n_nodes {
        adjacency[node * n_nodes + node] = 0.0;
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    for node in 0..n_nodes {
        let next = (node + 1) % n_nodes;
        let weight = rng.random_range(0.05f32..1.0f32);
        adjacency[node * n_nodes + next] = weight;
        adjacency[next * n_nodes + node] = weight;
    }

    for _ in 0..(n_nodes * 3) {
        let a = rng.random_range(0..n_nodes);
        let mut b = rng.random_range(0..(n_nodes - 1));
        if b >= a {
            b += 1;
        }
        let weight = rng.random_range(0.05f32..1.0f32);
        adjacency[a * n_nodes + b] = weight;
        adjacency[b * n_nodes + a] = weight;
    }

    adjacency
}

#[test]
#[ignore = "requires local cubecl-wgpu adapter"]
fn ultrametric_cubecl_matches_cpu_for_seeded_64_node_graph() {
    if !UltrametricCubeclKernel::is_available() {
        eprintln!("cubecl-wgpu unavailable; skipping ultrametric minimax parity");
        return;
    }

    let n_nodes = 64;
    let adjacency = seeded_graph(n_nodes, 42);
    let cpu = minimax_distance_matrix_cpu(&adjacency, n_nodes)
        .unwrap_or_else(|err| panic!("CPU minimax failed: {err}"));
    let cubecl = UltrametricCubeclKernel::minimax_distance_matrix(&adjacency, n_nodes)
        .unwrap_or_else(|err| panic!("cubecl minimax failed: {err}"));
    assert_eq!(cubecl, cpu);
}
