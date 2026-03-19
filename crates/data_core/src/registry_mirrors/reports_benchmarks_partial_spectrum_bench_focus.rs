//! # Partial Spectrum Bench
//!
//! | family | size | objective | k | fastest solver | lowest max abs error |
//! | --- | ---: | --- | ---: | --- | --- |
//! | quantized_obstruction_graph | 16 | largest_abs | 1 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 16 | largest_abs | 2 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 16 | largest_abs | 4 | policy_full_truncated | partial_subspace |
//! | quantized_obstruction_graph | 16 | smallest_abs | 1 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 16 | smallest_abs | 2 | policy_full_truncated | partial_subspace |
//! | quantized_obstruction_graph | 16 | smallest_abs | 4 | policy_full_truncated | partial_subspace |
//! | quantized_obstruction_graph | 32 | largest_abs | 1 | partial_subspace | partial_subspace |
//! | quantized_obstruction_graph | 32 | largest_abs | 2 | partial_subspace | partial_subspace |
//! | quantized_obstruction_graph | 32 | largest_abs | 4 | partial_subspace | partial_subspace |
//! | quantized_obstruction_graph | 32 | smallest_abs | 1 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 32 | smallest_abs | 2 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 32 | smallest_abs | 4 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 64 | largest_abs | 1 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 64 | largest_abs | 2 | partial_subspace | partial_subspace |
//! | quantized_obstruction_graph | 64 | largest_abs | 4 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 64 | smallest_abs | 1 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 64 | smallest_abs | 2 | partial_subspace | reference_full_truncated |
//! | quantized_obstruction_graph | 64 | smallest_abs | 4 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 16 | largest_abs | 1 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 16 | largest_abs | 2 | reference_full_truncated | reference_full_truncated |
//! | real_obstruction | 16 | largest_abs | 4 | policy_full_truncated | partial_subspace |
//! | real_obstruction | 16 | smallest_abs | 1 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 16 | smallest_abs | 2 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 16 | smallest_abs | 4 | policy_full_truncated | reference_full_truncated |
//! | real_obstruction | 32 | largest_abs | 1 | partial_subspace | partial_subspace |
//! | real_obstruction | 32 | largest_abs | 2 | partial_subspace | partial_subspace |
//! | real_obstruction | 32 | largest_abs | 4 | partial_subspace | partial_subspace |
//! | real_obstruction | 32 | smallest_abs | 1 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 32 | smallest_abs | 2 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 32 | smallest_abs | 4 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 64 | largest_abs | 1 | partial_subspace | partial_subspace |
//! | real_obstruction | 64 | largest_abs | 2 | partial_subspace | partial_subspace |
//! | real_obstruction | 64 | largest_abs | 4 | partial_subspace | partial_subspace |
//! | real_obstruction | 64 | smallest_abs | 1 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 64 | smallest_abs | 2 | partial_subspace | reference_full_truncated |
//! | real_obstruction | 64 | smallest_abs | 4 | partial_subspace | reference_full_truncated |
//!
