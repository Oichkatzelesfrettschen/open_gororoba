//! # Block Jacobi Backend Sweep
//!
//! | family | size | fastest solver | lowest max abs error |
//! | --- | ---: | --- | --- |
//! | clustered_pairs | 8 | block_jacobi_b4 | block_jacobi_b4 |
//! | clustered_pairs | 16 | reference_f64 | x87 |
//! | geometric_decay | 8 | block_jacobi_b4 | x87 |
//! | geometric_decay | 16 | reference_f64 | double_double |
//! | identity_nullspace_plateau | 8 | block_jacobi_b4 | block_jacobi_b4 |
//! | identity_nullspace_plateau | 16 | block_jacobi_b4 | reference_f64 |
//! | known_spectrum | 8 | block_jacobi_b4 | block_jacobi_b4 |
//! | known_spectrum | 16 | reference_f64 | double_double |
//! | obstruction_plateau | 8 | block_jacobi_b4 | double_double |
//! | obstruction_plateau | 16 | block_jacobi_b4 | double_double |
//! | quantized_obstruction_graph | 8 | block_jacobi_b4 | block_jacobi_b4 |
//! | quantized_obstruction_graph | 16 | reference_f64 | x87 |
//! | quantized_shell_permutation | 8 | block_jacobi_b4 | block_jacobi_b4 |
//! | quantized_shell_permutation | 16 | reference_f64 | reference_f64 |
//! | spiked_tail | 8 | block_jacobi_b4 | block_jacobi_b4 |
//! | spiked_tail | 16 | reference_f64 | double_double |
//!
