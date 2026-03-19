//! # Block Jacobi Backend Sweep
//!
//! | family | size | fastest solver | lowest max abs error |
//! | --- | ---: | --- | --- |
//! | quantized_obstruction_graph | 16 | reference_f64 | x87 |
//! | quantized_obstruction_graph | 32 | reference_f64 | double_double |
//! | quantized_obstruction_graph | 64 | reference_f64 | double_double |
//! | quantized_shell_permutation | 16 | reference_f64 | reference_f64 |
//! | quantized_shell_permutation | 32 | reference_f64 | double_double |
//! | quantized_shell_permutation | 64 | reference_f64 | double_double |
//! | real_obstruction | 16 | reference_f64 | x87 |
//! | real_obstruction | 32 | reference_f64 | x87 |
//! | real_obstruction | 64 | reference_f64 | double_double |
//!
