//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->
//!
//! # Requirements: Quantum (Qiskit via Docker)
//!
//! Host Python may be too new for Qiskit wheels. Use Docker.
//!
//! ```texttext
//! docker build -t qiskit-env -f docker/Dockerfile .
//! ./run_quantum_container.sh src/quantum/local_test.py
//! ```texttext
//!
//! Makefile equivalents:
//! ```texttext
//! make docker-quantum-build
//! make docker-quantum-run ARGS="src/quantum/local_test.py"
//! ```texttext
//!
//! Optional local install (Python 3.11/3.12 only; markers skip installs on >=3.13):
//! ```texttext
//! make install-quantum
//! ```texttext
//!
//! See also: `docs/QISKIT_SETUP.md`.
//!
