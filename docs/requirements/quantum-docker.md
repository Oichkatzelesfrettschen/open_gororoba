<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Quantum (Qiskit via Docker)

Host Python may be too new for Qiskit wheels. Use Docker.

```ignore
docker build -t qiskit-env -f docker/Dockerfile .
./run_quantum_container.sh src/quantum/local_test.py
```

Makefile equivalents:

```ignore
make docker-quantum-build
make docker-quantum-run ARGS="src/quantum/local_test.py"
```

Optional local install on Python 3.11/3.12 only:

```ignore
make install-quantum
```
