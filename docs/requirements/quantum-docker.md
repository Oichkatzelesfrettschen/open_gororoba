# Requirements: Quantum (Qiskit via Docker)

Host Python may be too new for Qiskit wheels. Use Docker.

```bash
docker build -t qiskit-env -f docker/Dockerfile .
bin/run_quantum_container.sh src/quantum/local_test.py
```

Makefile equivalents:
```bash
make docker-quantum-build
make docker-quantum-run ARGS="src/quantum/local_test.py"
```

Optional local install (Python 3.11/3.12 only; markers skip installs on >=3.13):
```bash
make install-quantum
```

See also: `docs/QISKIT_SETUP.md`.
