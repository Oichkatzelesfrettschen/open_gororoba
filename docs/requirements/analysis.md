# Requirements: Analysis (Topology + Embeddings)

These scripts use optional dependencies such as `networkx`, `ripser`, and friends.

```bash
make install-analysis
```

Notes:
- Common entrypoints that require these extras:
  - `src/topology_analysis.py` (ripser/persim)
  - `src/vis_box_kites.py` (scikit-learn)
  - `src/vis_advanced_projections.py` (scikit-learn, networkx)
  - `src/holo_tensor_net.py`, `src/vis_hyper_mera*.py` (networkx)
- Some scientific wheels may not exist for very new Python versions; if installs fail, use a container (or a Python 3.11/3.12 env) for this module.
