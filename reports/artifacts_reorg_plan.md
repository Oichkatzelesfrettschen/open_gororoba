# Artifacts reorganization plan (non-destructive)

This report proposes a future migration for legacy flat files under data/artifacts/.
It does not move any files. It is safe to regenerate.

## Summary

- Flat files (excluding known index docs): 13
- Referenced somewhere in repo text: 0
- Present in artifacts manifest: 0

## Proposed moves

Rule of thumb: only move files with ref_count==0 and not in the manifest first.

## Phase buckets (current state)

- Phase 0 candidates (safe): 13
- Phase 1 candidates (referenced): 0
- Phase 2 candidates (manifested): 0

- data/artifacts/3d_time_evolution_mosaic.png -> data/artifacts/images/3d_time_evolution_mosaic.png (0.05 MiB)
  - sha256: 6d29c9f91473980e973e0e8f6a3d8d7c2521a6f9dbc39b9e5290e473f1956d2f
- data/artifacts/4d_entropy_mosaic.png -> data/artifacts/images/4d_entropy_mosaic.png (0.05 MiB)
  - sha256: 6d29c9f91473980e973e0e8f6a3d8d7c2521a6f9dbc39b9e5290e473f1956d2f
- data/artifacts/entropy_pde_nd.png -> data/artifacts/images/entropy_pde_nd.png (0.04 MiB)
  - sha256: 24662bb302691697d96a9619944d6338e2b07f84a9e51af1b1970443c7cdc263
- data/artifacts/extracted_equations.md -> data/artifacts/reports/extracted_equations.md (0.00 MiB)
  - sha256: 0b8c09e8f986f3b2b2f3bc0391690d94b8e4c6d399f48b34b28d36c058784ce2
- data/artifacts/m3_tensor_log.txt -> data/artifacts/reports/m3_tensor_log.txt (0.00 MiB)
  - sha256: acdc0707764c4d4eec6d3877f429684e486aa7d6b0d612dc2d9a73a4b9b75cb2
- data/artifacts/modular_chaos_plot.png -> data/artifacts/images/modular_chaos_plot.png (0.07 MiB)
  - sha256: a3bff21412949b1b9eb4dede32475776e7d22853769f92019b5f93c1942fdbac
- data/artifacts/nonlocal_slope_nd.png -> data/artifacts/images/nonlocal_slope_nd.png (0.04 MiB)
  - sha256: 52ce2ba5692cada513d9d0a78eebba3537f645cf56f454f7f7f98501b3df44ff
- data/artifacts/reality_check_and_synthesis.md -> data/artifacts/reports/reality_check_and_synthesis.md (0.00 MiB)
  - sha256: a7f0cac2602404c08960ad5d4176c1727759f1b08a8c5c7714f3c38a576a60aa
- data/artifacts/sedenion_field_3D_plot.png -> data/artifacts/images/sedenion_field_3D_plot.png (0.04 MiB)
  - sha256: 92e663fd1a187214502a4c6e2f7e95a1126b5b5b6823963b564194b7f3b481b3
- data/artifacts/sedenion_field_3d_plot.png -> data/artifacts/images/sedenion_field_3d_plot.png (0.04 MiB)
  - sha256: f7dcd83cefe95a6b638389089857b6e56ee2f03e41d554b27d951806132bcbd2
- data/artifacts/sedenion_field_4D_plot.png -> data/artifacts/images/sedenion_field_4D_plot.png (0.04 MiB)
  - sha256: 10ed02b3c2faf6dd7f857dcd688028b99703f7f9405477d0ed8c7afd8d13619c
- data/artifacts/spectral_flow_plot.png -> data/artifacts/images/spectral_flow_plot.png (0.05 MiB)
  - sha256: e1f78522b22c50331f20264187a036cc7411dadf48d67d81b00c41a29a627306
- data/artifacts/tscp_predictions_v1.csv -> data/artifacts/tables/tscp_predictions_v1.csv (2.87 MiB)
  - sha256: 81c7ad27aa5af167d959570c001e28eb4c904a12bd6d7dfdbd97c9473d8ca4d7

## Suggested migration phases

1. Phase 0 (safe): move unreferenced + unmanifested files only.
2. Phase 1: for referenced files, update references then move.
3. Phase 2: for manifested files, update ARTIFACTS_MANIFEST.csv entries (append-only policy), then move.
   If needed, keep a legacy redirect (or symlink if acceptable) during transition.

