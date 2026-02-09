<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Cayley-Dickson Motif Census: CSV Schemas

This repo treats "motif census" as a reproducible export of connected components of a
diagonal-zero-product graph built from sparse 2-blades.

For a given dimension `dim = 2^n` we define the cross-assessor family:
- nodes are pairs `(low, high)` with `low in [1, dim/2 - 1]` and `high in [dim/2, dim-1]`.
- the two diagonals correspond to `(e_low +/- e_high)`.
- an undirected edge exists between assessors `a` and `b` if **any** sign choice makes the product
  exactly zero under the repo's CD multiplication convention.

Export entrypoint:
- `src/export_cd_motif_census.py`

## `data/csv/cd_motif_components_{dim}d.csv`

One row per connected component.

Columns:
- `dim` (int)
- `component_id` (int, 0-based within that `dim`)
- `node_count` (int)
- `edge_count` (int)
- `degree_min` (int)
- `degree_max` (int)
- `degree_mean` (float)
- `is_octahedron_k222` (bool) -- octahedron graph / K_{2,2,2}: 6 nodes, 12 edges, degree 4
- `k2_multipartite_part_count` (int) -- if nonzero, the component is a complete multipartite graph
  with all parts of size 2 (complement is a perfect matching); value is the number of 2-vertex parts
- `is_cuboctahedron` (bool) -- 12 nodes, 24 edges, degree 4
- `sampled` (bool) -- true if `--max-nodes` sampling was used
- `sample_max_nodes` (int or empty)
- `seed` (int)

## `data/csv/cd_motif_nodes_{dim}d.csv`

One row per node occurrence (nodes are unique per component).

Columns:
- `dim` (int)
- `component_id` (int)
- `low` (int)
- `high` (int)

This file is written only when `src/export_cd_motif_census.py` is run without `--summary-only`.

## `data/csv/cd_motif_edges_{dim}d.csv`

One row per undirected edge.

Columns:
- `dim` (int)
- `component_id` (int)
- `a_low` (int)
- `a_high` (int)
- `b_low` (int)
- `b_high` (int)

This file is written only when `src/export_cd_motif_census.py` is run without `--summary-only`.

Notes:
- This export intentionally omits per-edge sign-solution lists for now (keeps schema stable and
  small). If/when sign classes are needed (trefoil/zigzag generalizations), add a new column with a
  clearly versioned encoding.
