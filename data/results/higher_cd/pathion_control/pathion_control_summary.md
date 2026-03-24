# Pathion Control Summary

## Summary

- Algebra: `Pathion`
- Ambient dimension: `32`
- Requested `V_k` rank cap: `20`
- Actual extracted rank: `1`
- Assessor count: `210`
- ZD graph edges: `465`
- Connected components: `2`
- Positive Laplacian eigenvalues: `30`
- Leading/trailing singular values: `2.033735180020` / `2.033735180020`

## Method

This control-lane bundle is derived stepwise in pure Rust from
`cd_kernel` sign/associator primitives, `extract_vk_basis`, and the
dimension-parametric control report assembly in
`algebra_experimental::higher_cd_control`.

## Interpretation

Pathion remains a higher-CD control/falsification lane. These outputs are
derived support artifacts for the Cayley-Dickson stack, not the primary
bridge architecture.
