from __future__ import annotations

from gemini_physics.cd_motif_census import (
    motif_components_for_cross_assessors,
    xor_bucket,
)
from gemini_physics.de_marrais_boxkites import box_kites, primitive_assessors


def test_cd_motif_census_16d_matches_de_marrais_box_kites() -> None:
    comps = motif_components_for_cross_assessors(16)
    assert len(comps) == 7
    assert sorted(len(c.nodes) for c in comps) == [6] * 7
    assert all(c.is_octahedron_graph() for c in comps)

    # Cross-check the 42-node set agrees with the de Marrais primitive assessors.
    union = set().union(*(c.nodes for c in comps))
    assert union == set(primitive_assessors())
    assert set().union(*(bk.assessors for bk in box_kites())) == union


def test_cd_motif_edges_respect_xor_bucket_necessity() -> None:
    # The XOR bucket condition is necessary for a 2-blade diagonal cancellation.
    for comp in motif_components_for_cross_assessors(16):
        for a, b in comp.edges:
            assert xor_bucket(a) == xor_bucket(b)


def test_cd_motif_census_32d_has_k2_multipartite_components() -> None:
    comps = motif_components_for_cross_assessors(32)
    assert comps
    assert any(c.k2_multipartite_part_count() > 0 for c in comps)
