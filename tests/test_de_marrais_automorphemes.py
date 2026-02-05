import itertools
from collections import Counter

from gemini_physics.de_marrais_boxkites import (
    O_TRIPS,
    automorpheme_assessors,
    automorphemes,
    automorphemes_containing_assessor,
    primitive_assessors,
    production_rule_3,
)


def test_o_trips_form_fano_plane_lines() -> None:
    assert len(O_TRIPS) == 7
    assert len(set(O_TRIPS)) == 7
    for t in O_TRIPS:
        assert len(t) == 3
        assert set(t) <= set(range(1, 8))

    # Each point lies on exactly 3 lines; any two points determine exactly 1 line.
    point_counts = Counter(p for t in O_TRIPS for p in t)
    assert sorted(point_counts.keys()) == list(range(1, 8))
    assert set(point_counts.values()) == {3}

    pairs = [
        tuple(sorted(pair)) for t in O_TRIPS for pair in itertools.combinations(t, 2)
    ]
    pair_counts = Counter(pairs)
    assert len(pair_counts) == 21
    assert set(pair_counts.values()) == {1}


def test_automorphemes_cover_primitive_assessors_twice() -> None:
    autos = automorphemes()
    assert len(autos) == 7
    assert sorted(len(a) for a in autos) == [12] * 7

    primitive = set(primitive_assessors())
    union = set().union(*autos)
    assert union == primitive

    membership_counts = Counter(a for s in autos for a in s)
    assert set(membership_counts.values()) == {2}
    assert set(membership_counts.keys()) == primitive

    # "Behind the 8-ball": no automorpheme contains any assessor with high==8 or high==8^low.
    for low in range(1, 8):
        assert automorphemes_containing_assessor((low, 8)) == []
        assert automorphemes_containing_assessor((low, 8 ^ low)) == []


def test_production_rule_3_returns_unique_other_automorpheme() -> None:
    for o_trip in O_TRIPS:
        assessors = automorpheme_assessors(o_trip)
        for a in assessors:
            other = production_rule_3(o_trip, a)
            assert other != o_trip
            assert a in automorpheme_assessors(other)
            assert sorted(automorphemes_containing_assessor(a)) == sorted([o_trip, other])
