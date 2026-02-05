from __future__ import annotations

from gemini_physics.nilpotent_orbits import jordan_type_nilpotent, nilpotency_index


def _jordan_nilpotent_block(size: int) -> list[list[int]]:
    """
    Strictly upper-shift nilpotent Jordan block J_size(0) with ones on superdiagonal.
    """

    a = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size - 1):
        a[i][i + 1] = 1
    return a


def _block_diag(blocks: list[list[list[int]]]) -> list[list[int]]:
    n = sum(len(b) for b in blocks)
    out = [[0 for _ in range(n)] for _ in range(n)]
    off = 0
    for b in blocks:
        m = len(b)
        for i in range(m):
            for j in range(m):
                out[off + i][off + j] = b[i][j]
        off += m
    return out


def test_nilpotency_index_and_jordan_type() -> None:
    # Blocks: 3,2,1 -> n=6.
    a = _block_diag(
        [
            _jordan_nilpotent_block(3),
            _jordan_nilpotent_block(2),
            _jordan_nilpotent_block(1),
        ]
    )

    assert nilpotency_index(a) == 3
    jt = jordan_type_nilpotent(a)
    assert jt.blocks == (3, 2, 1)
