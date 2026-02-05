from gemini_physics.group_theory import order_psl2_q


def test_order_psl2_7_is_168() -> None:
    assert order_psl2_q(7) == 168

