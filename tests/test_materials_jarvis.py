import numpy as np

from gemini_physics.materials_jarvis import jarvis_subset_to_dataframe


def test_jarvis_subset_derives_formula_elements_and_volume() -> None:
    records = [
        {
            "jid": "JVASP-X",
            "formation_energy_peratom": -0.1,
            "atoms": {
                "lattice_mat": np.eye(3).tolist(),
                "coords": [],
                "elements": ["Ag", "Te", "Te", "Tl"],
                "abc": [],
                "angles": [],
                "cartesian": True,
                "props": {},
            },
        }
    ]
    df = jarvis_subset_to_dataframe(records, n=1, seed=0)
    row = df.iloc[0].to_dict()
    assert row["jid"] == "JVASP-X"
    assert row["elements"] == "Ag,Te,Tl"
    assert row["nelements"] == 3
    assert row["formula"] == "AgTe2Tl"
    assert np.isclose(row["volume"], 1.0)

