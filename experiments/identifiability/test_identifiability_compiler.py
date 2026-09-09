import numpy as np
import identifiability_compiler as ic

def test_exact_target_nuisance_collapses_identifiability():
    t,n=ic.commutator_problem()
    n_adv=np.column_stack([n,t])
    assert ic.diagnostics(t,n_adv,0.08)["identifiable_fraction"] < 1e-10

def test_commutator_target_is_distinct_from_declared_nuisances():
    t,n=ic.commutator_problem()
    d=ic.diagnostics(t,n,0.08)
    assert d["identifiable_fraction"] > 0.9
    assert d["principal_angle_deg"] > 60.0
