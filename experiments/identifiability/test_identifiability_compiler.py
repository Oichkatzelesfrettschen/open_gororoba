import importlib.util
from pathlib import Path
import numpy as np

MODULE = Path(__file__).with_name("identifiability_compiler.py")
SPEC = importlib.util.spec_from_file_location("identifiability_compiler", MODULE)
ic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ic)

def test_exact_target_nuisance_collapses_identifiability():
    t,n=ic.commutator_problem(); n_adv=np.column_stack([n,t])
    assert ic.diagnostics(t,n_adv,0.08)["identifiable_fraction"] < 1e-10

def test_commutator_target_is_distinct_from_declared_nuisances():
    t,n=ic.commutator_problem(); d=ic.diagnostics(t,n,0.08)
    assert d["identifiable_fraction"] > 0.9
    assert d["principal_angle_deg"] > 60.0

def test_boundary_full_grid_loses_most_information_under_adversarial_expansion():
    d,z,v,s,c,t,n=ic.boundary_problem()
    base=ic.diagnostics(t,n,0.12)["identifiable_fraction"]
    n_adv=np.column_stack([n,s*c*z**3,s*c*z**5,s*c*np.exp(-((d-220.)/60.)**2)])
    expanded=ic.diagnostics(t,n_adv,0.12)["identifiable_fraction"]
    assert expanded < 0.1 * base
