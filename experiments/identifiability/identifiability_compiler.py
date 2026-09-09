#!/usr/bin/env python3
"""Synthetic identifiability experiment compiler; not physical evidence."""
import numpy as np

def orth_residual(target, nuisance):
    t=np.asarray(target,float).reshape(-1,1); n=np.asarray(nuisance,float)
    coef,*_=np.linalg.lstsq(n,t,rcond=None)
    return (t-n@coef).ravel()

def diagnostics(target,nuisance,sigma=1.0):
    t=np.asarray(target,float); r=orth_residual(t,nuisance)
    frac=np.linalg.norm(r)/np.linalg.norm(t)
    return {"identifiable_fraction":float(frac),"principal_angle_deg":float(np.degrees(np.arcsin(np.clip(frac,0,1)))),"efficient_fisher_info":float(r@r/sigma**2)}

def commutator_problem():
    rows=[(a,b,q,k) for a in (-1.,-.5,.5,1.) for b in (-1.,-.5,.5,1.) for q in (-1.,1.) for k in (-1.,0.,1.)]
    a,b,q,k=np.asarray(rows).T; target=q*a*b
    n=np.column_stack([np.ones(len(a)),a,b,a*b,a*a,b*b,k,k*k,q*a,q*b,q*(a*a-b*b),q*k,q*a*k,q*b*k])
    return target,n

def boundary_problem():
    rows=[(d,200./d,v,s,c) for d in np.arange(100.,401.,25.) for v in (-.15,-.075,0.,.075,.15) for s in (-1.,1.) for c in (-1.,1.)]
    d,z,v,s,c=np.asarray(rows).T
    shape=z**4*(1+.55*np.exp(-((d-220.)/42.)**2)-.25*np.exp(-((d-135.)/30.)**2))
    target=s*c*shape
    n=np.column_stack([np.ones(len(d)),z,z*z,z**4,v,v*v*z*z,s,s*z*z,s*v*z*z,s*v*v*z*z,s*z**5,s*z**6,c,c*z*z,c*v*z*z,s*c,s*c*z*z,s*c*v*z*z])
    return d,z,v,s,c,target,n

if __name__ == "__main__":
    t,n=commutator_problem(); print("commutator",diagnostics(t,n,0.08))
    *_,t,n=boundary_problem(); print("boundary",diagnostics(t,n,0.12))
