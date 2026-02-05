# Harper-Hofstadter Model: Chern Number Calculation via FHS Method
# Based on Fukui, Hatsugai, Suzuki (2005)
# Implements lattice link variables and plaquette phases.

from math import gcd

import matplotlib.pyplot as plt
import numpy as np


def reduced_fracs(Qmax):
    out = []
    for q in range(2, Qmax+1):
        for p in range(1, q):
            if gcd(p,q)==1:
                out.append((p,q))
    out.sort(key=lambda pq: pq[0]/pq[1])
    return out

def harper_H(kx, ky, p, q):
    """
    Harper Hamiltonian H(k) for flux alpha = p/q.
    """
    alpha = p/q
    m = np.arange(q)
    diag = 2.0*np.cos(kx + 2.0*np.pi*alpha*m)
    H = np.zeros((q,q), dtype=np.complex128)
    np.fill_diagonal(H, diag)
    # nearest neighbors
    for j in range(q-1):
        H[j,j+1]=1.0; H[j+1,j]=1.0
    # wrap with ky-phase
    phase = np.exp(1j*q*ky)
    H[0,q-1] = np.conj(phase)
    H[q-1,0] = phase
    return H

def fhs_chern_per_band(p, q, N):
    """
    Calculates Chern numbers for each band using the FHS method.
    """
    kxs = np.linspace(0, 2*np.pi, N, endpoint=False)
    kys = np.linspace(0, 2*np.pi, N, endpoint=False)
    evecs = np.zeros((N,N,q,q), dtype=np.complex128)

    # Diagonalize H(k) on the grid
    for i,kx in enumerate(kxs):
        for j,ky in enumerate(kys):
            H = harper_H(kx, ky, p, q)
            w, v = np.linalg.eigh(H)
            evecs[i,j,:,:] = v.T # bands first

    def link(i,j,n,dirx):
        """Compute U(1) link variable."""
        if dirx:
            i2=(i+1)%N; j2=j
        else:
            i2=i; j2=(j+1)%N
        v1=evecs[i,j,n,:]; v2=evecs[i2,j2,n,:]
        ov=np.vdot(v1,v2)
        return ov/(np.abs(ov)+1e-15)

    ch=np.zeros(q, dtype=int)
    for n in range(q):
        Fsum=0.0
        for i in range(N):
            for j in range(N):
                Ux = link(i,j,n,True)
                Uy = link((i+1)%N,j,n,False)
                Ux_inv = np.conj(link(i,(j+1)%N,n,True))
                Uy_inv = np.conj(link(i,j,n,False))
                Fij = np.angle(Ux*Uy*Ux_inv*Uy_inv)
                Fsum += Fij
        ch[n]=int(np.rint(Fsum/(2*np.pi)))
    return ch

def generate_chern_map():
    print("--- Generating Harper Chern Number Map ---")
    Q_MAX = 9
    N_GRID = 17

    fracs = reduced_fracs(Q_MAX)
    band_cherns = []

    print(f"Computing for {len(fracs)} reduced fractions up to q={Q_MAX}...")

    for (p,q) in fracs:
        ch = fhs_chern_per_band(p,q,N_GRID)
        band_cherns.append(ch)

    # Build gap Chern numbers (cumulative sums)
    gap_cherns = []
    for ch in band_cherns:
        cum = np.cumsum(ch)
        gap_cherns.append(cum[:-1]) # q-1 gaps

    # Visualization
    max_gaps = max(len(gc) for gc in gap_cherns)
    Cgap = np.full((len(fracs), max_gaps), np.nan)
    for i,gc in enumerate(gap_cherns):
        Cgap[i,:len(gc)] = gc

    # Rendering
    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "axes.edgecolor": "#1f2937",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "font.size": 12
    })

    fig, ax = plt.subplots(figsize=(12, 10))
    vmax = np.nanmax(np.abs(Cgap))
    im = ax.imshow(Cgap.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_title("Harper Gap Chern Numbers (Cumulative Band Sums)", fontsize=18, color='white')
    ax.set_xlabel("Flux alpha = p/q", fontsize=14)
    ax.set_ylabel("Gap index r", fontsize=14)

    # Overlay Integers
    for i in range(Cgap.shape[0]):
        for j in range(Cgap.shape[1]):
            val = Cgap[i,j]
            if np.isfinite(val):
                ax.text(i,j,f"{int(val)}", ha="center", va="center", fontsize=8, color="white")

    plt.tight_layout()
    plt.savefig("data/artifacts/images/harper_chern_map_highres.png", dpi=300)
    print("Saved Chern Map.")

if __name__ == "__main__":
    generate_chern_map()
