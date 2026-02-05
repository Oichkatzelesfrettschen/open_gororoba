# EXPERIMENT 1D -- Projector-FHS with polar link stabilization (QR/SVD) for gaps, q <= 11, Nk=17
# We compute non-Abelian link matrices via polar decomposition:
# For overlap W = V0^dagger V1 (rxr), take its unitary polar factor Q = U V^dagger from SVD(W)=U Sigma V^dagger, then link phase = det(Q).
# This is numerically unitary (|det(Q)|=1), stabilizing Chern sums on coarse meshes.
# Output: heatmap of (C_sum - C_pred) on wide gaps; dark mode, 3160x2820.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Rendering config
W, H = 3160, 2820
dpi = 100
fig_w, fig_h = W/dpi, H/dpi
plt.rcParams.update({
    "figure.facecolor": "#0d0f14",
    "axes.facecolor": "#0d0f14",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "font.size": 16
})

def harper_hamiltonian(kx, ky, p, q, t=1.0):
    alpha = p / q
    H = np.zeros((q, q), dtype=complex)
    for m in range(q):
        H[m, m] = 2.0 * t * np.cos(ky + 2.0*np.pi*alpha*m)
    hop = t * np.exp(1j*kx)
    for m in range(q):
        H[m, (m+1) % q] += hop
        H[(m+1) % q, m] += np.conj(hop)
    return H

def chern_sum_projector_polar(p, q, r, Nk=17):
    kxs = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    kys = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    frames = [[None for _ in range(Nk)] for _ in range(Nk)]
    evals = np.zeros((Nk, Nk, q), dtype=float)
    for i,kx in enumerate(kxs):
        for j,ky in enumerate(kys):
            w, v = np.linalg.eigh(harper_hamiltonian(kx, ky, p, q))
            frames[i][j] = v[:, :r]
            evals[i,j] = w
    phase_sum = 0.0
    for i in range(Nk):
        ip = (i+1)%Nk
        for j in range(Nk):
            jp = (j+1)%Nk
            V00 = frames[i][j]
            V10 = frames[ip][j]
            V11 = frames[ip][jp]
            V01 = frames[i][jp]
            # Overlaps
            W1 = V00.conj().T @ V10
            W2 = V10.conj().T @ V11
            W3 = V11.conj().T @ V01
            W4 = V01.conj().T @ V00
            # Polar unitary factors via SVD: W = U Sigma V^dagger => Q = U V^dagger
            def unitary_polar(W):
                U,S,Vh = np.linalg.svd(W, full_matrices=False)
                return U @ Vh
            Q1 = unitary_polar(W1)
            Q2 = unitary_polar(W2)
            Q3 = unitary_polar(W3)
            Q4 = unitary_polar(W4)
            loop = np.linalg.det(Q1) * np.linalg.det(Q2) * np.linalg.det(Q3) * np.linalg.det(Q4)
            phase_sum += np.angle(loop)
    Csum = int(np.rint(phase_sum/(2*np.pi)))
    # gap width estimate
    Emin = evals.min(axis=(0,1))
    Emax = evals.max(axis=(0,1))
    gap_wid = float(max(0.0, Emin[r] - Emax[r-1]))
    return Csum, gap_wid

def diophantine_pred(p,q,r):
    def egcd(a,b):
        if b==0: return (1,0,a)
        x,y,g = egcd(b, a%b)
        return (y, x-(a//b)*y, g)
    inv = egcd(p%q, q)[0] % q
    C = (inv * (r % q)) % q
    if C > q/2: C -= q
    return int(C)

gap_width_threshold = 0.10
records = []
print("Running Projector-FHS Polar calculation...")
for q in range(2, 12):       # q <= 11 to keep runtime modest with polar SVD per link
    for p in range(1, q):
        if np.gcd(p,q)!=1: continue
        for r in range(1, q):
            Csum, Wgap = chern_sum_projector_polar(p,q,r,Nk=17)
            if Wgap < gap_width_threshold:
                continue
            Cpred = diophantine_pred(p,q,r)
            diff = Csum - Cpred
            records.append((p/q, p, q, r, Wgap, Csum, Cpred, diff))

df = pd.DataFrame(records, columns=["alpha","p","q","r","gap_width","C_sum","C_pred","C_diff"])
df["rnorm"] = df["r"]/df["q"]
pivot = df.pivot_table(index="rnorm", columns="alpha", values="C_diff", aggfunc="mean")

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
ax = plt.gca()
im = ax.imshow(pivot.values, origin="lower", aspect="auto",
               extent=[float(pivot.columns.min()), float(pivot.columns.max()),
                       float(pivot.index.min()), float(pivot.index.max())],
               vmin=-0.5, vmax=0.5, cmap="coolwarm")
ax.set_xlabel(r"Flux $\alpha=p/q$")
ax.set_ylabel(r"Normalized gap index $r/q$")
ax.set_title("Projector-FHS (polar-stabilized) vs Diophantine\n"
             "q<=11, Nk=17, gap width >= 0.10", fontsize=28, loc="left")
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label(r"$C_{\mathrm{sum}} - C_{\mathrm{pred}}$")

vals = pivot.values
finite_mask = np.isfinite(vals)
n_tot = int(finite_mask.sum())
n_ok = int(np.sum(np.abs(vals[finite_mask]) < 0.25))
ax.text(0.01, 1.02, f"Exact matches: {n_ok}/{n_tot} (tol 0.25)", transform=ax.transAxes, fontsize=16)

plt.tight_layout()
outp = "data/artifacts/images/projector_fhs_polar_q11_3160x2820.png"
plt.savefig(outp, dpi=dpi, facecolor="#0d0f14")
print("Saved:", outp)
