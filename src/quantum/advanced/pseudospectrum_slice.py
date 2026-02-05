# BLOCK D -- Pseudospectrum slice: smallest singular value of (zI - L\lambda)
# Dark mode, 3160x2820.

import matplotlib.pyplot as plt
import numpy as np

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

print("Running Pseudospectrum Slice...")

# Build Dirichlet Laplacian and fractional powers
N = 80
h = 1.0/(N+1)
diag = 2.0*np.ones(N)/h**2
off = -1.0*np.ones(N-1)/h**2
L = np.diag(diag) + np.diag(off,1) + np.diag(off,-1)

w, V = np.linalg.eigh(L)
def Lpow(s):
    return (V * (w**s)) @ V.T

s0, s1, s2 = 1.0, 0.7, 0.4
c0, c1, c2 = 1.0, 0.5, 0.3
lam = 0.7
Llam = c0*Lpow(s0) + lam*c1*Lpow(s1) + (lam**2)*c2*Lpow(s2)
I = np.eye(N)

# Grid and smin computation
Re_vals = np.linspace(0.0, 40.0, 20)
Im_vals = np.linspace(-12.0, 12.0, 20)
smin = np.zeros((Im_vals.size, Re_vals.size))
for i, Imz in enumerate(Im_vals):
    for j, Rez in enumerate(Re_vals):
        z = Rez + 1j*Imz
        A = z*I - Llam
        sval = np.linalg.svd(A, compute_uv=False).min()
        smin[i, j] = sval

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
ax = plt.gca()
Z = np.log10(smin + 1e-14)
im = ax.imshow(Z, origin='lower', aspect='auto',
               extent=[Re_vals.min(), Re_vals.max(), Im_vals.min(), Im_vals.max()])
ax.set_xlabel("Re z", fontsize=22)
ax.set_ylabel("Im z", fontsize=22)
ax.set_title("Pseudospectrum Slice: log10 min_svd(zI - L_lambda)\n"
             f"lambda={lam}, s=(1,0.7,0.4), N={N}", fontsize=30, loc="left")
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("log10 min_svd", fontsize=16)
ax.grid(False)
plt.tight_layout()
outp = "data/artifacts/images/pseudospectrum_slice_3160x2820.png"
plt.savefig(outp, dpi=dpi, facecolor="#0d0f14")
plt.show()
print("Saved:", outp)
