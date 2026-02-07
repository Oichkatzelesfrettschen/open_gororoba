# EXPERIMENT 4B -- Expand CSS stabilizer families (12 X, 12 Z), compute ranks and search for a low-weight logical
# Strategy: Use Sierpinski mask; build 12 translations each; compute H_X, H_Z ranks; search for X-logical
# among even-parity translates (coarse grid). Render candidate support if found.
# Dark mode, 3160x2820.

import matplotlib.pyplot as plt
import numpy as np

# Rendering
W, H = 3160, 2820; dpi=100; fig_w, fig_h=W/dpi, H/dpi
plt.rcParams.update({
    "figure.facecolor":"#0d0f14",
    "axes.facecolor":"#0d0f14",
    "axes.edgecolor":"#1f2937",
    "axes.labelcolor":"white",
    "xtick.color":"white",
    "ytick.color":"white",
    "text.color":"white",
    "font.size":16
})

L=33
S = np.zeros((L,L), dtype=np.uint8)
S[0, L//2] = 1
for r in range(1, L):
    left = np.roll(S[r-1], 1, axis=0)
    right = np.roll(S[r-1], -1, axis=0)
    S[r] = left ^ right

coords = np.add.outer(np.arange(L), np.arange(L))
X_parity = ((coords % 2) == 0).astype(np.uint8)
Z_parity = 1 - X_parity

def translate(mask, tx, ty):
    return np.roll(np.roll(mask, ty, axis=0), tx, axis=1)

# 12 shifts each (even steps to preserve parity split)
X_shifts = [(2*a, 2*b) for a,b in [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(3,0),(0,3),(3,2),(2,3),(4,1),(1,4)]]
Z_shifts = [(2*a, 2*b) for a,b in [(1,0),(0,1),(2,0),(0,2),(3,1),(1,3),(2,2),(4,0),(0,4),(3,3),(4,2),(2,4)]]

X_stabs = [(translate(S, tx, ty) & X_parity) for (tx,ty) in X_shifts]
Z_stabs = [(translate(S, tx, ty) & Z_parity) for (tx,ty) in Z_shifts]

n = L*L
def flat(m): return m.reshape(-1) % 2
HX = np.array([flat(m) for m in X_stabs], dtype=np.uint8)
HZ = np.array([flat(m) for m in Z_stabs], dtype=np.uint8)

# GF(2) rank
def gf2_rank(A):
    A = A.copy().astype(np.uint8)
    m,n = A.shape
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i,c]:
                piv = i; break
        if piv is None:
            continue
        if piv != r:
            A[[r,piv]] = A[[piv,r]]
        for i in range(m):
            if i != r and A[i,c]:
                A[i] ^= A[r]
        r += 1
        if r == m: break
    return r

print("Computing GF(2) Ranks for Fractal Code...")
rank_HX = gf2_rank(HX)
rank_HZ = gf2_rank(HZ)
k = n - rank_HX - rank_HZ

# Candidate X-logical search among even-parity translates (coarse step=2)
best = None
for tx in range(0, L, 2):
    for ty in range(0, L, 2):
        xmask = translate(S, tx, ty) & X_parity
        xrow = flat(xmask)
        # must commute with all Z stabilizers
        if np.any((xrow @ HZ.T) % 2):
            continue
        # must be independent of H_X
        HX_ext = np.vstack([HX, xrow])
        if gf2_rank(HX_ext) == rank_HX:
            continue
        w = int(xrow.sum())
        if (best is None) or (w < best[0]):
            best = (w, tx, ty, xmask)

if best is None:
    print("No candidate X-logical found in this restricted family.")
else:
    w_best, tx_best, ty_best, Xlog = best
    print(f"Found X-Logical: w={w_best}, shift=({tx_best},{ty_best})")
    # Render
    img = np.zeros((L, L, 3), dtype=float)
    blue = np.array([0.38, 0.65, 0.96])
    img += Xlog[..., None] * blue
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = plt.gca()
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_title("Fractal CSS (Expanded): Candidate X-logical Support (Blue)\n"
                 f"L={L}, n={n}, rank(HX)={rank_HX}, rank(HZ)={rank_HZ}, k={k}, weight={w_best}",
                 fontsize=28, loc="left")
    ax.set_xlabel("Lattice X index"); ax.set_ylabel("Lattice Y index")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    outp = "data/artifacts/images/css_fractal_candidate_logical_expanded_3160x2820.png"
    plt.savefig(outp, dpi=dpi, facecolor="#0d0f14")
    plt.close()
    print("Saved:", outp)
