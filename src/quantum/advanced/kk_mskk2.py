# EXPERIMENT 3B -- True MSKK (two-anchor subtractive Kramers-Kronig) vs SSKK vs Standard
# Implement the 2-anchor formula:
#   \epsilon1(omega) = \epsilon1(omega0) + A(omega) [\epsilon1(omega1)-\epsilon1(omega0)] + (2/\pi) (omega^2-omega0^2)(omega^2-omega1^2) *
#            PV int omega' \epsilon2(omega') / ((omega'^2-omega^2)(omega'^2-omega0^2)(omega'^2-omega1^2)) domega'
# where A(omega) = (omega^2-omega0^2)/(omega1^2-omega0^2).
# Compare RMS errors to SSKK and Standard on three bands; dark mode, 3160x2820.

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

def eps_HN(omega, eps_inf=2.05, delta_eps=3.6, tau=7e-4, alpha=0.74, beta=0.65, sigma_dc=0.0):
    wta = (1j*omega*tau)**alpha
    hn = delta_eps / (1 + wta)**beta
    eps0 = 8.8541878128e-12
    cond = 1j * sigma_dc/(eps0*omega + 1e-300)
    return eps_inf + hn + cond

def kk_standard_from_im(omega, im_eps, omega_eval, eps_inf=0.0):
    w = omega; f = im_eps
    out = np.zeros_like(omega_eval, dtype=float)
    for idx, we in enumerate(omega_eval):
        denom = (w**2 - we**2)
        k = np.argmin(np.abs(w - we))
        denom[k] = np.inf
        out[idx] = eps_inf + (2/np.pi) * np.trapezoid(w * f / denom, w)
    return out

def kk_sskk_from_im(omega, im_eps, omega_eval, omega0, eps1_omega0):
    w = omega; f = im_eps
    out = np.zeros_like(omega_eval, dtype=float)
    for idx, we in enumerate(omega_eval):
        denom = (w**2 - we**2)*(w**2 - omega0**2)
        k1 = np.argmin(np.abs(w - we)); k2 = np.argmin(np.abs(w - omega0))
        denom[k1] = np.inf; denom[k2] = np.inf
        out[idx] = eps1_omega0 + (2*(we**2 - omega0**2)/np.pi) * np.trapezoid(w * f / denom, w)
    return out

def kk_mskk2_from_im(omega, im_eps, omega_eval, omega0, omega1, eps1_omega0, eps1_omega1):
    w = omega; f = im_eps
    out = np.zeros_like(omega_eval, dtype=float)
    k0 = np.argmin(np.abs(w - omega0))
    k1 = np.argmin(np.abs(w - omega1))
    for idx, we in enumerate(omega_eval):
        A = (we**2 - omega0**2)/(omega1**2 - omega0**2)
        denom = (w**2 - we**2)*(w**2 - omega0**2)*(w**2 - omega1**2)
        k_we = np.argmin(np.abs(w - we))
        denom[k_we] = np.inf; denom[k0] = np.inf; denom[k1] = np.inf
        integral = (2/np.pi) * (we**2 - omega0**2)*(we**2 - omega1**2) * np.trapezoid(w * f / denom, w)
        out[idx] = eps1_omega0 + A*(eps1_omega1 - eps1_omega0) + integral
    return out

print("Running MSKK-2 (Multi-Anchor KK) comparison...")

# Truth on wide band
W_true = np.logspace(0, 7, 20001)
eps_true = eps_HN(W_true)
eps1_true = eps_true.real
eps2_true = eps_true.imag

bands = [(1e1, 1e6), (3e1, 3e5), (1e2, 1e5)]
omega_eval = np.logspace(np.log10(1e2), np.log10(1e5), 600)

def rms_rel(a, b):
    denom = np.maximum(np.abs(b), 1e-12)
    return float(np.sqrt(np.mean(((a-b)/denom)**2)))

results = []
for (wmin, wmax) in bands:
    mask = (W_true>=wmin) & (W_true<=wmax)
    W_meas = W_true[mask]
    eps2_meas = eps2_true[mask]
    # Standard
    eps1_std = kk_standard_from_im(W_meas, eps2_meas, omega_eval, eps_inf=2.05)
    # SSKK
    w0 = np.sqrt(wmin*wmax)
    eps1_w0 = float(np.interp(w0, W_true, eps1_true))
    eps1_ss = kk_sskk_from_im(W_meas, eps2_meas, omega_eval, w0, eps1_w0)
    # MSKK with two anchors: choose w0 (center) and w1 at geometric center of lower decade
    w1 = np.sqrt(wmin*np.sqrt(wmin*wmax))
    eps1_w1 = float(np.interp(w1, W_true, eps1_true))
    eps1_ms = kk_mskk2_from_im(W_meas, eps2_meas, omega_eval, w0, w1, eps1_w0, eps1_w1)
    true_eval = np.interp(omega_eval, W_true, eps1_true)
    results.append((f"[{wmin:.0e},{wmax:.0e}]",
                    rms_rel(eps1_std, true_eval),
                    rms_rel(eps1_ss,  true_eval),
                    rms_rel(eps1_ms,  true_eval)))

# Bar chart
labels = [r[0] for r in results]
r_std = [r[1] for r in results]
r_ss  = [r[2] for r in results]
r_ms  = [r[3] for r in results]

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
ax = plt.gca()
x = np.arange(len(labels)); width = 0.25
ax.bar(x - width, r_std, width, label="Standard KK")
ax.bar(x,         r_ss,  width, label="SSKK (1 anchor)")
ax.bar(x + width, r_ms,  width, label="MSKK-2 (two anchors)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("RMS relative error of Re epsilon")
ax.set_title("Finite-Band Reconstruction: Standard vs SSKK vs MSKK-2 (two anchors)\n"
             "Havriliak-Negami synthetic; Dark mode", fontsize=28, loc="left")
ax.grid(True, axis="y", alpha=0.2)
ax.legend(fontsize=14, frameon=True)
plt.tight_layout()
outp = "data/artifacts/images/kk_mskk2_bars_fixed_3160x2820.png"
plt.savefig(outp, dpi=dpi, facecolor="#0d0f14")
print("Saved:", outp, "| Results:", results)
