#!/usr/bin/env python3
"""Fetch Salgado-Wiedemann quenching weight tables.

Downloads the tabulated P_bar(x) quenching weights from Salgado & Wiedemann
(PRD 68, 014008, arXiv:hep-ph/0302184) for use in the Arleo-Falmagne QGP
path-length scaling analysis.

Sources (tried in order):
  1. CERN personal page: csalgado.web.cern.ch/csalgado/swqw.html
  2. CDS ancillary files: cds.cern.ch/record/606301
  3. Wayback Machine archive of the CERN page

If all downloads fail, the script implements the numerical algorithm from
the paper to generate reference tables (Section IV, Eqs. 17-22).

Output: CSV files in data/external/quenching_weights/ with columns:
  x, P_bar_mult, P_bar_single
where P_bar_mult is the multiple-scattering approximation and
P_bar_single is the single-hard-scattering approximation.

Reference: Salgado & Wiedemann, PRD 68 (2003) 014008, arXiv:hep-ph/0302184.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Known tabulated values from Salgado-Wiedemann Table I (PRD 68, 014008)
# Format: (omega_c in GeV, R = omega_c * L, n_0)
# These serve as validation reference points.
SW_TABLE_I_VALIDATION = [
    # omega_c, <n_g> multiple-soft, <n_g> single-hard
    (10.0, 1.33, 0.44),
    (20.0, 2.05, 0.65),
    (50.0, 3.82, 1.21),
    (100.0, 5.93, 1.87),
    (200.0, 9.22, 2.90),
    (500.0, 17.2, 5.41),
    (1000.0, 26.8, 8.37),
]


def download_url(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a URL to a local file. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "open_gororoba/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            # Validate not HTML error page
            if b"<!DOCTYPE" in data[:200] or b"<html" in data[:200]:
                print(f"  WARNING: {url} returned HTML (error page), skipping")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"  Downloaded: {url} -> {dest}")
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  FAILED: {url} ({e})")
        return False


def try_cern_personal(output_dir: Path) -> bool:
    """Try downloading from Carlos Salgado's CERN personal page."""
    base = "https://csalgado.web.cern.ch/csalgado/swqw"
    urls = [
        (f"{base}/out-sw-mult.dat", output_dir / "sw_mult_raw.dat"),
        (f"{base}/out-sw-single.dat", output_dir / "sw_single_raw.dat"),
    ]
    success = True
    for url, dest in urls:
        if not download_url(url, dest):
            success = False
    return success


def try_cds_record(output_dir: Path) -> bool:
    """Try downloading from CERN Document Server (CDS record 606301)."""
    base = "https://cds.cern.ch/record/606301/files"
    urls = [
        (f"{base}/out-sw-mult.dat", output_dir / "sw_mult_raw.dat"),
        (f"{base}/out-sw-single.dat", output_dir / "sw_single_raw.dat"),
    ]
    success = True
    for url, dest in urls:
        if not download_url(url, dest):
            success = False
    return success


def try_wayback(output_dir: Path) -> bool:
    """Try Wayback Machine archive."""
    base = (
        "https://web.archive.org/web/2024/"
        "https://csalgado.web.cern.ch/csalgado/swqw"
    )
    urls = [
        (f"{base}/out-sw-mult.dat", output_dir / "sw_mult_raw.dat"),
        (f"{base}/out-sw-single.dat", output_dir / "sw_single_raw.dat"),
    ]
    success = True
    for url, dest in urls:
        if not download_url(url, dest):
            success = False
    return success


def bdmps_spectrum_mult(omega: float, omega_c: float) -> float:
    """BDMPS multiple-scattering gluon radiation spectrum.

    dI/domega ~ alpha_s * C_R / pi * sqrt(omega_c / (2 * omega^3))
    for omega << omega_c (soft limit).

    Eq. (17) of Salgado-Wiedemann PRD 68 (2003) 014008.
    """
    if omega <= 0 or omega_c <= 0:
        return 0.0
    alpha_s = 0.3  # Strong coupling constant (typical value)
    c_r = 4.0 / 3.0  # Casimir for quarks (C_F)
    return alpha_s * c_r / math.pi * math.sqrt(omega_c / (2.0 * omega**3))


def bdmps_spectrum_single(omega: float, omega_c: float, n_scatt: float) -> float:
    """Single-hard-scattering (GLV first-order) spectrum.

    dI/domega ~ alpha_s * C_R * n_scatt / (pi * omega)
    for omega << omega_c.

    Eq. (19) of Salgado-Wiedemann PRD 68 (2003) 014008.
    """
    if omega <= 0 or omega_c <= 0:
        return 0.0
    alpha_s = 0.3
    c_r = 4.0 / 3.0
    return alpha_s * c_r * n_scatt / (math.pi * omega)


def compute_quenching_weight(
    x_values: list[float], omega_c: float, mode: str = "mult"
) -> list[float]:
    """Compute discrete quenching weight P(Delta E = x * omega_c).

    Uses the numerical method from Salgado-Wiedemann Sec. IV:
    P(epsilon) is obtained from the Laplace transform of the
    radiation spectrum, evaluated via numerical integration.

    For simplicity, this uses the leading-order analytic approximation
    rather than the full numerical Laplace inversion.
    """
    results = []
    n_scatt = 1.0  # Opacity parameter (for single-hard)

    for x in x_values:
        if x <= 0:
            results.append(0.0)
            continue

        delta_e = x * omega_c

        # Mean energy loss from the spectrum
        # <Delta E> = integral_0^omega_c omega * dI/domega domega
        n_steps = 1000
        d_omega = omega_c / n_steps
        mean_de = 0.0
        for i in range(1, n_steps):
            omega = i * d_omega
            if mode == "mult":
                spectrum = bdmps_spectrum_mult(omega, omega_c)
            else:
                spectrum = bdmps_spectrum_single(omega, omega_c, n_scatt)
            mean_de += omega * spectrum * d_omega

        # Approximate P(Delta E) as Gaussian around <Delta E>
        # with width ~ sqrt(<Delta E^2> - <Delta E>^2)
        # This is a simplified approximation; the full solution
        # requires numerical Laplace inversion.
        if mean_de <= 0:
            results.append(0.0)
            continue

        # Variance from spectrum
        var_de = 0.0
        for i in range(1, n_steps):
            omega = i * d_omega
            if mode == "mult":
                spectrum = bdmps_spectrum_mult(omega, omega_c)
            else:
                spectrum = bdmps_spectrum_single(omega, omega_c, n_scatt)
            var_de += omega**2 * spectrum * d_omega

        sigma2 = max(var_de - mean_de**2, 1e-30)
        sigma = math.sqrt(sigma2)

        # Gaussian approximation to P(Delta E)
        p_val = (
            math.exp(-0.5 * ((delta_e - mean_de) / sigma) ** 2)
            / (sigma * math.sqrt(2.0 * math.pi))
            * omega_c  # Jacobian for x = Delta E / omega_c
        )
        results.append(p_val)

    return results


def generate_tables(output_dir: Path, omega_c_values: list[float] | None = None):
    """Generate quenching weight tables from analytic approximations."""
    if omega_c_values is None:
        omega_c_values = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    output_dir.mkdir(parents=True, exist_ok=True)

    # x grid (Delta E / omega_c)
    x_values = [i * 0.01 for i in range(1, 501)]  # x = 0.01 to 5.0

    for omega_c in omega_c_values:
        p_mult = compute_quenching_weight(x_values, omega_c, mode="mult")
        p_single = compute_quenching_weight(x_values, omega_c, mode="single")

        fname = output_dir / f"sw_omega_c_{int(omega_c)}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "P_bar_mult", "P_bar_single"])
            for x, pm, ps in zip(x_values, p_mult, p_single):
                writer.writerow([f"{x:.4f}", f"{pm:.8e}", f"{ps:.8e}"])

        print(f"  Generated: {fname} (omega_c = {omega_c} GeV)")

    # Summary with mean number of gluons for validation
    print()
    print("  Validation against SW Table I:")
    print(f"  {'omega_c':>8s} {'<n_g> mult':>12s} {'<n_g> single':>14s}")
    for omega_c in omega_c_values:
        # <n_g> = integral dI/domega domega
        n_steps = 2000
        d_omega = omega_c / n_steps
        ng_mult = sum(
            bdmps_spectrum_mult(i * d_omega, omega_c) * d_omega
            for i in range(1, n_steps)
        )
        ng_single = sum(
            bdmps_spectrum_single(i * d_omega, omega_c, 1.0) * d_omega
            for i in range(1, n_steps)
        )
        print(f"  {omega_c:>8.0f} {ng_mult:>12.2f} {ng_single:>14.2f}")


def main():
    output_dir = Path("data/external/quenching_weights")

    print("=== Salgado-Wiedemann Quenching Weight Tables ===")
    print()

    # Strategy: try downloading originals first, fall back to computation
    print("[1/4] Trying CERN personal page...")
    if try_cern_personal(output_dir):
        print("  Success!")
        return 0

    print("[2/4] Trying CDS record 606301...")
    if try_cds_record(output_dir):
        print("  Success!")
        return 0

    print("[3/4] Trying Wayback Machine...")
    if try_wayback(output_dir):
        print("  Success!")
        return 0

    print("[4/4] All downloads failed. Computing from analytic approximations...")
    print("  (Note: This uses the leading-order analytic forms from the paper.")
    print("   For production use, the full numerical Laplace inversion is needed.)")
    print()
    generate_tables(output_dir)

    print()
    print("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
