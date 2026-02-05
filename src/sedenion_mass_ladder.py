import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_mass_ladder(alpha=-1.5):
    """
    Calculates the 'Sedenion Mass Ladder' based on the negative dimension spectral hypothesis.
    M_n = m_0 * n^(-alpha)

    We calibrate m_0 such that n=10 matches the first LIGO peak (~35 M_sun).
    35 = m_0 * 10^(1.5)
    m_0 = 35 / 31.62 = 1.107 M_sun.
    So roughly, the fundamental mass unit is ~1.1 M_sun (close to Chandrasekhar limit?).
    """
    print(f"--- Calculating Mass Ladder for alpha={alpha} ---")

    # Calibration
    n_cal = 10
    M_cal = 35.0 # Solar masses
    m_0 = M_cal / (n_cal**(-alpha))
    print(f"Calibrated Fundamental Mass m_0: {m_0:.3f} M_sun")

    modes = np.arange(1, 51)
    masses = m_0 * modes**(-alpha)

    # Analyze Specific Modes
    # PISN Gap starts around 50-65 M_sun. Ends around 120-130 M_sun.

    df = pd.DataFrame({'Mode_n': modes, 'Predicted_Mass': masses})
    print(df.head(30))

    # Check for Gaps
    # In this model, the "Gap" isn't a lack of modes, but maybe a lack of STABILITY for certain n.
    # Sedenion Zero Divisors occur at specific indices.
    # If the "Mode n" corresponds to a non-associative triplet, maybe it's unstable?
    # For now, we just map the spectrum.

    # Plotting
    plt.figure(figsize=(12, 8))

    # The Spectrum
    plt.plot(modes, masses, 'o-', markersize=4, color='crimson', label='Sedenion Modes')

    # Known Astrophysical Regions
    plt.axhspan(2.5, 5, color='gray', alpha=0.2, label='Neutron Star / BH Gap (Historical)')
    plt.axhspan(50, 120, color='orange', alpha=0.2, label='Pair-Instability Supernova Gap')

    # Annotate key modes
    plt.annotate(f'n=10: {masses[9]:.1f} (LIGO Peak 1)', (10, masses[9]), xytext=(12, 30), arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'n=15: {masses[14]:.1f} (LIGO Peak 2)', (15, masses[14]), xytext=(17, 55), arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'n=25: {masses[24]:.1f} (Gap Upper Edge)', (25, masses[24]), xytext=(27, 120), arrowprops=dict(arrowstyle='->'))

    plt.title(f"Sedenion Mass Ladder vs Astrophysical Gaps ($m_0={m_0:.2f} M_{{\\odot}}$)")
    plt.xlabel("Spectral Mode (n)")
    plt.ylabel(r"Mass ($M_{\odot}$)")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.yscale('log')

    plt.savefig("data/artifacts/images/sedenion_mass_ladder_log.png")
    df.to_csv("data/csv/sedenion_mass_spectrum.csv", index=False)
    print("Mass Ladder Generation Complete.")

if __name__ == "__main__":
    calculate_mass_ladder(alpha=-1.5)
