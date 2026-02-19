import pandas as pd
import numpy as np
import scipy.signal as signal
import os

PHI = 1.618033988749895
GHOST_FREQ = 0.786151377757423
ALIASED_GHOST_FREQ = 1.0 - GHOST_FREQ # 0.2138

def analyze_nanograv_correlation():
    print("=== NANOGrav 15yr Multidimensional Phase Analysis ===")
    path = "data/external/nanograv_15yr_freespectrum.csv"
    if not os.path.exists(path):
        print("NANOGrav data missing.")
        return

    df = pd.read_csv(path)
    # The 'frequency' column in NANOGrav is in Hz (nHz range)
    # Filter valid rho
    valid_mask = ~df['log10_rho'].isna()
    rho = df['log10_rho'][valid_mask].values
    freqs = df['frequency'][valid_mask].values
    
    if len(rho) < 10:
        print(f"Insufficient valid NANOGrav samples: {len(rho)}")
        return

    # Analyze the 'shape' of the stochastic background
    rho_detrended = signal.detrend(rho)
    
    # FFT of the spectral shape
    n = len(rho_detrended)
    win = signal.windows.hann(n)
    spec_fft = np.abs(np.fft.rfft(rho_detrended * win))**2
    fft_freqs = np.fft.rfftfreq(n)
    
    mean_power = np.mean(spec_fft)
    eps = 1e-12
    
    print(f"Top peaks in NANOGrav spectral density modulation (N={n}):")
    peaks = np.argsort(spec_fft)[::-1][:10]
    for p in peaks:
        snr = spec_fft[p] / (mean_power + eps)
        print(f"  Scale-Freq: {fft_freqs[p]:.4f}, SNR: {snr:.2f}")
        if abs(fft_freqs[p] - ALIASED_GHOST_FREQ) < 0.05:
            print("  *** GHOST MODULATION DETECTED in GWB SHAPE ***")

def analyze_jpl_ephemeris():
    print("\n=== JPL Planets Multidimensional Range Analysis ===")
    path = "data/external/jpl_planets_2020_2030.csv"
    if not os.path.exists(path):
        print("JPL data missing.")
        return

    df = pd.read_csv(path)
    # Convert 'delta' to numeric, drop NaNs
    delta = pd.to_numeric(df['delta'], errors='coerce').dropna().values
    
    if len(delta) < 10:
        print(f"Insufficient valid JPL samples: {len(delta)}")
        return

    delta_dt = np.diff(delta) 
    
    # FFT
    ts = delta_dt - np.mean(delta_dt)
    n = len(ts)
    win = signal.windows.hann(n)
    power = np.abs(np.fft.rfft(ts * win))**2
    freqs = np.fft.rfftfreq(n)
    
    mean_p = np.mean(power)
    eps = 1e-12
    
    print(f"Top peaks in Planetary Range modulation (N={n}):")
    peaks = np.argsort(power)[::-1][:10]
    for p in peaks:
        snr = power[p] / (mean_p + eps)
        print(f"  Freq: {freqs[p]:.4f}, SNR: {snr:.2f}")
        if abs(freqs[p] - ALIASED_GHOST_FREQ) < 0.05:
            print("  *** GHOST CHIRP DETECTED IN PLANETARY MOTION ***")

if __name__ == "__main__":
    analyze_nanograv_correlation()
    analyze_jpl_ephemeris()
