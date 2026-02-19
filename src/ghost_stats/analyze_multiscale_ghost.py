import numpy as np
import scipy.signal as signal
import hdf5plugin
import h5py
import pandas as pd
import glob
import os

PHI = 1.618033988749895
GHOST_FREQ = 0.786151377757423
ALIASED_GHOST_FREQ = 1.0 - GHOST_FREQ # 0.2138

def ricker_wavelet(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss

def simple_cwt(data, widths):
    output = np.zeros((len(widths), len(data)))
    for i, width in enumerate(widths):
        # Length of wavelet kernel: usually 10*width
        points = int(min(10 * width, len(data)))
        # Use our own ricker implementation
        wavelet = ricker_wavelet(points, width)
        output[i, :] = signal.convolve(data, wavelet, mode='same')
    return output

def analyze_cwt(data, fs=1.0, title="Signal"):
    print(f"\n--- Multiscale CWT Analysis: {title} ---")
    n = len(data)
    
    # Detrend
    data = data - np.mean(data)
    
    # Scales: 1 to 50
    widths = np.arange(1, 51)
    
    # Use manual CWT
    cwt_mat = simple_cwt(data, widths)
    
    # Power scalogram
    power = np.abs(cwt_mat)**2
    
    # Global Scale Spectrum
    scale_spectrum = np.mean(power, axis=1)
    
    # Find peak width
    peak_idx = np.argmax(scale_spectrum)
    peak_width = widths[peak_idx]
    
    # Convert width to frequency
    # f = 2 / (sqrt(3) * pi * width)
    peak_freq = 2 / (np.sqrt(3) * np.pi * peak_width) * fs
    
    print(f"Top Scale Peak:")
    print(f"  Width: {peak_width}")
    print(f"  Approx Freq: {peak_freq:.4f} cycles/sample")
    print(f"  Ghost Freq Target: {ALIASED_GHOST_FREQ:.4f}")
    
    # Check Ghost
    if abs(peak_freq - ALIASED_GHOST_FREQ) < 0.05:
        print("  *** GHOST SCALE MATCH (CWT) ***")
        return True
        
    # Check Ghost Band Power
    ghost_band_power = np.mean(scale_spectrum[0:3]) # Widths 1,2,3
    total_power = np.mean(scale_spectrum)
    ratio = ghost_band_power / total_power
    print(f"  Ghost Band Power Ratio: {ratio:.2f}")
    if ratio > 0.5:
        print("  *** SIGNIFICANT GHOST BAND POWER ***")
        return True
        
    return False

def load_wow_sband():
    files = sorted(glob.glob("data/bl_6equj5_gbt/*.h5"))
    if not files: return None
    stacked_ts = None
    count = 0
    for f in files:
        try:
            with h5py.File(f, 'r') as h5:
                dset = h5['data']
                chunk = dset[:, 0, :1000] 
                ts = np.mean(chunk, axis=1)
                if stacked_ts is None: stacked_ts = np.zeros_like(ts)
                stacked_ts += ts
                count += 1
        except Exception: pass
    if stacked_ts is not None: stacked_ts /= count
    return stacked_ts

def load_alice():
    try:
        df = pd.read_csv("data/external/cms_oo_raa/alice_pbpb_raa_0to5pct_baseline.csv")
        return df['R_{AA}'].values
    except: return None

def load_swarm():
    try:
        df = pd.read_csv("data/external/swarm_f_header.csv")
        return df['F'].values[::10]
    except: return None

def main():
    print("=== Multidimensional 'Soft' Solver: CWT Analysis ===")
    wow_data = load_wow_sband()
    if wow_data is not None: analyze_cwt(wow_data, title="Wow! S-Band Stacked")
    
    alice_data = load_alice()
    if alice_data is not None: analyze_cwt(alice_data, title="ALICE Pb-Pb R_AA")
    
    swarm_data = load_swarm()
    if swarm_data is not None: analyze_cwt(swarm_data, title="Swarm Magnetic Field (F)")

if __name__ == "__main__":
    main()
