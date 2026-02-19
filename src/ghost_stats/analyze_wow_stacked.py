import hdf5plugin
from blimpy import Waterfall
import numpy as np
import scipy.signal as signal
import os
import glob

PHI = 1.618033988749895
GHOST_FREQ = 0.786151377757423
ALIASED_GHOST_FREQ = 1.0 - GHOST_FREQ # 0.2138

def analyze_stacked(directory):
    print(f"Deep Stacked Analysis of {directory}...")
    files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if not files:
        print("No H5 files found.")
        return

    stacked_power = None
    n_stacked = 0
    freqs = None

    for path in files:
        try:
            print(f"  Processing {os.path.basename(path)}...")
            wf = Waterfall(path)
            # data: (time, beams, channels) -> (279, 65536)
            data = wf.data[:, 0, :] 
            nt, nch = data.shape
            
            # Integrated time series
            time_series = np.mean(data, axis=1)
            
            # FFT
            ts_centered = time_series - np.mean(time_series)
            win = signal.windows.hann(len(ts_centered))
            
            # Use rfft
            current_freqs = np.fft.rfftfreq(len(ts_centered))
            power = np.abs(np.fft.rfft(ts_centered * win))**2
            
            # Normalize power to noise floor = 1.0
            nf = np.mean(power)
            power_norm = power / nf
            
            # Initialize stack if first file
            if stacked_power is None:
                stacked_power = np.zeros_like(power_norm)
                freqs = current_freqs
            
            # Check length consistency
            if len(power_norm) == len(stacked_power):
                stacked_power += power_norm
                n_stacked += 1
            else:
                print(f"    Skipping (length mismatch): {len(power_norm)} vs {len(stacked_power)}")

        except Exception as e:
            print(f"    Error: {e}")

    if n_stacked == 0:
        print("No files successfully processed.")
        return

    # Average the stack
    stacked_power /= n_stacked
    print(f"\n=== Stacked Analysis Results (N={n_stacked}) ===")
    
    # Peak finding on stacked spectrum
    # Threshold: 5 sigma (conservative)
    peaks, _ = signal.find_peaks(stacked_power, height=5.0)
    
    print(f"Top 10 peaks in STACKED signal:")
    peak_indices = np.argsort(stacked_power[peaks])[::-1][:10]
    
    found_ghost = False
    for idx in peak_indices:
        p = peaks[idx]
        snr = stacked_power[p]
        print(f"  Freq: {freqs[p]:.6f}, SNR: {snr:.2f}")
        
        # Check Ghost (Aliased)
        if abs(freqs[p] - ALIASED_GHOST_FREQ) < 0.02:
            print(f"  *** GHOST FREQUENCY MATCH (Aliased) ***")
            found_ghost = True
            
        # Check Ghost (Fundamental) - unlikely in 0..0.5 range if aliased
        # But check just in case
        if abs(freqs[p] - GHOST_FREQ) < 0.02:
             print(f"  *** GHOST FREQUENCY MATCH (Fundamental) ***")
             found_ghost = True

    if not found_ghost:
        # Check specific SNR at ghost freq
        ghost_bin = np.argmin(np.abs(freqs - ALIASED_GHOST_FREQ))
        ghost_snr = stacked_power[ghost_bin]
        print(f"\nTarget Check at {ALIASED_GHOST_FREQ:.4f}: SNR = {ghost_snr:.2f}")
        if ghost_snr > 3.0:
            print("  -> Weak sub-threshold excess detected.")
        else:
            print("  -> No excess detected (Null result).")

if __name__ == "__main__":
    analyze_stacked("data/bl_6equj5_gbt")
