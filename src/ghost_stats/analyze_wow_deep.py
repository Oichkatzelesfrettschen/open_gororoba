import hdf5plugin
from blimpy import Waterfall
import numpy as np
import scipy.signal as signal
import os

PHI = 1.618033988749895
GHOST_FREQ = 0.786151377757423
ALIASED_GHOST_FREQ = 1.0 - GHOST_FREQ # 0.2138

def analyze_file(path):
    print(f"Deep Analysis of {path} using blimpy...")
    if not os.path.exists(path):
        print("File missing.")
        return

    try:
        # blimpy handles headers and some compression details automatically
        wf = Waterfall(path)
        print(f"Metadata: fch1={wf.header['fch1']}, foff={wf.header['foff']}, nchans={wf.header['nchans']}")
        
        # Load data: (time, beams, channels)
        # We integrate over chunks to manage memory if needed, but let's try a full load of first 10k chans
        # wf.data shape is (n_time, n_ifs, n_chans)
        data = wf.data[:, 0, :] 
        nt, nch = data.shape
        print(f"Loaded data shape: {data.shape}")
        
        # 1. Broad Spectral Integration (Across all channels)
        # This collapses the frequency axis to see if there's a common temporal oscillation
        # across the entire band (the 'Ghost' signature)
        print("Computing integrated time series...")
        time_series = np.mean(data, axis=1)
        
        # FFT Analysis
        ts_centered = time_series - np.mean(time_series)
        win = signal.windows.hann(len(ts_centered))
        freqs = np.fft.rfftfreq(nt)
        power = np.abs(np.fft.rfft(ts_centered * win))**2
        
        nf = np.mean(power)
        print(f"Integrated Signal SNR analysis (Noise floor: {nf:.2e}):")
        peak_indices = np.argsort(power)[::-1][:10]
        for p in peak_indices:
            snr = power[p] / nf
            print(f"  Freq: {freqs[p]:.6f}, SNR: {snr:.2f}")
            if abs(freqs[p] - ALIASED_GHOST_FREQ) < 0.05:
                print(f"  *** GHOST FREQUENCY MATCH (Aliased) ***")
            if abs(freqs[p] - GHOST_FREQ) < 0.05:
                print(f"  *** GHOST FREQUENCY MATCH (Fundamental) ***")

        # 2. Search for the strongest narrowband signal in the band
        # (Traditional SETI search might have missed it if it's drifting weirdly)
        print("\nSearching for strongest narrowband carrier...")
        max_power_per_chan = np.max(data, axis=0)
        best_chan = np.argmax(max_power_per_chan)
        best_freq = wf.header['fch1'] + best_chan * wf.header['foff']
        print(f"Strongest channel: {best_chan} at {best_freq:.6f} MHz")
        
        # 3. Correlation with Sedenion structure (Advanced)
        # If we had the Sedenion weights, we could check if the intensity
        # distribution matches the associator tensor density.
        
    except Exception as e:
        print(f"Error during deep analysis: {e}")

if __name__ == "__main__":
    # We analyze the first file as a representative sample
    analyze_file("data/bl_6equj5_gbt/blc40_guppi_59720_37784_DIAG_6EQUJ5_1_0016.rawspec.0002.h5")
