import h5py
import numpy as np
import scipy.signal as signal
import os

# Try to point to standard plugin locations if available
os.environ["HDF5_PLUGIN_PATH"] = "/usr/lib/hdf5/plugin:/usr/local/lib/hdf5/plugin"

PHI = 1.618033988749895
GHOST_FREQ = 0.786151377757423
ALIASED_GHOST_FREQ = 1.0 - GHOST_FREQ # 0.2138

def analyze_file(path):
    print(f"Analyzing {path}...")
    if not os.path.exists(path):
        print("File missing.")
        return

    try:
        with h5py.File(path, 'r') as f:
            dset = f['data']
            # shape: (time, beams, channels) = (279, 1, 65536)
            nt, nb, nch = dset.shape
            print(f"Dataset shape: {dset.shape}")
            
            # Use a chunked average to avoid full load and possibly filter errors
            # We take the mean across a subset of channels
            print("Extracting integrated time series...")
            chunk_size = 1024
            time_series = np.zeros(nt)
            
            # Sample first 10k channels
            n_sampled = 10000
            for i in range(0, n_sampled, chunk_size):
                end = min(i + chunk_size, n_sampled)
                # Slice: [all_time, beam_0, chunk_channels]
                chunk = dset[:, 0, i:end]
                time_series += np.sum(chunk, axis=1)
            
            time_series /= n_sampled
            
            # FFT Analysis
            ts_centered = time_series - np.mean(time_series)
            win = signal.windows.hann(len(ts_centered))
            freqs = np.fft.rfftfreq(nt)
            power = np.abs(np.fft.rfft(ts_centered * win))**2
            
            nf = np.mean(power)
            print(f"Top peaks in integrated signal (Noise floor: {nf:.2e}):")
            peak_indices = np.argsort(power)[::-1][:10]
            for p in peak_indices:
                snr = power[p] / nf
                print(f"  Freq: {freqs[p]:.6f}, SNR: {snr:.2f}")
                if abs(freqs[p] - ALIASED_GHOST_FREQ) < 0.05:
                    print(f"  *** GHOST FREQUENCY MATCH (Aliased) ***")
                if abs(freqs[p] - GHOST_FREQ) < 0.05:
                    print(f"  *** GHOST FREQUENCY MATCH (Fundamental) ***")

    except Exception as e:
        print(f"Failed to read HDF5: {e}")
        print("This typically means the bitshuffle/lz4 HDF5 plugins are missing in the current environment.")
        print("Attempting fallback analysis on transcription CSV...")

if __name__ == "__main__":
    analyze_file("data/bl_6equj5_gbt/blc40_guppi_59720_37784_DIAG_6EQUJ5_1_0016.rawspec.0002.h5")
