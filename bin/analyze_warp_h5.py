import h5py
import numpy as np
import os
import sys

def analyze_h5(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"=== Analysis of {file_path} ===")
            
            # 1. Inspect Groups and Datasets
            print("\nStructure:")
            def print_attrs(name, obj):
                print(f"  {name} ({type(obj).__name__})")
                for key, val in obj.attrs.items():
                    print(f"    Attr: {key} = {val}")
            f.visititems(print_attrs)

            # 2. Analyze Trace Data (Time Series)
            # Based on structure: simulation/trace/...
            if 'simulation' in f and 'trace' in f['simulation']:
                trace = f['simulation']['trace']
                time = np.array(trace['time'])
                density = np.array(trace['rho_mean'])
                enstrophy = np.array(trace['enstrophy'])
                
                print("\nTime-Series Summary:")
                print(f"  Steps recorded: {len(time)}")
                if len(density) > 0:
                    print(f"  Initial Mean Density: {density[0]:.6f}")
                    print(f"  Final Mean Density: {density[-1]:.6f}")
                    print(f"  Density StdDev: {np.std(density):.2e}")
                if len(enstrophy) > 0:
                    print(f"  Final Enstrophy: {enstrophy[-1]:.6e}")
                
                # Check for stability issues
                if np.any(np.isnan(density)) or np.any(np.isinf(density)):
                    print("  CRITICAL: Instability (NaN/Inf) detected in density trace!")
                elif len(density) > 0 and np.abs(density[-1] - 1.0) > 0.1:
                    print(f"  WARNING: High mass drift: {density[-1] - 1.0:+.4f}")

            # 3. Experiment Results Check
            if 'experiment/results' in f:
                res = f['experiment/results']
                md = res.attrs.get('mean_density', 0.0)
                if np.isnan(md):
                    print("\n  Forensic: 'mean_density' in results attribute is NaN. Investigating trace...")

    except Exception as e:
        print(f"Failed to analyze HDF5: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_warp_h5.py <path_to_h5>")
    else:
        analyze_h5(sys.argv[1])
