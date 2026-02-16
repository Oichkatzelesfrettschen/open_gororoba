import h5py
import numpy as np
import os
import sys

def analyze_field(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"=== Field Snapshot Analysis: {file_path} ===")
            
            if 'velocity' in f:
                u_flat = np.array(f['velocity'])
                print(f"  Raw data size: {len(u_flat)}")
                
                # Check for NaNs
                nan_count = np.isnan(u_flat).sum()
                if nan_count > 0:
                    print(f"  CRITICAL: Found {nan_count} NaNs in velocity field!")
                
                # Reshape if possible (assuming N^3 * 3)
                n_total = len(u_flat) // 3
                res = int(round(n_total**(1/3)))
                if res**3 == n_total:
                    u = u_flat.reshape((res, res, res, 3))
                    print(f"  Detected Grid: {res}^3")
                    
                    # Compute Magnitudes
                    mag = np.linalg.norm(u, axis=3)
                    print(f"  Max Velocity: {np.max(mag):.6e}")
                    print(f"  Mean Velocity: {np.mean(mag):.6e}")
                    
                    # Simple Energy Check
                    energy = 0.5 * np.sum(mag**2)
                    print(f"  Total Kinetic Energy: {energy:.6e}")
                else:
                    print(f"  Unknown grid layout. Total elements: {len(u_flat)}")

    except Exception as e:
        print(f"Failed to analyze field HDF5: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_field_h5.py <path_to_h5>")
    else:
        analyze_field(sys.argv[1])
