import pandas as pd
import ast

def extract_cores(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter for significance
    sig_df = df[df['significant'] == True].copy()
    
    # Parse subsets
    # Subset format in CSV is string like "log_DM+gl" or "ra+dec+pmra+pmdec"
    # We need to split by '+'
    
    results = {}
    
    for dataset in sig_df['dataset'].unique():
        ds_sig = sig_df[sig_df['dataset'] == dataset]
        
        # Get all significant subsets as sets of attributes
        # metric often duplicates rows (um_fraction vs tolerance_auc)
        # We just need the unique subsets that are significant in AT LEAST ONE metric
        subsets = []
        for s_str in ds_sig['subset'].unique():
            parts = set(s_str.split('+'))
            subsets.append(parts)
            
        # Find cores (minimal subsets)
        cores = []
        for s in subsets:
            is_core = True
            for other in subsets:
                if s == other: continue
                if other.issubset(s):
                    is_core = False
                    break
            if is_core:
                cores.append(s)
                
        # Format for display
        cores_formatted = [sorted(list(c)) for c in cores]
        results[dataset] = cores_formatted
        
    return results

if __name__ == "__main__":
    path = "data/csv/c071g_exploration_gpu_10M_1000perm.csv"
    try:
        cores_by_ds = extract_cores(path)
        print("Ultrametric Cores by Dataset:")
        for ds, cores in cores_by_ds.items():
            print(f"\n{ds}:")
            if not cores:
                print("  (None)")
            for c in cores:
                print(f"  - {' + '.join(c)}")
                
    except Exception as e:
        print(f"Error: {e}")
