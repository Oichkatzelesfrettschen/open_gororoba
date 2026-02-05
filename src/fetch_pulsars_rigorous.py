import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier


def fetch_real_pulsar_masses():
    print("--- Acquiring Rigorous Pulsar Mass Data via Vizier ---")

    # Catalog: J/ApJ/832/167 - "Millisecond pulsars masses & radii" (Antoniadis+, 2016)
    # Or J/ApJ/772/96 - "Neutron star mass distribution" (Kiziltan+, 2013)

    Vizier.ROW_LIMIT = -1 # Unlimited rows

    # Target catalogs with mass data
    catalogs_to_try = ["J/ApJ/772/96/table1", "J/ApJ/832/167/table1"]

    dfs = []

    for cat in catalogs_to_try:
        try:
            print(f"Querying {cat}...")
            # We want specific columns. 'M' or 'Mass'
            tables = Vizier.get_catalogs(cat)
            if len(tables) > 0:
                df = tables[0].to_pandas()
                print(f"Columns: {df.columns}")

                # Normalize column names
                if 'M' in df.columns:
                    masses = df['M']
                elif 'Mass' in df.columns:
                    masses = df['Mass']
                elif 'Mpsr' in df.columns:
                    masses = df['Mpsr']
                else:
                    print(f"No explicit mass column in {cat}. Checking next.")
                    continue

                # Filter valid masses (e.g., > 0)
                masses = masses.dropna()
                masses = masses[masses > 0]
                dfs.append(masses)
                print(f"Extracted {len(masses)} masses from {cat}.")
        except Exception as e:
            print(f"Failed to query {cat}: {e}")

    if dfs:
        all_masses = pd.concat(dfs)
        output_path = "data/external/Real_Pulsar_Masses.csv"
        pd.DataFrame({'Mass': all_masses}).to_csv(output_path, index=False)
        print(f"Success. Combined {len(all_masses)} unique mass measurements.")
    else:
        print("Could not retrieve real mass data. Checking ATNF catalog parameters via search...")
        # Fallback to searching for a generic pulsar table
        try:
            v = Vizier(columns=['*'], keywords=['pulsar', 'mass'])
            tables = v.query_object("PSR J0437-4715") # Try to find a catalog linking to a known pulsar
            print("Search result catalogs:", [t.name for t in tables])
        except:
            print("Search failed.")

if __name__ == "__main__":
    fetch_real_pulsar_masses()
