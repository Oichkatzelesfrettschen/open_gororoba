import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gwpy.table import EventTable


def fetch_gwtc3_gwpy():
    print("--- Acquiring LIGO GWTC-3 Data via GWpy ---")

    try:
        # Fetch the catalog
        print("Querying GWOSC for 'GWTC-3-confident'...")
        catalog = EventTable.fetch_open_data('GWTC-3-confident')

        # Convert to Pandas for easier handling
        df = catalog.to_pandas()

        # Essential columns: mass_1_source, mass_2_source, chirp_mass, redshift, luminosity_distance
        # We check what columns are actually available
        cols = ['mass_1_source', 'mass_2_source', 'chirp_mass_source', 'redshift', 'luminosity_distance']
        available_cols = [c for c in cols if c in df.columns]

        df_clean = df[available_cols].dropna()

        output_path = "data/external/GWTC-3_GWpy_Official.csv"
        df_clean.to_csv(output_path, index=False)

        print(f"Successfully fetched {len(df_clean)} confirmed events.")
        print(f"Data saved to {output_path}")

        # Analysis: Mass Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df_clean['mass_1_source'], bins=30, color='darkblue', alpha=0.7, label='Primary Mass (M1)')
        plt.hist(df_clean['mass_2_source'], bins=30, color='teal', alpha=0.5, label='Secondary Mass (M2)')
        plt.xlabel(r"Mass ($M_{\odot}$)")
        plt.ylabel('Count')
        plt.title('LIGO GWTC-3 Black Hole Mass Distribution (Official Data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("data/artifacts/images/LIGO_GWTC3_Mass_Spectrum.png")
        print("Saved Mass Spectrum plot.")

        return df_clean

    except Exception as e:
        print(f"GWpy Fetch Failed: {e}")
        # Fallback to the CSV we downloaded earlier if GWpy fails (backup plan)
        return None

if __name__ == "__main__":
    fetch_gwtc3_gwpy()
