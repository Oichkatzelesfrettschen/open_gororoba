import warnings

warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd
from astroquery.mast import Observations
from astroquery.sdss import SDSS


def fetch_observatory_data():
    print("--- Scientific Data Acquisition Pipeline ---")

    # 1. Hubble Space Telescope (MAST)
    print("Querying MAST for Hubble Deep Field data...")
    try:
        # Query for a famous field (e.g., Hubble Ultra Deep Field)
        # Coordinates: RA 53.4427, Dec -27.7914
        obs_table = Observations.query_criteria(
            coordinates="53.4427 -27.7914",
            radius="0.1 deg",
            project="HST",
            instrument_name="ACS/WFC",
            filters=["F775W"]
        )
        print(f"Found {len(obs_table)} Hubble observations.")

        # Save Metadata
        df_hst = obs_table.to_pandas()
        df_hst.to_csv("data/external/HST_UDF_Metadata.csv", index=False)
        print("Saved HST Metadata.")

    except Exception as e:
        print(f"MAST Query Failed: {e}")

    # 2. Kepler/TESS Exoplanets
    print("Querying MAST for TESS Exoplanet Candidates...")
    try:
        # Query for TESS Sector 1 data around a specific star or random cone
        obs_table_tess = Observations.query_criteria(
            project="TESS",
            radius="0.5 deg",
            coordinates="300.0 45.0"
        )
        print(f"Found {len(obs_table_tess)} TESS observations.")
        df_tess = obs_table_tess.to_pandas()
        df_tess.to_csv("data/external/TESS_Metadata.csv", index=False)
    except Exception as e:
        print(f"TESS Query Failed: {e}")

    # 3. SDSS Spectroscopy (Quasars) - Replacing the manual curl
    print("Querying SDSS for Quasars (Limit 500)...")
    try:
        query = "SELECT TOP 500 z, ra, dec, bestObjID FROM SpecObj WHERE class='QSO' AND z > 0.1"
        res = SDSS.query_sql(query)
        if res:
            df_sdss = res.to_pandas()
            df_sdss.to_csv("data/external/SDSS_Quasars_Astroquery.csv", index=False)
            print(f"Saved {len(df_sdss)} SDSS Quasars.")
        else:
            print("No SDSS results returned.")

    except Exception as e:
        print(f"SDSS Query Failed: {e}")

if __name__ == "__main__":
    fetch_observatory_data()
