import pandas as pd
import numpy as np

df = pd.read_csv("data/output/heliosphere/feature_cube.csv")
print(df.groupby(["mission", "product"])[["density_cm3", "speed_kms", "temperature_k", "b_mag", "crs_flux", "spectral_peak"]].count())
print("\n--- Mean values (checking for 0.0 or NaNs) ---")
print(df.groupby(["mission", "product"])[["density_cm3", "speed_kms", "temperature_k", "b_mag", "crs_flux", "spectral_peak"]].mean())
