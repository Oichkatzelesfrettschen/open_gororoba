
import numpy as np
import matplotlib.pyplot as plt
import os

def vp_int(n, p):
    """Compute p-adic valuation of integer n."""
    if n == 0:
        return 0
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k

def kolmogorov_spectrum(k, k_min=1.0, epsilon=1.0):
    """Standard Kolmogorov spectrum E(k) ~ epsilon^(2/3) * k^(-5/3)."""
    return epsilon**(2/3) * k**(-5/3) * (k >= k_min)

def simulate_warp_filtering():
    """
    Simulate the effect of p-adic and negative-dimension filters 
    on a synthetic Kolmogorov turbulent spectrum.
    """
    k_min = 1
    k_max = 1024
    prime = 2
    alpha = -0.5  # Negative dimension exponent
    reg_epsilon = 0.01

    k_values = np.arange(k_min, k_max + 1)
    
    # 1. Baseline Kolmogorov Spectrum
    E_base = kolmogorov_spectrum(k_values)
    
    # 2. P-adic Filter
    # w_p = p^{-v_p(k)}
    padic_weights = np.array([prime**(-vp_int(int(k), prime)) for k in k_values])
    E_padic = E_base * padic_weights
    
    # 3. Negative-Dimension Filter
    # w_nd = (k + epsilon)^alpha
    neg_dim_weights = (k_values + reg_epsilon)**alpha
    E_neg_dim = E_base * neg_dim_weights
    
    # 4. Combined Warp Filter
    E_warp = E_base * padic_weights * neg_dim_weights

    # --- Analysis ---
    
    # Sparsity: Fraction of energy in top 10% of modes
    def sparsity(E):
        sorted_E = np.sort(E)[::-1]
        top_10_count = int(len(E) * 0.1)
        return np.sum(sorted_E[:top_10_count]) / np.sum(E)

    print(f"Sparsity (Baseline): {sparsity(E_base):.4f}")
    print(f"Sparsity (P-adic):   {sparsity(E_padic):.4f}")
    print(f"Sparsity (Neg-Dim):  {sparsity(E_neg_dim):.4f}")
    print(f"Sparsity (Warp):     {sparsity(E_warp):.4f}")

    # Slope Analysis (Log-Log fit)
    def fit_slope(k, E):
        log_k = np.log10(k)
        log_E = np.log10(E)
        slope, _ = np.polyfit(log_k, log_E, 1)
        return slope

    slope_base = fit_slope(k_values, E_base)
    slope_warp = fit_slope(k_values, E_warp)
    
    print(f"Spectral Slope (Baseline): {slope_base:.4f} (Expected ~ -1.67)")
    print(f"Spectral Slope (Warp):     {slope_warp:.4f}")

    # Save data for plotting/verification
    output_dir = "data/csv"
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.column_stack((k_values, E_base, E_padic, E_neg_dim, E_warp))
    header = "k,E_base,E_padic,E_neg_dim,E_warp"
    np.savetxt(f"{output_dir}/warp_physics_simulation.csv", data, delimiter=",", header=header, comments="")
    print(f"Data saved to {output_dir}/warp_physics_simulation.csv")

if __name__ == "__main__":
    simulate_warp_filtering()
