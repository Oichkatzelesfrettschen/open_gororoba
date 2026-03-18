"""
MaNGA Zero-Dark Density Null Result: Decisive Experiment Harness

=== Dataset ===
Synthetic MaNGA-like rotation curves (N=6992 galaxies) generated from:
- NFW dark matter profiles with Duffy+2008 concentration-mass relation
- Moster+2013 stellar-mass-to-halo-mass relation
- Calibrated baryonic systematics (+5% bulge excess, -12% cusp, +5% IFU edge)
- Galaxy-to-galaxy scatter: 8% RMS, measurement noise: 3-10% of v_circ

=== Distribution Shift ===
Three regimes: full_sample, face_on (i<45), mass_Q3 (3rd quartile)

=== Conditions ===
20 conditions: 4 baselines + 4 proposed methods + 12 ablations.
Priority ordering ensures baselines and primary methods complete first.

=== Evaluation ===
20 conditions x 3 regimes x 20 seeds = 1200 runs (time-budget-guarded).
Primary metric: detection_snr (lower = stronger null).
"""

import json
import math
import sys
import time
import traceback

import numpy as np
from config import (
    CONDITION_PRIORITY,
    HYPERPARAMETERS,
    REGIMES,
    SEEDS,
)
from data_utils import (
    filter_inclination,
    filter_mass_quartile,
    generate_manga_sample,
)
from models import get_all_conditions


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)


def apply_regime_filter(galaxies: list, regime_name: str, regime_cfg: dict) -> list:
    """Apply regime-specific galaxy filter."""
    filt = regime_cfg.get("filter")
    if filt is None:
        return galaxies
    elif filt == "inclination":
        return filter_inclination(
            galaxies, regime_cfg["inc_min"], regime_cfg["inc_max"])
    elif filt == "mass_quartile":
        return filter_mass_quartile(galaxies, regime_cfg["quartile"])
    return galaxies


def main():
    time_budget = HYPERPARAMETERS["time_budget_seconds"]
    t_start = time.time()

    # -- Metric definitions --
    print("METRIC_DEF: primary_metric | direction=lower | "
          "desc=alpha_zd upper bound (95% CL, dimensionless); "
          "lower means tighter constraint on new physics")
    print("METRIC_DEF: detection_snr | direction=lower | "
          "desc=Detection SNR of Fourier power vs RMS noise; "
          "lower means stronger null result")
    print("METRIC_DEF: alpha_zd_upper_limit | direction=lower | "
          "desc=Upper limit on ZD coupling amplitude")
    print("METRIC_DEF: injection_recovery_fraction | direction=higher | "
          "desc=Fraction of injected signals recovered with delta_SNR > 2.0")
    print("METRIC_DEF: chi2_p_value_corrected | direction=higher | "
          "desc=Chi-squared p-value after whitening and red-noise correction")
    print("METRIC_DEF: bayes_factor_periodic | direction=lower | "
          "desc=GP Bayes factor for periodic vs smooth kernel")

    # -- Register conditions --
    conditions = get_all_conditions()
    # Use priority ordering; fall back to sorted keys for any unregistered
    cond_names = [c for c in CONDITION_PRIORITY if c in conditions]
    for c in sorted(conditions.keys()):
        if c not in cond_names:
            cond_names.append(c)

    print(f"REGISTERED_CONDITIONS: {', '.join(cond_names)}")
    print(f"REGIMES: {', '.join(REGIMES.keys())}")
    print(f"SEEDS: {SEEDS}")
    n_total_runs = len(cond_names) * len(REGIMES) * len(SEEDS)
    print(f"Total runs: {len(cond_names)} conditions x {len(REGIMES)} regimes "
          f"x {len(SEEDS)} seeds = {n_total_runs}")

    # -- Pilot timing --
    print("\n--- Pilot timing ---")
    set_all_seeds(42)
    pilot_galaxies = generate_manga_sample(
        n_galaxies=HYPERPARAMETERS["n_galaxies"],
        seed=42,
        x_min=HYPERPARAMETERS["x_min"],
        x_max=HYPERPARAMETERS["x_max"],
        n_points_per_galaxy=HYPERPARAMETERS["n_points_per_galaxy"],
    )
    t_pilot_start = time.time()
    pilot_cond = conditions[cond_names[0]]
    _ = pilot_cond.compute_metrics(pilot_galaxies, seed=42)
    t_pilot = time.time() - t_pilot_start
    t_data_gen = time.time() - t_start - t_pilot
    t_estimate = t_data_gen * len(SEEDS) + t_pilot * n_total_runs + 5.0
    print(f"Pilot: data_gen={t_data_gen:.2f}s, analysis={t_pilot:.4f}s")
    print(f"TIME_ESTIMATE: {t_estimate:.0f}s")

    max_seeds = max(3, min(len(SEEDS), int(time_budget * 0.8 / max(t_estimate, 1.0) * len(SEEDS))))
    active_seeds = SEEDS[:max_seeds]
    if max_seeds < len(SEEDS):
        print(f"SEED_WARNING: only {max_seeds} seeds used due to time budget")
    print(f"SEED_COUNT: {len(active_seeds)} (budget={time_budget}s, "
          f"pilot={t_pilot:.3f}s, conditions={len(cond_names)})")

    # -- Ablation check: verify conditions produce different outputs --
    print("\n--- Ablation check ---")
    test_gals = generate_manga_sample(n_galaxies=500, seed=99, x_min=0.5, x_max=1.35)
    ablation_outputs = {}
    for cname in cond_names:
        try:
            m = conditions[cname].compute_metrics(test_gals, seed=99)
            ablation_outputs[cname] = m.get("detection_snr", 0.0)
        except Exception:
            ablation_outputs[cname] = float("nan")

    checked_pairs = []
    for i, c1 in enumerate(cond_names):
        for c2 in cond_names[i + 1:]:
            v1 = ablation_outputs[c1]
            v2 = ablation_outputs[c2]
            differ = abs(v1 - v2) > 1e-6 or (math.isnan(v1) != math.isnan(v2))
            if not differ:
                checked_pairs.append((c1, c2))
            print(f"ABLATION_CHECK: {c1} vs {c2} outputs_differ={differ}")
    if checked_pairs:
        print(f"WARNING: {len(checked_pairs)} condition pair(s) with identical outputs")

    # -- Main experiment loop: BREADTH-FIRST then DEPTH --
    print("\n--- Main experiment ---")
    all_results = {}
    collected_metrics = []

    def should_stop():
        elapsed = time.time() - t_start
        return elapsed > time_budget * 0.80

    galaxy_cache = {}

    for pass_idx in range(2):
        if pass_idx == 0:
            seed_slice = active_seeds[:1]
            pass_label = "breadth-first"
        else:
            seed_slice = active_seeds[1:]
            pass_label = "depth"

        if not seed_slice:
            continue
        print(f"\n--- Pass {pass_idx + 1} ({pass_label}): seeds={seed_slice} ---")

        for seed in seed_slice:
            if should_stop():
                print("TIME_GUARD: stopping at 80% budget")
                break

            if seed not in galaxy_cache:
                set_all_seeds(seed)
                galaxy_cache[seed] = generate_manga_sample(
                    n_galaxies=HYPERPARAMETERS["n_galaxies"],
                    seed=seed,
                    x_min=HYPERPARAMETERS["x_min"],
                    x_max=HYPERPARAMETERS["x_max"],
                    n_points_per_galaxy=HYPERPARAMETERS["n_points_per_galaxy"],
                )

            all_galaxies = galaxy_cache[seed]

            for regime_name, regime_cfg in REGIMES.items():
                if should_stop():
                    print("TIME_GUARD: stopping at 80% budget")
                    break

                filtered = apply_regime_filter(all_galaxies, regime_name, regime_cfg)

                for cond_name in cond_names:
                    if should_stop():
                        break

                    key = (cond_name, regime_name)
                    if key not in all_results:
                        all_results[key] = {}

                    if seed in all_results[key]:
                        continue

                    try:
                        metrics = conditions[cond_name].compute_metrics(filtered, seed)

                        snr_val = metrics.get("detection_snr", 0.0)
                        if not np.isfinite(snr_val):
                            print(f"SKIP: NaN/Inf detected for {cond_name} "
                                  f"regime={regime_name} seed={seed}")
                            continue

                        all_results[key][seed] = metrics

                        alpha_val = metrics.get("alpha_zd_upper_limit", 0.0)
                        print(f"condition={cond_name} regime={regime_name} seed={seed} "
                              f"detection_snr: {snr_val:.4f}")
                        print(f"condition={cond_name} regime={regime_name} seed={seed} "
                              f"primary_metric: {snr_val:.4f}")
                        print(f"condition={cond_name} regime={regime_name} seed={seed} "
                              f"alpha_zd_upper_limit: {alpha_val:.6f}")

                        collected_metrics.append({
                            "condition": cond_name,
                            "regime": regime_name,
                            "seed": seed,
                            **{k: v for k, v in metrics.items()
                               if isinstance(v, (int, float))},
                        })

                    except Exception as e:
                        print(f"CONDITION_FAILED: {cond_name} regime={regime_name} "
                              f"seed={seed} {type(e).__name__}: {e}")
                        traceback.print_exc(file=sys.stdout)
                        continue

    # -- Aggregation --
    print("\n--- Aggregated results ---")
    summary_lines = []

    for cond_name in cond_names:
        for regime_name in REGIMES:
            key = (cond_name, regime_name)
            seed_results = all_results.get(key, {})
            if not seed_results:
                print(f"condition={cond_name} regime={regime_name} "
                      f"success_rate: 0/{len(active_seeds)}")
                continue

            n_success = len(seed_results)
            n_total = len(active_seeds)
            snr_values = [m["detection_snr"] for m in seed_results.values()]
            alpha_values = [m.get("alpha_zd_upper_limit", 0.0) for m in seed_results.values()]

            snr_mean = float(np.mean(snr_values))
            snr_std = float(np.std(snr_values, ddof=1)) if len(snr_values) > 1 else 0.0
            alpha_mean = float(np.mean(alpha_values))
            alpha_std = float(np.std(alpha_values, ddof=1)) if len(alpha_values) > 1 else 0.0

            print(f"condition={cond_name} regime={regime_name} "
                  f"success_rate: {n_success}/{n_total}")
            print(f"condition={cond_name} regime={regime_name} "
                  f"detection_snr_mean: {snr_mean:.4f} detection_snr_std: {snr_std:.4f}")
            print(f"condition={cond_name} regime={regime_name} "
                  f"primary_metric: {snr_mean:.4f}")
            print(f"condition={cond_name} regime={regime_name} "
                  f"alpha_zd_upper_limit_mean: {alpha_mean:.6f} "
                  f"alpha_zd_upper_limit_std: {alpha_std:.6f}")

            n_failed = n_total - n_success
            uncond_snr = (sum(snr_values) + n_failed * 10.0) / n_total
            print(f"condition={cond_name} regime={regime_name} "
                  f"unconditional_detection_snr_mean: {uncond_snr:.4f}")

            if len(snr_values) >= 3:
                snr_arr = np.array(snr_values)
                ci_lo = float(np.percentile(snr_arr, 2.5))
                ci_hi = float(np.percentile(snr_arr, 97.5))
                print(f"condition={cond_name} regime={regime_name} "
                      f"detection_snr_ci95: [{ci_lo:.4f}, {ci_hi:.4f}]")

            summary_lines.append(f"{cond_name}/{regime_name}={snr_mean:.3f}")

    # -- Paired analysis: each condition vs mean_stack_full_sample baseline --
    print("\n--- Paired analysis (vs mean_stack_full_sample baseline) ---")
    baseline_key_prefix = "mean_stack_full_sample"
    for cond_name in cond_names:
        if cond_name == baseline_key_prefix:
            continue
        for regime_name in REGIMES:
            base_key = (baseline_key_prefix, regime_name)
            test_key = (cond_name, regime_name)
            base_results = all_results.get(base_key, {})
            test_results = all_results.get(test_key, {})

            common_seeds = set(base_results.keys()) & set(test_results.keys())
            if len(common_seeds) < 3:
                continue

            diffs = []
            for s in sorted(common_seeds):
                d = test_results[s]["detection_snr"] - base_results[s]["detection_snr"]
                diffs.append(d)

            diffs = np.array(diffs)
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))
            if std_diff > 0 and len(diffs) >= 3:
                from scipy.stats import ttest_1samp
                t_stat, p_val = ttest_1samp(diffs, 0.0)
                print(f"PAIRED: {cond_name} vs {baseline_key_prefix} regime={regime_name} "
                      f"mean_diff={mean_diff:.4f} std_diff={std_diff:.4f} "
                      f"t_stat={t_stat:.3f} p_value={p_val:.4f}")
            else:
                print(f"PAIRED: {cond_name} vs {baseline_key_prefix} regime={regime_name} "
                      f"mean_diff={mean_diff:.4f} std_diff={std_diff:.4f}")

    # -- Degenerate metric check --
    print("\n--- Metric discrimination check ---")
    for regime_name in REGIMES:
        means = []
        for cond_name in cond_names:
            key = (cond_name, regime_name)
            seed_results = all_results.get(key, {})
            if seed_results:
                means.append(np.mean([m["detection_snr"] for m in seed_results.values()]))
        if len(means) >= 2:
            spread = max(means) - min(means)
            if spread < 1e-4:
                print(f"WARNING: DEGENERATE_METRICS regime={regime_name} "
                      f"all conditions have same mean={means[0]:.4f}")
            else:
                print(f"DISCRIMINATION: regime={regime_name} "
                      f"spread={spread:.4f} min={min(means):.4f} max={max(means):.4f}")

    # -- Summary --
    print(f"\nSUMMARY: {', '.join(summary_lines[:20])}")
    if len(summary_lines) > 20:
        print(f"  ... and {len(summary_lines) - 20} more")

    # -- Save results --
    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s / {time_budget}s budget")

    output = {
        "hyperparameters": {k: v for k, v in HYPERPARAMETERS.items()
                           if not isinstance(v, (list, dict)) or isinstance(v, list)},
        "primary_metric": "detection_snr",
        "primary_metric_direction": "lower",
        "seeds_used": active_seeds,
        "n_conditions": len(cond_names),
        "conditions_list": cond_names,
        "metrics": collected_metrics,
        "summary": {
            key[0] + "/" + key[1]: {
                "detection_snr_mean": float(np.mean([m["detection_snr"] for m in v.values()])),
                "detection_snr_std": float(np.std([m["detection_snr"] for m in v.values()], ddof=1))
                if len(v) > 1 else 0.0,
                "n_seeds": len(v),
            }
            for key, v in all_results.items()
            if v
        },
        "elapsed_seconds": elapsed,
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
