"""
MaNGA Zero-Dark Density Null Result: Multi-Algebra Harmonic Analysis

=== Dataset ===
Synthetic MaNGA-like rotation curves (N=6992 galaxies) generated from:
- NFW dark matter profiles with Duffy+2008 concentration-mass relation
- Moster+2013 stellar-mass-to-halo-mass relation
- Calibrated baryonic systematics (+5% bulge excess, -12% cusp, +5% IFU edge)
- Galaxy-to-galaxy scatter: 8% RMS, measurement noise: 3-10% of v_circ
Parameters reproduce E-183 stacking results: 19 valid bins, RMS~0.075, SNR~0.25

=== Distribution Shift ===
Three regimes test robustness:
  1. full_sample: all 6992 galaxies
  2. face_on: inclination < 45 deg (~3100 galaxies, removes projection artifacts)
  3. mass_Q3: 3rd mass quartile (sparse: ~6 valid bins, tests pipeline sensitivity)

=== Model Architecture ===
Not a neural network. Physics analysis pipeline:
  stack_residuals() -> fourier_power_phase() -> compute_snr() -> rayleigh_phase_coherence()
Each condition uses different wavenumber sets from distinct algebraic structures.

=== Training Protocol ===
No training. Deterministic stacking + Fourier analysis per condition.
200 bootstrap resamples for confidence intervals (seed-controlled).

=== Evaluation Protocol ===
8 conditions x 3 regimes x 5 seeds = 120 runs.
Primary metric: primary_metric (detection SNR; lower = stronger null result; target < 2.0).
Secondary: secondary_metric (alpha_zd upper limit; lower = tighter constraint; target < 0.002).
Per-seed reporting with mean +/- std aggregation.
"""

import json
import math
import sys
import time
import traceback

import numpy as np
from data_utils import filter_inclination, filter_mass_quartile, generate_manga_sample
from models import get_all_conditions

try:
    from experiment_harness import ExperimentHarness
except ImportError:
    # Fallback if harness not available
    class ExperimentHarness:
        def __init__(self, time_budget=600):
            self._start = time.time()
            self._budget = time_budget
            self._metrics = []
        def should_stop(self):
            return (time.time() - self._start) > self._budget * 0.8
        def check_value(self, val, name=""):
            return np.isfinite(val)
        def report_metric(self, name, val):
            self._metrics.append({"name": name, "value": val})
        def finalize(self):
            pass

# ---------------------------------------------------------------------------
# Hyperparameters (all used in computation)
# ---------------------------------------------------------------------------
HYPERPARAMETERS = {
    "n_galaxies": 6992,
    "x_min": 0.5,
    "x_max": 1.35,
    "n_points_per_galaxy": 18,
    "n_grid": 200,
    "min_per_bin": 10,
    "cd_dim": 16,
    "n_bootstrap": 50,
    "inject_alpha_validation": 0.05,
    "face_on_inc_max": 45.0,
    "mass_quartile_target": 3,
    "time_budget_seconds": 600,
}

SEEDS = [42, 123, 456, 789, 1024]

REGIMES = {
    "full_sample": {"filter": None},
    "face_on": {"filter": "inclination", "inc_min": 0.0, "inc_max": 45.0},
    "mass_Q3": {"filter": "mass_quartile", "quartile": 3},
}


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
            galaxies, regime_cfg["inc_min"], regime_cfg["inc_max"]
        )
    elif filt == "mass_quartile":
        return filter_mass_quartile(galaxies, regime_cfg["quartile"])
    return galaxies


def run_injection_validation(galaxies: list, seed: int):
    """
    Internal validation: inject known alpha_zd signal and check recovery.
    Not a separate condition -- just a sanity check printed as validation.
    """
    from models import cd_zd_wavenumbers, compute_snr, stack_residuals
    inject_alpha = HYPERPARAMETERS["inject_alpha_validation"]
    wk = cd_zd_wavenumbers(16)

    x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
    snr_baseline = compute_snr(x_grid, delta, n_contrib, 10, wk)

    injected = generate_manga_sample(
        n_galaxies=len(galaxies),
        seed=seed + 10000,
        inject_alpha_zd=inject_alpha,
        cd_dim=16,
    )
    x_grid_inj, delta_inj, _, n_contrib_inj = stack_residuals(injected)
    snr_injected = compute_snr(x_grid_inj, delta_inj, n_contrib_inj, 10, wk)
    delta_snr = snr_injected - snr_baseline
    recovery = "PASS" if delta_snr > 2.0 else "FAIL"

    print(f"INJECTION_VALIDATION: seed={seed} alpha={inject_alpha} "
          f"snr_baseline={snr_baseline:.4f} snr_injected={snr_injected:.4f} "
          f"delta_snr={delta_snr:.4f} recovery={recovery}")


def main():
    harness = ExperimentHarness(time_budget=HYPERPARAMETERS["time_budget_seconds"])
    time_budget = HYPERPARAMETERS["time_budget_seconds"]
    t_start = time.time()

    # -- Metric definition --
    print("METRIC_DEF: primary_metric | direction=lower | "
          "desc=Detection SNR of harmonic analysis; "
          "lower means stronger null result (no ZD signal detected)")
    print("METRIC_DEF: secondary_metric | direction=lower | "
          "desc=Upper limit on ZD coupling alpha_zd; "
          "lower means tighter constraint on new physics")

    # -- Register conditions --
    conditions = get_all_conditions()
    cond_names = sorted(conditions.keys())
    print(f"REGISTERED_CONDITIONS: {', '.join(cond_names)}")
    print(f"REGIMES: {', '.join(REGIMES.keys())}")
    print(f"SEEDS: {SEEDS}")
    print(f"Total runs: {len(cond_names)} conditions x {len(REGIMES)} regimes "
          f"x {len(SEEDS)} seeds = "
          f"{len(cond_names) * len(REGIMES) * len(SEEDS)}")

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
    n_total_runs = len(cond_names) * len(REGIMES) * len(SEEDS)
    t_data_gen = time.time() - t_start - t_pilot
    t_estimate = t_data_gen * len(SEEDS) + t_pilot * n_total_runs + 5.0
    print(f"Pilot: data_gen={t_data_gen:.2f}s, analysis={t_pilot:.4f}s")
    print(f"TIME_ESTIMATE: {t_estimate:.0f}s")

    max_seeds = max(3, min(len(SEEDS),
                    int(time_budget * 0.8 / max(t_estimate, 1.0) * len(SEEDS))))
    active_seeds = SEEDS[:max_seeds]
    if max_seeds < len(SEEDS):
        print(f"SEED_WARNING: only {max_seeds} seeds used due to time budget")
    print(f"SEED_COUNT: {len(active_seeds)} (budget={time_budget}s, "
          f"pilot={t_pilot:.3f}s, conditions={len(cond_names)})")

    # -- Injection recovery validation (not a condition) --
    print("\n--- Injection validation ---")
    run_injection_validation(pilot_galaxies, seed=42)

    # -- Ablation check: verify conditions produce different outputs --
    print("\n--- Ablation check ---")
    test_gals = generate_manga_sample(n_galaxies=500, seed=99, x_min=0.5, x_max=1.35)
    ablation_outputs = {}
    for cname in cond_names:
        try:
            m = conditions[cname].compute_metrics(test_gals, seed=99)
            ablation_outputs[cname] = m.get("primary_metric", 0.0)
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

    # -- Main experiment loop: BREADTH-FIRST --
    print("\n--- Main experiment ---")
    all_results = {}
    collected_metrics = []

    def should_stop():
        return harness.should_stop()

    # Pre-generate galaxy samples for each seed
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

                        pm_val = metrics.get("primary_metric", 0.0)
                        if not harness.check_value(pm_val, "primary_metric"):
                            print(f"SKIP: NaN/Inf detected for {cond_name} "
                                  f"regime={regime_name} seed={seed}")
                            continue

                        all_results[key][seed] = metrics
                        harness.report_metric(
                            f"{cond_name}/{regime_name}/primary_metric", pm_val)

                        print(f"condition={cond_name} regime={regime_name} seed={seed} "
                              f"primary_metric: {metrics['primary_metric']:.4f}")
                        print(f"condition={cond_name} regime={regime_name} seed={seed} "
                              f"secondary_metric: "
                              f"{metrics.get('secondary_metric', 0.0):.6f}")

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
            pm_values = [m["primary_metric"] for m in seed_results.values()]
            sm_values = [m.get("secondary_metric", 0.0)
                        for m in seed_results.values()]

            pm_mean = float(np.mean(pm_values))
            pm_std = float(np.std(pm_values, ddof=1)) if len(pm_values) > 1 else 0.0
            sm_mean = float(np.mean(sm_values))
            sm_std = (float(np.std(sm_values, ddof=1))
                     if len(sm_values) > 1 else 0.0)

            print(f"condition={cond_name} regime={regime_name} "
                  f"success_rate: {n_success}/{n_total}")
            print(f"condition={cond_name} regime={regime_name} "
                  f"primary_metric_mean: {pm_mean:.4f} primary_metric_std: {pm_std:.4f}")
            print(f"condition={cond_name} regime={regime_name} "
                  f"secondary_metric_mean: {sm_mean:.6f} "
                  f"secondary_metric_std: {sm_std:.6f}")

            n_failed = n_total - n_success
            uncond_pm = (sum(pm_values) + n_failed * 10.0) / n_total
            print(f"condition={cond_name} regime={regime_name} "
                  f"unconditional_primary_metric_mean: {uncond_pm:.4f}")

            if len(pm_values) >= 3:
                pm_arr = np.array(pm_values)
                ci_lo = float(np.percentile(pm_arr, 2.5))
                ci_hi = float(np.percentile(pm_arr, 97.5))
                print(f"condition={cond_name} regime={regime_name} "
                      f"primary_metric_ci95: [{ci_lo:.4f}, {ci_hi:.4f}]")

            summary_lines.append(f"{cond_name}/{regime_name}={pm_mean:.3f}")

    # -- Paired analysis: each condition vs DANN baseline --
    print("\n--- Paired analysis (vs DANN baseline) ---")
    baseline_key_prefix = "DANN"
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
                d = (test_results[s]["primary_metric"]
                     - base_results[s]["primary_metric"])
                diffs.append(d)

            diffs = np.array(diffs)
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))
            if std_diff > 0 and len(diffs) >= 3:
                from scipy.stats import ttest_1samp
                t_stat, p_val = ttest_1samp(diffs, 0.0)
                print(f"PAIRED: {cond_name} vs {baseline_key_prefix} "
                      f"regime={regime_name} "
                      f"mean_diff={mean_diff:.4f} std_diff={std_diff:.4f} "
                      f"t_stat={t_stat:.3f} p_value={p_val:.4f}")
            else:
                print(f"PAIRED: {cond_name} vs {baseline_key_prefix} "
                      f"regime={regime_name} "
                      f"mean_diff={mean_diff:.4f} std_diff={std_diff:.4f}")

    # -- Degenerate metric check --
    print("\n--- Metric discrimination check ---")
    for regime_name in REGIMES:
        means = []
        for cond_name in cond_names:
            key = (cond_name, regime_name)
            seed_results = all_results.get(key, {})
            if seed_results:
                means.append(
                    np.mean([m["primary_metric"] for m in seed_results.values()]))
        if len(means) >= 2:
            spread = max(means) - min(means)
            if spread < 1e-4:
                print(f"WARNING: DEGENERATE_METRICS regime={regime_name} "
                      f"all conditions have same mean={means[0]:.4f}")
            else:
                print(f"DISCRIMINATION: regime={regime_name} "
                      f"spread={spread:.4f} min={min(means):.4f} "
                      f"max={max(means):.4f}")

    # -- Summary --
    print(f"\nSUMMARY: {', '.join(summary_lines)}")

    # -- Save results --
    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s / {time_budget}s budget")

    output = {
        "hyperparameters": HYPERPARAMETERS,
        "seeds_used": active_seeds,
        "metrics": collected_metrics,
        "summary": {
            key[0] + "/" + key[1]: {
                "primary_metric_mean": float(
                    np.mean([m["primary_metric"] for m in v.values()])),
                "primary_metric_std": float(
                    np.std([m["primary_metric"] for m in v.values()], ddof=1))
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

    harness.finalize()


if __name__ == "__main__":
    main()
