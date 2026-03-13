import re

file_path = "crates/gororoba_cli_physics/src/bin/gravastar_orbit_crucible.rs"

with open(file_path, "r") as f:
    content = f.read()

# Replace the duplicated loop code causing the syntax error
old_code = """    let elapsed = t_start.elapsed();
    eprintln!("Integration complete in {:.2?}", elapsed);

    let e_ctrl_final = compute_total_energy(&states_ctrl);
    let e_grav_final = compute_total_energy(&states_grav);
    let final_drift_grav = (e_grav_final - e0_grav) / e0_grav.abs();
    eprintln!("--- Final Energy Drift ---");
    eprintln!("Schwarzschild: {:.5e}", (e_ctrl_final - e0_ctrl) / e0_ctrl.abs());
    eprintln!("Gravastar:     {:.5e}", final_drift_grav);

    if is_ska_mode {
        eprintln!("SKA Monte Carlo threshold for non-symplectic drift: < 1e-11");
        assert!(
            final_drift_grav.abs() < 1e-10,
            "Symplectic drift failed SKA PTA threshold! Drift: {}",
            final_drift_grav
        );
    }                    step,
                    t_years,
                    name,
                    r_ctrl / 1.496e8, // AU
                    r_grav / 1.496e8,
                    delta_r,
                    drift_ctrl,
                    drift_grav,
                    coupling
                );
            }
        }
    }"""

new_code = """    let elapsed = t_start.elapsed();
    eprintln!("Integration complete in {:.2?}", elapsed);

    let e_ctrl_final = compute_total_energy(&states_ctrl);
    let e_grav_final = compute_total_energy(&states_grav);
    let final_drift_grav = (e_grav_final - e0_grav) / e0_grav.abs();
    eprintln!("--- Final Energy Drift ---");
    eprintln!("Schwarzschild: {:.5e}", (e_ctrl_final - e0_ctrl) / e0_ctrl.abs());
    eprintln!("Gravastar:     {:.5e}", final_drift_grav);

    if is_ska_mode {
        eprintln!("SKA Monte Carlo threshold for non-symplectic drift: < 1e-11");
        assert!(
            final_drift_grav.abs() < 1e-10,
            "Symplectic drift failed SKA PTA threshold! Drift: {}",
            final_drift_grav
        );
    }"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, "w") as f:
        f.write(content)
    print("Patched!")
else:
    print("Could not find the target code to patch.")
