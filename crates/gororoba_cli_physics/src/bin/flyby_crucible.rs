//! flyby-crucible: Test the 64D Chingon non-alternative drag against
//! anomalous spacecraft flyby velocity jumps.

use gr_core::forces::chingon_drag::compute_chingon_drag;
use algebra_core::construction::chingon::AlternativityViolationTensor;
use nalgebra::Vector3;

fn main() -> anyhow::Result<()> {
    println!("=== 64D Chingon-Vlasov Flyby Crucible ===");

    // 1. Setup Parameters
    let mass_earth = 398600.4418; // GM in km^3/s^2
    let coupling_chingon = 8.0e-14; // alpha_chingon
    let rho_dm = 1.0;
    
    let v_wind_rel = Vector3::new(-30.0, 220.0, 0.0);

    // 2. Load 64D Alternativity Violation Tensor
    println!("Computing 64D AVT...");
    let avt = AlternativityViolationTensor::new(64);

    // 3. Define Flyby Trajectory
    let pos = Vector3::new(-30000.0, -15000.0, 0.0);
    let vel = Vector3::new(8.0, 5.0, 0.0);
    
    let dt = 1.0; // Fixed 1s steps for high precision integration
    let steps = 10000;

    println!("Integrating trajectory ({} steps)...", steps);

    // 4. Integration Loops
    let run_sim = |use_chingon: bool| {
        let mut p = pos;
        let mut v = vel;
        
        for _ in 0..steps {
            // Standard RK4 step
            let accel = |p_in: Vector3<f64>, v_in: Vector3<f64>| {
                let r_sq = p_in.norm_squared();
                let r = r_sq.sqrt();
                let a_grav = -p_in * (mass_earth / (r_sq * r));
                if use_chingon {
                    a_grav + compute_chingon_drag(v_in - v_wind_rel, rho_dm, coupling_chingon, &avt)
                } else {
                    a_grav
                }
            };

            let k1_v = accel(p, v);
            let k1_p = v;
            
            let k2_v = accel(p + k1_p * (dt/2.0), v + k1_v * (dt/2.0));
            let k2_p = v + k1_v * (dt/2.0);
            
            let k3_v = accel(p + k2_p * (dt/2.0), v + k2_v * (dt/2.0));
            let k3_p = v + k2_v * (dt/2.0);
            
            let k4_v = accel(p + k3_p * dt, v + k3_v * dt);
            let k4_p = v + k3_v * dt;
            
            p += (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p) * (dt / 6.0);
            v += (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v) * (dt / 6.0);
        }
        v.norm()
    };

    let v_out_ctrl = run_sim(false);
    let v_out_chingon = run_sim(true);

    // 5. Verdict
    let delta_v = (v_out_chingon - v_out_ctrl) * 1e6; // mm/s
    println!("\n=== Final Results ===");
    println!("Control Outbound:  {:.10} km/s", v_out_ctrl);
    println!("Chingon Outbound:  {:.10} km/s", v_out_chingon);
    println!("Chingon Delta-V:   {:.4} mm/s", delta_v);

    Ok(())
}
