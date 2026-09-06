//! Fixed source parameters separate frequency rounding from curve fitting.

use optics_core::{
    fano_tcmt::{CrossSections, FanoDrudeParams},
    mie_cylinder::{
        ConcentricCylinder, ruan_fan_mdm_fig5, try_mie_scattering, try_scattering_channel,
    },
};
use std::{error::Error, fs::OpenOptions, io::Write};

fn retained_channel_sum(
    geometry: &ConcentricCylinder,
    omega: f64,
) -> Result<CrossSections, Box<dyn Error>> {
    let mut result = CrossSections {
        c_sct: 0.0,
        c_abs: 0.0,
        c_ext: 0.0,
    };
    for (order, multiplicity) in [(0, 1.0), (1, 2.0), (2, 2.0)] {
        let value = try_scattering_channel(geometry, order, omega)?.cross_sections;
        result.c_sct += multiplicity * value.scattering;
        result.c_abs += multiplicity * value.absorption;
        result.c_ext += multiplicity * value.extinction;
    }
    Ok(result)
}

#[test]
fn printed_frequency_envelope_retains_exact_anchor_failure() -> Result<(), Box<dyn Error>> {
    let geometry = ruan_fan_mdm_fig5(&FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    });
    let signed = try_mie_scattering(&geometry, 0.2282, 2)?.cross_sections;
    let exact = retained_channel_sum(&geometry, 0.2282)?;
    println!(
        "signed-minus-weighted scattering={:.17e} absorption={:.17e}",
        signed.c_sct - exact.c_sct,
        signed.c_abs - exact.c_abs
    );
    println!(
        "exact scattering={:.17e} absorption={:.17e}",
        exact.c_sct, exact.c_abs
    );
    assert!((exact.c_sct - 0.141_739_879_640_738_14).abs() < 1e-12);
    assert!((exact.c_abs - 0.398_104_096_787_583_85).abs() < 1e-12);
    let mut output = String::from("intervals,index,omega,scattering,absorption,joint_anchor\n");
    for intervals in [200, 2000] {
        let mut scattering_range = [f64::INFINITY, f64::NEG_INFINITY];
        let mut absorption_range = [f64::INFINITY, f64::NEG_INFINITY];
        let mut joint_count = 0;
        for index in 0..=intervals {
            let omega = 0.22815 + 0.0001 * index as f64 / intervals as f64;
            let values = retained_channel_sum(&geometry, omega)?;
            assert!(values.c_sct.is_finite() && values.c_abs.is_finite());
            scattering_range[0] = scattering_range[0].min(values.c_sct);
            scattering_range[1] = scattering_range[1].max(values.c_sct);
            absorption_range[0] = absorption_range[0].min(values.c_abs);
            absorption_range[1] = absorption_range[1].max(values.c_abs);
            let joint =
                (0.025..=0.035).contains(&values.c_sct) && (0.315..=0.325).contains(&values.c_abs);
            joint_count += usize::from(joint);
            output.push_str(&format!(
                "{intervals},{index},{omega:.17e},{:.17e},{:.17e},{joint}\n",
                values.c_sct, values.c_abs
            ));
        }
        println!(
            "intervals={intervals} scattering={scattering_range:?} absorption={absorption_range:?} joint_count={joint_count}"
        );
        assert_eq!(joint_count, 0, "sampled source-anchor exclusion failed");
    }
    if let Ok(path) = std::env::var("OPTICS_FREQUENCY_ENVELOPE_OUTPUT") {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(output.as_bytes())?;
        file.sync_all()?;
    }
    Ok(())
}
