use gororoba_algebra::physics::octonion_field::FieldParams;
use gororoba_engine::simulation::state_3d::{SimulationConfig3D, SimulationState3D};
use lbm_3d_cuda::Precision;
use std::error::Error;

pub fn run() -> Result<(), Box<dyn Error>> {
    let res: usize = 128;
    let prec = Precision::FP32;

    let config = SimulationConfig3D {
        nx: res,
        ny: res,
        nz: res,
        tau: 0.6,
        use_gpu: true,
        precision: prec,
        algebra_params: FieldParams::default(),
        coupling_fluid_algebra: 0.1,
        coupling_algebra_fluid: 0.1,
    };

    let mut state = SimulationState3D::new(config)?;

    // Initial step to populate velocities
    state.step()?;

    // Calculate enstrophy once
    let enstrophy = match state.fluid {
        gororoba_engine::simulation::state_3d::LbmBackend3D::Gpu(ref mut solver) => {
            solver.calculate_enstrophy()?
        }
        _ => panic!("GPU backend not enabled or selected"),
    };

    println!("Enstrophy: {}", enstrophy);
    println!("ENSTROPHY_TEST_COMPLETE");

    Ok(())
}
