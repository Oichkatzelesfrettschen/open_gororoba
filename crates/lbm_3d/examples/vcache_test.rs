use lbm_3d::solver::LbmSolver3D;

fn main() {
    println!("=== Testing V-Cache Tiling ===");

    // Create a massive grid: 128 x 128 x 128.
    // 128^3 = 2,097,152 cells.
    // At ~328 bytes per cell, this is ~687 MB, way past the 86.4 MB L3 target.
    let mut solver = LbmSolver3D::new(128, 128, 128, 1.0);

    // Evolve exactly one step to trigger the cache tiling debug log
    solver.evolve_one_step();

    println!("Step 1 complete.");
}
