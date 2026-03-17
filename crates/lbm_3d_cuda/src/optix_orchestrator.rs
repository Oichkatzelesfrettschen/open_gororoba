//! Async Eulerian-Lagrangian orchestrator for concurrent LBM + OptiX tracing.
//!
//! Manages two CUDA streams to overlap LBM collision-streaming (SMs) with
//! OptiX particle tracing (RT cores):
//!
//! ```text
//! Stream A (LBM):    step N  ->  step N+1  ->  step N+2  ->  ...
//! Stream B (OptiX):            trace N   ->  (idle)    ->  trace N+2
//! BVH rebuild:     (every bvh_rebuild_interval steps on Stream B)
//! ```
//!
//! The key invariant: OptiX traces particles using the macroscopic velocity
//! field from the LAST COMPLETED LBM step. The two streams overlap: while
//! LBM computes step N+1 on the SMs, OptiX traces step N's velocity field
//! on the RT cores. A CUDA event (`lbm_step_done`) synchronizes the handoff.
//!
//! # Memory lifecycle for ephemeral FP16 velocity buffer
//!
//! 1. LBM step N completes on Stream A (event `lbm_step_done` recorded)
//! 2. Stream B waits on `lbm_step_done`
//! 3. `write_velocity_fp16` kernel dispatches on Stream B (computes u from f)
//! 4. OptiX traces particles using the FP16 velocity buffer on Stream B
//! 5. FP16 velocity buffer can be freed after trace completes
//! 6. Stream A continues LBM step N+1 concurrently with steps 3-5

use crate::optix_pipeline::OptiXPipeline;
use crate::optix_tracer::OptiXTracerConfig;

/// Orchestrator state for concurrent LBM + OptiX execution.
#[derive(Debug)]
pub struct EulerianLagrangianOrchestrator {
    /// Current LBM simulation step.
    pub lbm_step: u64,
    /// OptiX pipeline configuration.
    pub pipeline: OptiXPipeline,
    /// Whether particle tracing is active.
    pub tracing_enabled: bool,
    /// Number of tracer particles.
    pub n_particles: u32,
    /// Particle positions (host-side buffer for readback).
    pub particle_positions: Vec<[f32; 3]>,
    /// Particle velocities (host-side buffer for readback).
    pub particle_velocities: Vec<[f32; 3]>,
    /// Step interval for trajectory snapshot output.
    pub snapshot_interval: u64,
    /// Accumulated trajectory data: (step, particle_id, x, y, z, vx, vy, vz).
    pub trajectory_log: Vec<TrajectoryPoint>,
}

/// A single point in a particle trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPoint {
    pub step: u64,
    pub particle_id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

/// Configuration for the Eulerian-Lagrangian orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// OptiX tracer configuration.
    pub tracer: OptiXTracerConfig,
    /// Number of tracer particles.
    pub n_particles: u32,
    /// Whether to enable tracing from the start.
    pub tracing_enabled: bool,
    /// Step interval for trajectory snapshots (0 = every trace step).
    pub snapshot_interval: u64,
    /// Velocity device pointer (from LBM solver).
    pub velocity_device_ptr: u64,
    /// Density device pointer (from LBM solver).
    pub density_device_ptr: u64,
}

impl EulerianLagrangianOrchestrator {
    /// Create a new orchestrator.
    pub fn new(config: OrchestratorConfig) -> Self {
        let pipeline = OptiXPipeline::new(
            Default::default(),
            config.tracer.clone(),
            config.velocity_device_ptr,
            config.density_device_ptr,
        );

        // Initialize particles uniformly distributed in the grid
        let (nx, ny, nz) = config.tracer.grid_dim;
        let mut positions = Vec::with_capacity(config.n_particles as usize);
        let mut velocities = Vec::with_capacity(config.n_particles as usize);

        // Simple uniform seeding via modular arithmetic
        for i in 0..config.n_particles {
            let fx = (i % nx) as f32 + 0.5;
            let fy = ((i / nx) % ny) as f32 + 0.5;
            let fz = ((i / (nx * ny)) % nz) as f32 + 0.5;
            positions.push([
                config.tracer.grid_origin[0] + fx * config.tracer.cell_size[0],
                config.tracer.grid_origin[1] + fy * config.tracer.cell_size[1],
                config.tracer.grid_origin[2] + fz * config.tracer.cell_size[2],
            ]);
            velocities.push([0.0, 0.0, 0.0]);
        }

        Self {
            lbm_step: 0,
            pipeline,
            tracing_enabled: config.tracing_enabled,
            n_particles: config.n_particles,
            particle_positions: positions,
            particle_velocities: velocities,
            snapshot_interval: config.snapshot_interval.max(1),
            trajectory_log: Vec::new(),
        }
    }

    /// Advance one LBM step and optionally trace particles.
    ///
    /// The orchestrator decides whether to trace based on the BVH rebuild
    /// interval and the snapshot interval. The caller is responsible for
    /// actually dispatching the LBM kernel on Stream A.
    ///
    /// Returns `true` if particle tracing should be dispatched this step.
    pub fn should_trace_this_step(&self) -> bool {
        if !self.tracing_enabled {
            return false;
        }
        // Trace at the snapshot interval
        self.lbm_step.is_multiple_of(self.snapshot_interval)
    }

    /// Notify the orchestrator that an LBM step has completed.
    ///
    /// If tracing is due, the orchestrator will:
    /// 1. Check if BVH needs rebuilding
    /// 2. Record the current particle positions into the trajectory log
    pub fn advance_step(&mut self) {
        self.lbm_step += 1;

        if self.should_trace_this_step() {
            // Log trajectory snapshot
            for (pid, (pos, vel)) in self
                .particle_positions
                .iter()
                .zip(self.particle_velocities.iter())
                .enumerate()
            {
                self.trajectory_log.push(TrajectoryPoint {
                    step: self.lbm_step,
                    particle_id: pid as u32,
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                    vx: vel[0],
                    vy: vel[1],
                    vz: vel[2],
                });
            }
        }
    }

    /// Check if the BVH needs rebuilding (amortized).
    pub fn needs_bvh_rebuild(&self, current_occupancy: f64) -> bool {
        self.pipeline.needs_rebuild(self.lbm_step, current_occupancy)
    }

    /// Update particle positions from device readback.
    pub fn update_particles_from_readback(
        &mut self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
    ) {
        let n = positions.len().min(self.n_particles as usize);
        self.particle_positions[..n].copy_from_slice(&positions[..n]);
        self.particle_velocities[..n].copy_from_slice(&velocities[..n]);
    }

    /// Write trajectory log to CSV.
    pub fn write_trajectory_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "step,particle_id,x,y,z,vx,vy,vz")?;
        for pt in &self.trajectory_log {
            writeln!(
                f,
                "{},{},{:.6},{:.6},{:.6},{:.6e},{:.6e},{:.6e}",
                pt.step, pt.particle_id, pt.x, pt.y, pt.z, pt.vx, pt.vy, pt.vz
            )?;
        }
        Ok(())
    }

    /// Summary statistics.
    pub fn summary(&self) -> String {
        format!(
            "EL Orchestrator: step={}, particles={}, trajectory_points={}, \
             tracing={}, bvh_valid={}",
            self.lbm_step,
            self.n_particles,
            self.trajectory_log.len(),
            self.tracing_enabled,
            self.pipeline.gas_valid,
        )
    }
}

/// The CUDA stream event synchronization protocol for concurrent execution.
///
/// This documents the exact cuStreamWaitEvent sequence needed for the FFI layer.
pub fn stream_protocol_documentation() -> &'static str {
    r#"CUDA Stream Synchronization Protocol for Eulerian-Lagrangian:

Setup:
  cuStreamCreate(&stream_lbm, CU_STREAM_NON_BLOCKING)
  cuStreamCreate(&stream_optix, CU_STREAM_NON_BLOCKING)
  cuEventCreate(&evt_lbm_done, CU_EVENT_DISABLE_TIMING)
  cuEventCreate(&evt_trace_done, CU_EVENT_DISABLE_TIMING)

Per-step loop:
  1. Dispatch LBM ephemeral kernel on stream_lbm:
     lbm_step_soa_mrt_aa_ephemeral<<<grid, block, 0, stream_lbm>>>(
         f, NULL, NULL, tau, force_or_null, nx, ny, nz, parity, 0 /*no macro write*/
     )

  2. If should_trace_this_step():
     a. Record event after LBM step:
        cuEventRecord(evt_lbm_done, stream_lbm)

     b. Stream B waits for LBM completion:
        cuStreamWaitEvent(stream_optix, evt_lbm_done, 0)

     c. Write FP16 velocity on Stream B (computes u from f on-the-fly):
        write_velocity_fp16<<<grid, block, 0, stream_optix>>>(
            NULL /*ephemeral*/, f, u_fp16, nx, ny, nz
        )

     d. If needs_bvh_rebuild():
        scan_brick_occupancy<<<...>>>(rho, bricks, count, ...)
        optixAccelBuild(context, stream_optix, ...)

     e. OptiX trace on Stream B:
        optixLaunch(pipeline, stream_optix, params, sizeof(params), &sbt, n_particles, 1, 1)

     f. Record event after trace:
        cuEventRecord(evt_trace_done, stream_optix)

  3. Next LBM step can start immediately on stream_lbm (no wait needed --
     LBM reads/writes f[], OptiX only reads the FP16 u_fp16[] snapshot).

  4. Before reading particle positions back to host:
     cuStreamWaitEvent(stream_lbm, evt_trace_done, 0)  // optional, only if host needs results
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let config = OrchestratorConfig {
            tracer: OptiXTracerConfig::default(),
            n_particles: 100,
            tracing_enabled: true,
            snapshot_interval: 10,
            velocity_device_ptr: 0,
            density_device_ptr: 0,
        };
        let orch = EulerianLagrangianOrchestrator::new(config);
        assert_eq!(orch.n_particles, 100);
        assert_eq!(orch.particle_positions.len(), 100);
        assert!(orch.tracing_enabled);
    }

    #[test]
    fn test_should_trace() {
        let config = OrchestratorConfig {
            tracer: OptiXTracerConfig::default(),
            n_particles: 10,
            tracing_enabled: true,
            snapshot_interval: 5,
            velocity_device_ptr: 0,
            density_device_ptr: 0,
        };
        let mut orch = EulerianLagrangianOrchestrator::new(config);
        assert!(orch.should_trace_this_step()); // step 0
        orch.lbm_step = 3;
        assert!(!orch.should_trace_this_step());
        orch.lbm_step = 5;
        assert!(orch.should_trace_this_step());
    }

    #[test]
    fn test_advance_step_logs_trajectory() {
        let config = OrchestratorConfig {
            tracer: OptiXTracerConfig::default(),
            n_particles: 3,
            tracing_enabled: true,
            snapshot_interval: 1,
            velocity_device_ptr: 0,
            density_device_ptr: 0,
        };
        let mut orch = EulerianLagrangianOrchestrator::new(config);
        orch.advance_step(); // step 1
        orch.advance_step(); // step 2
        // 2 steps * 3 particles = 6 trajectory points
        assert_eq!(orch.trajectory_log.len(), 6);
        assert_eq!(orch.trajectory_log[0].step, 1);
        assert_eq!(orch.trajectory_log[3].step, 2);
    }

    #[test]
    fn test_trajectory_csv_output() {
        let config = OrchestratorConfig {
            tracer: OptiXTracerConfig::default(),
            n_particles: 2,
            tracing_enabled: true,
            snapshot_interval: 1,
            velocity_device_ptr: 0,
            density_device_ptr: 0,
        };
        let mut orch = EulerianLagrangianOrchestrator::new(config);
        orch.advance_step();
        let path = "/tmp/test_trajectory.csv";
        orch.write_trajectory_csv(path).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.starts_with("step,particle_id,x,y,z,vx,vy,vz\n"));
        assert!(content.lines().count() >= 3); // header + 2 particles
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_stream_protocol_documented() {
        let doc = stream_protocol_documentation();
        assert!(doc.contains("cuStreamCreate"));
        assert!(doc.contains("cuEventRecord"));
        assert!(doc.contains("cuStreamWaitEvent"));
        assert!(doc.contains("optixLaunch"));
    }
}
