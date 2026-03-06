use nalgebra::Vector3;

/// A sampler for the dark matter wind fluid, bridging LBM to N-Body.
pub struct FluidWindSampler {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub rho: Vec<f32>,
    pub velocity: Vec<f32>, // Flat [u_x, u_y, u_z] x N
}

impl FluidWindSampler {
    pub fn new(nx: usize, ny: usize, nz: usize, rho: Vec<f32>, velocity: Vec<f32>) -> Self {
        Self { nx, ny, nz, rho, velocity }
    }

    /// Tri-linearly interpolates density at coordinate (x, y, z) in grid units.
    pub fn sample_rho(&self, x: f32, y: f32, z: f32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let z0 = z.floor() as i32;
        
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;
        
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;
        let dz = z - z0 as f32;
        
        let c000 = self.get_rho(x0, y0, z0);
        let c100 = self.get_rho(x1, y0, z0);
        let c010 = self.get_rho(x0, y1, z0);
        let c110 = self.get_rho(x1, y1, z0);
        let c001 = self.get_rho(x0, y0, z1);
        let c101 = self.get_rho(x1, y0, z1);
        let c011 = self.get_rho(x0, y1, z1);
        let c111 = self.get_rho(x1, y1, z1);
        
        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;
        
        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;
        
        c0 * (1.0 - dz) + c1 * dz
    }

    /// Tri-linearly interpolates velocity at coordinate (x, y, z) in grid units.
    pub fn sample_velocity(&self, x: f32, y: f32, z: f32) -> Vector3<f64> {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let z0 = z.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;
        let dz = z - z0 as f32;

        let v000 = self.get_vel(x0, y0, z0);
        let v100 = self.get_vel(x1, y0, z0);
        let v010 = self.get_vel(x0, y1, z0);
        let v110 = self.get_vel(x1, y1, z0);
        let v001 = self.get_vel(x0, y0, z1);
        let v101 = self.get_vel(x1, y0, z1);
        let v011 = self.get_vel(x0, y1, z1);
        let v111 = self.get_vel(x1, y1, z1);

        let v00 = v000 * (1.0 - dx) + v100 * dx;
        let v01 = v001 * (1.0 - dx) + v101 * dx;
        let v10 = v010 * (1.0 - dx) + v110 * dx;
        let v11 = v011 * (1.0 - dx) + v111 * dx;
        let v0 = v00 * (1.0 - dy) + v10 * dy;
        let v1 = v01 * (1.0 - dy) + v11 * dy;
        
        let res = v0 * (1.0 - dz) + v1 * dz;
        Vector3::new(res.x as f64, res.y as f64, res.z as f64)
    }

    fn get_rho(&self, x: i32, y: i32, z: i32) -> f32 {
        let x = x.rem_euclid(self.nx as i32) as usize;
        let y = y.rem_euclid(self.ny as i32) as usize;
        let z = z.rem_euclid(self.nz as i32) as usize;
        self.rho[x + self.nx * (y + self.ny * z)]
    }

    fn get_vel(&self, x: i32, y: i32, z: i32) -> Vector3<f32> {
        let x = x.rem_euclid(self.nx as i32) as usize;
        let y = y.rem_euclid(self.ny as i32) as usize;
        let z = z.rem_euclid(self.nz as i32) as usize;
        let idx = x + self.nx * (y + self.ny * z);
        let stride = self.nx * self.ny * self.nz;
        Vector3::new(
            self.velocity[idx],
            self.velocity[idx + stride],
            self.velocity[idx + 2 * stride],
        )
    }
}
