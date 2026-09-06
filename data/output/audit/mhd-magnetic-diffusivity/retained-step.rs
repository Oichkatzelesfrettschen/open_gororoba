fn retained_step(field: &mut MhdField, u: &[[f64; 3]]) {
        let nx = field.nx;
        let ny = field.ny;
        let nz = field.nz;
        let n = nx * ny * nz;
        let dt = field.config.dt_mhd;
        let eta = field.config.eta;

        debug_assert_eq!(u.len(), n, "velocity field length mismatch");

        // Compute v x B at each grid point
        let mut vxb_x = vec![0.0; n];
        let mut vxb_y = vec![0.0; n];
        let mut vxb_z = vec![0.0; n];

        for idx in 0..n {
            let [ux, uy, uz] = u[idx];
            let bx_i = field.bx[idx];
            let by_i = field.by[idx];
            let bz_i = field.bz[idx];
            // v x B = (uy*Bz - uz*By, uz*Bx - ux*Bz, ux*By - uy*Bx)
            vxb_x[idx] = uy * bz_i - uz * by_i;
            vxb_y[idx] = uz * bx_i - ux * bz_i;
            vxb_z[idx] = ux * by_i - uy * bx_i;
        }

        // Compute curl(v x B) via central differences (periodic)
        let mut dbx = vec![0.0; n];
        let mut dby = vec![0.0; n];
        let mut dbz = vec![0.0; n];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;

                    // Periodic neighbors
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    // curl(F)_x = dFz/dy - dFy/dz
                    // curl(F)_y = dFx/dz - dFz/dx
                    // curl(F)_z = dFy/dx - dFx/dy
                    let curl_vxb_x = 0.5 * (vxb_z[yp] - vxb_z[ym]) - 0.5 * (vxb_y[zp] - vxb_y[zm]);
                    let curl_vxb_y = 0.5 * (vxb_x[zp] - vxb_x[zm]) - 0.5 * (vxb_z[xp] - vxb_z[xm]);
                    let curl_vxb_z = 0.5 * (vxb_y[xp] - vxb_y[xm]) - 0.5 * (vxb_x[yp] - vxb_x[ym]);

                    dbx[idx] = curl_vxb_x;
                    dby[idx] = curl_vxb_y;
                    dbz[idx] = curl_vxb_z;

                    // Resistive term: -eta * curl(curl(B)) = eta * Laplacian(B)
                    // (using vector identity: curl(curl(B)) = grad(div B) - Lap(B),
                    //  and div B = 0 ideally)
                    if eta > 0.0 {
                        let lap_bx = field.bx[xp]
                            + field.bx[xm]
                            + field.bx[yp]
                            + field.bx[ym]
                            + field.bx[zp]
                            + field.bx[zm]
                            - 6.0 * field.bx[idx];
                        let lap_by = field.by[xp]
                            + field.by[xm]
                            + field.by[yp]
                            + field.by[ym]
                            + field.by[zp]
                            + field.by[zm]
                            - 6.0 * field.by[idx];
                        let lap_bz = field.bz[xp]
                            + field.bz[xm]
                            + field.bz[yp]
                            + field.bz[ym]
                            + field.bz[zp]
                            + field.bz[zm]
                            - 6.0 * field.bz[idx];
                        dbx[idx] += eta * lap_bx;
                        dby[idx] += eta * lap_by;
                        dbz[idx] += eta * lap_bz;
                    }
                }
            }
        }

        // Euler forward step
        for idx in 0..n {
            field.bx[idx] += dt * dbx[idx];
            field.by[idx] += dt * dby[idx];
            field.bz[idx] += dt * dbz[idx];
        }

        // Divergence cleaning (Dedner hyperbolic)
        if field.config.cleaning_rate > 0.0 {
            let ch = field.config.cleaning_rate;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;
                        let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                        let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                        let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                        let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                        let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                        let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                        let div_b = 0.5 * (field.bx[xp] - field.bx[xm])
                            + 0.5 * (field.by[yp] - field.by[ym])
                            + 0.5 * (field.bz[zp] - field.bz[zm]);

                        field.psi[idx] = -ch * ch * div_b;
                    }
                }
            }
            // Apply correction: B -= dt * grad(psi)
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;
                        let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                        let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                        let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                        let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                        let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                        let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                        field.bx[idx] -= dt * 0.5 * (field.psi[xp] - field.psi[xm]);
                        field.by[idx] -= dt * 0.5 * (field.psi[yp] - field.psi[ym]);
                        field.bz[idx] -= dt * 0.5 * (field.psi[zp] - field.psi[zm]);
                    }
                }
            }
        }
    }
