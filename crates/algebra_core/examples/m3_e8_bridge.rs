//! M3-E8 Bridge: Testing the Trilinear M3 Operator on E8 Lattice Roots.
//!
//! This executable tests the hypothesis that the M3 trilinear operation,
//! when restricted to Octonionic basis elements, preserves the E8 root lattice
//! structure or reveals specific "scalar" (Fano) vs "vector" (Non-Fano) modes.

use algebra_core::{generate_e8_roots, E8Root};
use algebra_core::{oct_conjugate, oct_multiply, oct_norm_sq, Octonion};

fn main() {
    println!("=== M3-E8 Bridge: Algebraic Topology Probe ===");

    // 1. Generate E8 Roots
    let roots = generate_e8_roots();
    println!("Generated {} E8 roots.", roots.len());

    // 2. Test Bianchi Identity on Random Triples
    println!("\n--- Testing Bianchi Identity: M3(x,y,z) + M3(y,z,x) + M3(z,x,y) ---");
    let mut bianchi_failures = 0;
    let mut antisymmetric_failures = 0;
    let iterations = 1000;

    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    for _ in 0..iterations {
        let u_root = roots.choose(&mut rng).unwrap();
        let v_root = roots.choose(&mut rng).unwrap();
        let w_root = roots.choose(&mut rng).unwrap();

        let u = root_to_octonion(u_root);
        let v = root_to_octonion(v_root);
        let w = root_to_octonion(w_root);

        // Compute M3 cyclic sum
        let m3_uvw = compute_m3(&u, &v, &w);
        let m3_vwu = compute_m3(&v, &w, &u);
        let m3_wuv = compute_m3(&w, &u, &v);

        let sum = add(&add(&m3_uvw, &m3_vwu), &m3_wuv);

        if oct_norm_sq(&sum) > 1e-9 {
            bianchi_failures += 1;
        }

        // Test Antisymmetry: M3(u,v,w) = -M3(v,u,w)
        let m3_vuw = compute_m3(&v, &u, &w);
        let sum_anti = add(&m3_uvw, &m3_vuw);
        if oct_norm_sq(&sum_anti) > 1e-9 {
            antisymmetric_failures += 1;
        }
    }

    println!(
        "Bianchi Identity Failures: {} / {}",
        bianchi_failures, iterations
    );
    println!(
        "Antisymmetry Failures: {} / {}",
        antisymmetric_failures, iterations
    );

    // 3. Test M3 Output on E8 Lattice
    println!("\n--- Testing E8 Lattice Closure ---");
    let mut on_lattice = 0;
    let mut off_lattice = 0;
    let mut zero_out = 0;

    for _ in 0..iterations {
        let u_root = roots.choose(&mut rng).unwrap();
        let v_root = roots.choose(&mut rng).unwrap();
        let w_root = roots.choose(&mut rng).unwrap();

        let u = root_to_octonion(u_root);
        let v = root_to_octonion(v_root);
        let w = root_to_octonion(w_root);

        let res = compute_m3(&u, &v, &w);
        let len_sq = oct_norm_sq(&res);

        if len_sq < 1e-9 {
            zero_out += 1;
            continue;
        }

        // Check if result is in E8 lattice (possibly scaled by 2)
        // M3 often introduces factor of 1/2 or 1/4.
        if is_in_e8_lattice(&res) {
            on_lattice += 1;
        } else {
            // Check scaled by 2
            let scaled = scale(&res, 2.0);
            if is_in_e8_lattice(&scaled) {
                on_lattice += 1;
            } else {
                off_lattice += 1;
            }
        }
    }

    println!("Zero outputs: {}", zero_out);
    println!("On E8 Lattice (direct or 2x): {}", on_lattice);
    println!("Off Lattice: {}", off_lattice);
}

// --- Helpers ---

fn root_to_octonion(r: &E8Root) -> Octonion {
    r.coords
}

fn add(a: &Octonion, b: &Octonion) -> Octonion {
    let mut out = [0.0; 8];
    for i in 0..8 {
        out[i] = a[i] + b[i];
    }
    out
}

fn sub(a: &Octonion, b: &Octonion) -> Octonion {
    let mut out = [0.0; 8];
    for i in 0..8 {
        out[i] = a[i] - b[i];
    }
    out
}

fn scale(a: &Octonion, s: f64) -> Octonion {
    let mut out = [0.0; 8];
    for i in 0..8 {
        out[i] = a[i] * s;
    }
    out
}

/// Sedenion-like operations for M3
/// Sedenion = (Octonion, Octonion)
#[derive(Clone, Debug)]
struct Sedenion {
    low: Octonion,
    high: Octonion,
}

fn s_mul(u: &Sedenion, v: &Sedenion) -> Sedenion {
    // (A,B)(C,D) = (AC - D*B, DA + BC*)

    let ac = oct_multiply(&u.low, &v.low);
    let d_conj = oct_conjugate(&v.high);
    let db = oct_multiply(&d_conj, &u.high); // D* B

    let low = sub(&ac, &db); // AC - D*B

    let da = oct_multiply(&v.high, &u.low);
    let c_conj = oct_conjugate(&v.low);
    let bc = oct_multiply(&u.high, &c_conj); // B C*

    let high = add(&da, &bc); // DA + BC*

    Sedenion { low, high }
}

fn h_map(s: &Sedenion) -> Sedenion {
    // h(a,b) = ((a-b)/2, -(a-b)/2)
    let diff = sub(&s.low, &s.high);
    let half_diff = scale(&diff, 0.5);
    let neg_half_diff = scale(&half_diff, -1.0);
    Sedenion {
        low: half_diff,
        high: neg_half_diff,
    }
}

fn p_map(s: &Sedenion) -> Octonion {
    // p(a,b) = (a+b)/2
    let sum = add(&s.low, &s.high);
    scale(&sum, 0.5)
}

fn compute_m3(x: &Octonion, y: &Octonion, z: &Octonion) -> Octonion {
    // Embed x -> (x, x)
    let big_x = Sedenion { low: *x, high: *x };
    let big_y = Sedenion { low: *y, high: *y };
    let big_z = Sedenion { low: *z, high: *z };

    // term1 = p(h(xy)z)
    let xy = s_mul(&big_x, &big_y);
    let h_xy = h_map(&xy);
    let h_xy_z = s_mul(&h_xy, &big_z);
    let term1 = p_map(&h_xy_z);

    // term2 = p(x h(yz))
    let yz = s_mul(&big_y, &big_z);
    let h_yz = h_map(&yz);
    let x_h_yz = s_mul(&big_x, &h_yz);
    let term2 = p_map(&x_h_yz);

    sub(&term1, &term2)
}

fn is_in_e8_lattice(o: &Octonion) -> bool {
    // E8 definition:
    // 1. All coords integer, sum even.
    // 2. All coords half-integer, sum even.

    let coords = o;

    let is_int = coords.iter().all(|x| (x.round() - x).abs() < 1e-5);

    if is_int {
        let sum: i64 = coords.iter().map(|x| x.round() as i64).sum();
        return sum % 2 == 0;
    }

    let is_half = coords
        .iter()
        .all(|x| ((x * 2.0).round() - x * 2.0).abs() < 1e-5 && (x.round() - x).abs() > 0.1);
    if is_half {
        let sum: f64 = coords.iter().sum();
        // sum of half-integers: must be integer with even sum?
        // Actually, if all are half-integers, sum is (N/2) or (N+1)/2 etc.
        // For E8, sum must be an even integer (if all are integers)
        // For half-integers: sum of 8 half-integers is an integer.
        // E8 condition for half-integers: sum is even.
        return (sum.round() as i64) % 2 == 0;
    }

    false
}
