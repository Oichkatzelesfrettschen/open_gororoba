#[cfg(test)]
use crate::construction::chingon::AlternativityViolationTensor;

#[cfg(test)]
#[test]
fn test_alternativity_violation_tensor_8d() {
    let avt = AlternativityViolationTensor::new(8);
    assert_eq!(avt.violations.len(), 0, "Octonions must be alternative");
}

#[cfg(test)]
#[test]
fn test_alternativity_violation_tensor_16d() {
    let avt = AlternativityViolationTensor::new(16);
    assert!(
        !avt.violations.is_empty(),
        "Sedenions must violate alternativity"
    );
}

#[cfg(test)]
#[test]
fn test_alternativity_violation_tensor_64d() {
    let avt = AlternativityViolationTensor::new(64);
    assert!(
        !avt.violations.is_empty(),
        "Chingons must violate alternativity"
    );
}

#[cfg(test)]
#[test]
fn test_pack_roundtrip_16d() {
    let avt = AlternativityViolationTensor::new(16);
    let packed = avt.pack_for_gpu();
    assert_eq!(packed.data.len(), avt.violations.len());
    assert_eq!(packed.index_bits, 4); // log2(16) = 4
    assert_eq!(packed.dim, 16);

    for (idx, &(i, j, _k, m, sign)) in avt.violations.iter().enumerate() {
        let (ui, uj, um, usign) = packed.unpack(idx);
        assert_eq!(ui, i, "i mismatch at violation {}", idx);
        assert_eq!(uj, j, "j mismatch at violation {}", idx);
        assert_eq!(um, m, "m mismatch at violation {}", idx);
        assert_eq!(usign, sign, "sign mismatch at violation {}", idx);
    }
}

#[cfg(test)]
#[test]
fn test_pack_roundtrip_64d() {
    let avt = AlternativityViolationTensor::new(64);
    let packed = avt.pack_for_gpu();
    assert_eq!(packed.data.len(), avt.violations.len());
    assert_eq!(packed.index_bits, 6); // log2(64) = 6
    assert_eq!(packed.violation_count, avt.violations.len() as u32);

    // Spot-check first 100 and last 100 violations
    let n = avt.violations.len();
    let check_indices: Vec<usize> = (0..100.min(n)).chain(n.saturating_sub(100)..n).collect();

    for idx in check_indices {
        let (i, j, _k, m, sign) = avt.violations[idx];
        let (ui, uj, um, usign) = packed.unpack(idx);
        assert_eq!(
            (ui, uj, um, usign),
            (i, j, m, sign),
            "roundtrip mismatch at violation {}",
            idx
        );
    }
}

#[cfg(test)]
#[test]
fn test_pack_sign_values_are_pm2() {
    // Verify all sign values in 64D AVT are exactly +2 or -2
    let avt = AlternativityViolationTensor::new(64);
    for &(_, _, _, _, sign) in &avt.violations {
        assert!(
            sign == 2 || sign == -2,
            "unexpected sign value: {} (expected +/-2)",
            sign
        );
    }
}

#[cfg(test)]
#[test]
fn test_index_bits_dimensions() {
    use crate::construction::chingon::index_bits_for_dim;
    assert_eq!(index_bits_for_dim(16), 4);
    assert_eq!(index_bits_for_dim(32), 5);
    assert_eq!(index_bits_for_dim(64), 6);
    assert_eq!(index_bits_for_dim(128), 7);
    assert_eq!(index_bits_for_dim(256), 8);
    assert_eq!(index_bits_for_dim(512), 9);
    assert_eq!(index_bits_for_dim(1024), 10);
}

#[cfg(test)]
#[test]
fn test_avt_128d() {
    let t = std::time::Instant::now();
    let avt = AlternativityViolationTensor::new(128);
    let dt = t.elapsed();
    println!("\n=== 128D AVT (uncapped) ===");
    println!("  violations: {}", avt.violations.len());
    println!("  time: {:.3}s", dt.as_secs_f64());
    println!(
        "  memory: ~{:.1} MB",
        avt.violations.len() as f64 * 40.0 / 1e6
    );
    // 128D must have violations (non-alternative for dim >= 16)
    assert!(
        avt.violations.len() > 50_000,
        "128D should have many violations"
    );
    // Pack roundtrip
    let packed = avt.pack_for_gpu();
    assert_eq!(packed.index_bits, 7);
    assert_eq!(packed.dim, 128);
    let (ui, uj, um, usign) = packed.unpack(0);
    let (i, j, _k, m, sign) = avt.violations[0];
    assert_eq!((ui, uj, um, usign), (i, j, m, sign));
}

#[cfg(test)]
#[test]
fn test_avt_256d_capped() {
    let cap = 10_000_000;
    let t = std::time::Instant::now();
    let avt = AlternativityViolationTensor::new_with_cap(256, cap);
    let dt = t.elapsed();
    println!("\n=== 256D AVT (10M cap) ===");
    println!("  violations: {}", avt.violations.len());
    println!("  time: {:.3}s", dt.as_secs_f64());
    println!(
        "  memory: ~{:.1} MB",
        avt.violations.len() as f64 * 40.0 / 1e6
    );
    println!("  hit cap: {}", avt.violations.len() >= cap);
    assert!(
        avt.violations.len() > 50_000,
        "256D should have many violations"
    );
    // Pack roundtrip
    let packed = avt.pack_for_gpu();
    assert_eq!(packed.index_bits, 8);
    assert_eq!(packed.dim, 256);
    println!(
        "  packed SSBO: {:.1} MB",
        packed.data.len() as f64 * 4.0 / 1e6
    );
}

#[cfg(test)]
#[test]
fn test_avt_scaling_survey() {
    println!("\n=== AVT Dimensional Scaling Survey ===");
    println!(
        "{:>5} {:>12} {:>10} {:>12} {:>10}",
        "dim", "violations", "time_ms", "ratio", "v/d^3"
    );
    let mut prev = 0usize;
    for &dim in &[8, 16, 32, 64] {
        let t = std::time::Instant::now();
        let avt = AlternativityViolationTensor::new(dim);
        let dt = t.elapsed().as_millis();
        let v = avt.violations.len();
        let ratio = if prev > 0 {
            format!("{:.2}x", v as f64 / prev as f64)
        } else {
            "-".into()
        };
        let density = v as f64 / (dim as f64).powi(3);
        println!(
            "{:>5} {:>12} {:>10} {:>12} {:>10.6}",
            dim, v, dt, ratio, density
        );
        prev = v;
    }
    // 128D and 256D from profiling tests above
    println!(
        "{:>5} {:>12} {:>10} {:>12} {:>10.6}",
        128,
        468720,
        "~250",
        format!("{:.2}x", 468720.0 / prev as f64),
        468720.0 / 128.0f64.powi(3)
    );
    println!(
        "{:>5} {:>12} {:>10} {:>12} {:>10.6}",
        256,
        3968496,
        "~2100",
        format!("{:.2}x", 3968496.0 / 468720.0),
        3968496.0 / 256.0f64.powi(3)
    );
}
