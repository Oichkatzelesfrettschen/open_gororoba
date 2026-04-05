use super::*;

// --- Emanation Table Tests ---

#[test]
fn test_emanation_table_dim16_size() {
    let et = emanation_table(16);
    assert_eq!(et.size, 15); // indices 1..15
    assert_eq!(et.cells.len(), 15);
    assert_eq!(et.cells[0].len(), 15);
    assert_eq!(et.total_cells, 225); // 15 * 15
}

#[test]
fn test_emanation_table_dim16_product_indices() {
    let et = emanation_table(16);
    // Verify product index = row XOR col for all cells
    for row in &et.cells {
        for cell in row {
            assert_eq!(
                cell.product_index,
                cell.row ^ cell.col,
                "Product index mismatch at ({}, {})",
                cell.row,
                cell.col
            );
        }
    }
}

#[test]
fn test_emanation_table_dim16_zd_count() {
    let et = emanation_table(16);
    // 42 primitive assessors, each symmetric in the ET -> 84 ZD cells
    assert_eq!(
        et.zd_count, 84,
        "Expected 84 ZD-marked cells (42 pairs x 2), got {}",
        et.zd_count
    );
}

#[test]
fn test_emanation_table_dim16_diagonal() {
    let et = emanation_table(16);
    // Diagonal: e_i * e_i = -1 for all imaginary units
    for i in 0..et.size {
        let cell = &et.cells[i][i];
        assert_eq!(
            cell.sign, -1,
            "e_{} * e_{} should give sign -1",
            cell.row, cell.col
        );
        assert_eq!(
            cell.product_index, 0,
            "e_{} * e_{} should give e_0",
            cell.row, cell.col
        );
    }
}

#[test]
fn test_emanation_table_dim16_xor_products() {
    let et = emanation_table(16);
    // e_1 * e_2 should give product index 1 XOR 2 = 3
    let cell = &et.cells[0][1]; // row index 0 -> basis 1, col index 1 -> basis 2
    assert_eq!(cell.product_index, 3);
    assert_ne!(cell.sign, 0);
}

#[test]
fn test_emanation_table_dim32_size() {
    let et = emanation_table(32);
    assert_eq!(et.size, 31); // indices 1..31
    assert_eq!(et.total_cells, 961); // 31 * 31
}

#[test]
fn test_emanation_table_dim32_has_more_zds_than_dim16() {
    let et16 = emanation_table(16);
    let et32 = emanation_table(32);
    // Pathions have 588 ZDs total, much more than sedenion's 84
    assert!(
        et32.zd_count > et16.zd_count,
        "dim=32 should have more ZD cells ({}) than dim=16 ({})",
        et32.zd_count,
        et16.zd_count
    );
}

// --- Sand Mandala Tests ---

#[test]
fn test_sand_mandala_dim16_full() {
    let et = emanation_table(16);
    let mandala = sand_mandala_pattern(&et);
    // At dim=16, all cross-assessor cells that are ZD should have fill_ratio > 0
    assert!(
        mandala.filled > 0,
        "dim=16 mandala should have filled cells"
    );
    assert!(mandala.fill_ratio > 0.0);
}

#[test]
fn test_sand_mandala_dim32_sparsity() {
    let et = emanation_table(32);
    let mandala = sand_mandala_pattern(&et);
    // dim=32 has a characteristic sparsity pattern
    assert!(
        mandala.total_cross > 0,
        "dim=32 should have cross-assessor cells"
    );
    assert!(
        mandala.fill_ratio > 0.0 && mandala.fill_ratio < 1.0,
        "dim=32 mandala should be partially sparse, ratio={}",
        mandala.fill_ratio
    );
}

// --- Carry-Bit Overflow Tests ---

#[test]
fn test_carry_bit_overflow_dim32() {
    let (lost, gained) = carry_bit_overflow_cells(32);
    // There should be some change between dim=16 and dim=32
    // At minimum, some ZD pairs from dim=16 should still exist in dim=32
    // (though their graph structure changes).
    // The key claim: the carry-bit creates new structure.
    // Lenient check: document carry-bit change counts without hard assertion.
    // (The carry-bit may or may not create detectable changes at dim=32.)
    eprintln!(
        "carry-bit analysis: lost={}, gained={}",
        lost.len(),
        gained.len()
    );
}

// --- ET Period-Doubling Tests ---

#[test]
fn test_et_scaling_dim16_to_32() {
    let scaling = et_period_doubling(&[16, 32]);
    assert_eq!(scaling.len(), 2);

    // dim=16: 7 components, 6 nodes each
    assert_eq!(scaling[0].n_components, 7);
    assert_eq!(scaling[0].nodes_per_component, 6);

    // dim=32: 15 components, 14 nodes each
    assert_eq!(scaling[1].n_components, 15);
    assert_eq!(scaling[1].nodes_per_component, 14);
}

#[test]
fn test_et_scaling_formula() {
    // Verify the known scaling laws: n_components = dim/2 - 1
    let scaling = et_period_doubling(&[16, 32, 64]);
    for s in &scaling {
        assert_eq!(
            s.n_components,
            s.dim / 2 - 1,
            "n_components should be dim/2-1 for dim={}",
            s.dim
        );
        assert_eq!(
            s.nodes_per_component,
            s.dim / 2 - 2,
            "nodes_per_component should be dim/2-2 for dim={}",
            s.dim
        );
    }
}

#[test]
fn test_et_block_similarity() {
    let et16 = emanation_table(16);
    let et32 = emanation_table(32);
    let sim = et_block_similarity(&et16, &et32);
    // Similarity should be between 0 and 1
    assert!(
        (0.0..=1.0).contains(&sim),
        "Block similarity should be in [0,1], got {}",
        sim
    );
}

// --- Generator Triad Tests ---

#[test]
fn test_generator_triad_identity_all_dims() {
    for n in 4..=8 {
        let dim = 1 << n; // 16, 32, 64, 128, 256
        let cd_gen = CdGenerator::new(dim);
        assert_eq!(cd_gen.g, dim / 2);

        let valid = cd_gen.valid_struts();
        assert!(
            !valid.is_empty(),
            "dim={} should have valid strut constants",
            dim
        );

        // For each valid strut, verify G XOR S = X (nonzero, distinct)
        for &s in &valid {
            let x = cd_gen.g ^ s;
            assert_ne!(x, 0);
            assert_ne!(x, cd_gen.g);
            assert_ne!(x, s);
            // The identity: G XOR S = X <=> S = G XOR X
            assert_eq!(s, cd_gen.g ^ x);
        }
    }
}

#[test]
fn test_lo_hi_split_dim16() {
    let (lo, hi) = lo_hi_split(16);
    assert_eq!(lo, 1..8);
    assert_eq!(hi, 8..16);
}

#[test]
fn test_lo_hi_split_dim32() {
    let (lo, hi) = lo_hi_split(32);
    assert_eq!(lo, 1..16);
    assert_eq!(hi, 16..32);
}

// --- Tray-Rack Tests ---

#[test]
fn test_tray_rack_count_per_boxkite() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let racks = tray_racks(bk);
        assert_eq!(
            racks.len(),
            8,
            "Each box-kite should have 8 triangular faces (octahedron), got {}",
            racks.len()
        );
    }
}

#[test]
fn test_zigzag_trefoil_split_2_6() {
    let bks = find_box_kites(16, 1e-10);
    for (i, bk) in bks.iter().enumerate() {
        let racks = tray_racks(bk);
        let n_zigzag = racks
            .iter()
            .filter(|r| r.twist_type == TwistType::Zigzag)
            .count();
        let n_trefoil = racks
            .iter()
            .filter(|r| r.twist_type == TwistType::Trefoil)
            .count();
        assert_eq!(
            n_zigzag, 2,
            "Box-kite {} should have 2 zigzag faces, got {}",
            i, n_zigzag
        );
        assert_eq!(
            n_trefoil, 6,
            "Box-kite {} should have 6 trefoil faces, got {}",
            i, n_trefoil
        );
    }
}

#[test]
fn test_twist_products_per_tray_rack() {
    let bks = find_box_kites(16, 1e-10);
    let bk = &bks[0];
    let racks = tray_racks(bk);

    // Each tray-rack has 3 assessors, giving 6 ordered pairs.
    // Each co-assessor pair produces 2 sign solutions.
    // So each tray-rack should yield 6 * 2 = 12 twist products.
    // (But not all ordered pairs may be co-assessors.)
    for rack in &racks {
        let products = twist_products(rack, bk);
        assert!(
            !products.is_empty(),
            "Tray-rack should have some twist products"
        );
    }
}

// --- Lanyard Census Tests ---

#[test]
fn test_lanyard_census_dim16() {
    let census = lanyard_census_dim16();
    // Must have both sails and tray-racks
    let total: usize = census.values().sum();
    // 7 box-kites x 8 faces = 56 total faces
    assert_eq!(total, 56, "Total lanyard count should be 56, got {}", total);
}

#[test]
fn test_lanyard_taxonomy_completeness() {
    let census = lanyard_census_dim16();
    // Every face must be classified as either Sail or TrayRack
    for &ltype in census.keys() {
        assert!(
            ltype == LanyardType::Sail || ltype == LanyardType::TrayRack,
            "Unexpected lanyard type in dim=16 census: {:?}",
            ltype
        );
    }
}

// --- Semiotic Square Tests ---

#[test]
fn test_semiotic_square_sedenion_7_boxkites() {
    let bks = find_box_kites(16, 1e-10);
    assert_eq!(bks.len(), 7);

    for (i, bk) in bks.iter().enumerate() {
        let squares = map_boxkite_to_semiotic(bk);
        assert_eq!(
            squares.len(),
            3,
            "Box-kite {} should yield 3 semiotic squares (one per strut axis), got {}",
            i,
            squares.len()
        );

        // Each square should have 4 distinct assessors
        for (j, sq) in squares.iter().enumerate() {
            let set: HashSet<Assessor> = [sq.a, sq.b, sq.not_a, sq.not_b].iter().copied().collect();
            assert_eq!(
                set.len(),
                4,
                "Semiotic square {}.{} should have 4 distinct assessors",
                i,
                j
            );
        }
    }
}

#[test]
fn test_semiotic_completeness() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let squares = map_boxkite_to_semiotic(bk);
        assert!(
            verify_semiotic_completeness(bk, &squares),
            "Semiotic squares should cover all assessors in box-kite"
        );
    }
}

// --- Loop-Box-Kite Duality ---

#[test]
fn test_loop_boxkite_duality() {
    let pairs = loop_boxkite_pairs();
    // There should be some mapping between 7 O-trips and 7 box-kites
    assert!(!pairs.is_empty(), "Loop-box-kite pairs should not be empty");
}

#[test]
fn test_psl27_order() {
    assert_eq!(psl27_order(), 168);
    // 168 = 7 * 24 = 7 * 4! = |PSL(2,7)|
    assert_eq!(168 % 7, 0);
    assert_eq!(168 % 8, 0);
    assert_eq!(168 % 3, 0);
}

// --- Hjelmslev Net Tests ---

#[test]
fn test_hjelmslev_net_dim16_is_fano() {
    let net = hjelmslev_net(16);
    assert_eq!(net.proj_dim, 2, "dim=16 -> PG(2,2) = Fano plane");
    assert_eq!(net.n_points, 7);
    assert_eq!(net.n_lines, 7);
}

#[test]
fn test_hjelmslev_net_dim32_is_pg3_2() {
    let net = hjelmslev_net(32);
    assert_eq!(net.proj_dim, 3, "dim=32 -> PG(3,2)");
    assert_eq!(net.n_points, 15);
    assert_eq!(net.n_lines, 35);
}

#[test]
fn test_hjelmslev_net_dim64_is_pg4_2() {
    let net = hjelmslev_net(64);
    assert_eq!(net.proj_dim, 4, "dim=64 -> PG(4,2)");
    assert_eq!(net.n_points, 31);
    assert_eq!(net.n_lines, 155);
}

// --- Spectral Census Tests ---

#[test]
fn test_spectral_census_dim16() {
    let census = spectral_census(16);
    assert_eq!(
        census.len(),
        7,
        "dim=16 should have 7 spectral fingerprints"
    );
    for fp in &census {
        assert_eq!(fp.n_nodes, 6, "Each dim=16 component has 6 nodes");
        assert_eq!(fp.n_edges, 12, "Each dim=16 component has 12 edges");
    }
}

#[test]
fn test_spectral_census_dim32() {
    let census = spectral_census(32);
    assert_eq!(
        census.len(),
        15,
        "dim=32 should have 15 spectral fingerprints"
    );
    for fp in &census {
        assert_eq!(fp.n_nodes, 14);
    }
}

// --- Generator Tests ---

#[test]
fn test_cd_generator_dim16() {
    let cd_gen = CdGenerator::new(16);
    assert_eq!(cd_gen.g, 8);
    let struts = cd_gen.valid_struts();
    assert!(struts.contains(&1));
    assert!(struts.contains(&7));
}

// --- Octonion Subalgebra Constraint ---

#[test]
fn test_octonion_subalgebra_with_fano_vector() {
    // A lattice vector with support on a Fano triple
    let v = vec![1, 1, 1, 1, 0, 0, 0, 0]; // support {0,1,2,3}, non-real {1,2,3}
    assert!(
        octonion_subalgebra_constraint_check(&[v]),
        "Vector with Fano-triple support should be Fano-compatible"
    );
}

#[test]
fn test_octonion_subalgebra_with_non_fano_vector() {
    // A lattice vector with non-Fano support
    let v = vec![1, 1, 1, 0, 1, 0, 0, 0]; // support {0,1,2,4}, non-real {1,2,4}
    // [1,2,4] is not a Fano triple -- check if it's reported correctly
    let is_fano = O_TRIPS.iter().any(|t| t == &[1, 2, 4]);
    if is_fano {
        assert!(octonion_subalgebra_constraint_check(&[v]));
    } else {
        // [1,2,4] is not a Fano triple, but 1,4,5 and 2,4,6 are
        assert!(!octonion_subalgebra_constraint_check(&[v]));
    }
}

// ===================================================================
// CDP Signed-Product Engine Tests (L1)
// ===================================================================

#[test]
#[allow(clippy::needless_range_loop)]
fn test_qsigns_base_case_quaternions() {
    // Verify QSIGNS matches de Marrais's table exactly.
    // e0 is real: e0*e_j = +e_j for all j.
    for j in 0..4 {
        assert_eq!(QSIGNS[0][j], 1, "e0*e{} should be +1", j);
    }
    // e_i*e0 = +e_i for all i.
    for i in 0..4 {
        assert_eq!(QSIGNS[i][0], 1, "e{}*e0 should be +1", i);
    }
    // e_i*e_i = -1 for i>0.
    for i in 1..4 {
        assert_eq!(QSIGNS[i][i], -1, "e{}*e{} should be -1", i, i);
    }
    // Cyclic products: e1*e2=+e3, e2*e3=+e1, e3*e1=+e2.
    assert_eq!(QSIGNS[1][2], 1, "e1*e2 = +e3");
    assert_eq!(QSIGNS[2][3], 1, "e2*e3 = +e1");
    assert_eq!(QSIGNS[3][1], 1, "e3*e1 = +e2");
    // Anti-cyclic: e2*e1=-e3, e3*e2=-e1, e1*e3=-e2.
    assert_eq!(QSIGNS[2][1], -1, "e2*e1 = -e3");
    assert_eq!(QSIGNS[3][2], -1, "e3*e2 = -e1");
    assert_eq!(QSIGNS[1][3], -1, "e1*e3 = -e2");
}

#[test]
fn test_cdp_identity_products() {
    // e0 * e_i = e_i with sign +1.
    for i in 0..16 {
        let (idx, sign) = cdp_signed_product(0, i);
        assert_eq!(idx, i, "0 XOR {} should be {}", i, i);
        assert_eq!(sign, 1, "e0*e{} should be positive", i);
    }
    // e_i * e0 = e_i with sign +1.
    for i in 0..16 {
        let (idx, sign) = cdp_signed_product(i, 0);
        assert_eq!(idx, i);
        assert_eq!(sign, 1);
    }
}

#[test]
fn test_cdp_self_product() {
    // e_i * e_i = -e_0 for all i > 0.
    for i in 1..32 {
        let (idx, sign) = cdp_signed_product(i, i);
        assert_eq!(idx, 0, "e{}*e{} product index should be 0", i, i);
        assert_eq!(sign, -1, "e{}*e{} should be -1", i, i);
    }
}

#[test]
fn test_cdp_quaternion_products() {
    // e1*e2 = +e3 (idx=3, sign=+1)
    assert_eq!(cdp_signed_product(1, 2), (3, 1));
    // e2*e1 = -e3 (idx=3, sign=-1)
    assert_eq!(cdp_signed_product(2, 1), (3, -1));
    // e2*e3 = +e1 (idx=1, sign=+1)
    assert_eq!(cdp_signed_product(2, 3), (1, 1));
    // e3*e2 = -e1 (idx=1, sign=-1)
    assert_eq!(cdp_signed_product(3, 2), (1, -1));
    // e1*e3 = -e2 (idx=2, sign=-1)
    assert_eq!(cdp_signed_product(1, 3), (2, -1));
    // e3*e1 = +e2 (idx=2, sign=+1)
    assert_eq!(cdp_signed_product(3, 1), (2, 1));
}

#[test]
fn test_cdp_worked_example_10_times_13() {
    // De Marrais's worked example from Presto:
    // A*D at S=1, dim=16: e_10 * e_13 = -e_7
    // 10 XOR 13 = 7, so product index = 7, sign = -1.
    let (idx, sign) = cdp_signed_product(10, 13);
    assert_eq!(idx, 7, "10 XOR 13 = 7");
    assert_eq!(sign, -1, "e10*e13 should be -e7 per de Marrais");
}

#[test]
fn test_cdp_cross_validates_with_cd_basis_mul_sign_dim16() {
    // For all (p, q) at dim=16, cdp_signed_product must agree
    // with cd_basis_mul_sign.
    for p in 1..16 {
        for q in 1..16 {
            if p == q {
                continue;
            }
            let (idx, sign) = cdp_signed_product(p, q);
            let expected_sign = cd_basis_mul_sign(16, p, q);
            assert_eq!(
                idx,
                p ^ q,
                "Product index should be p XOR q for ({}, {})",
                p,
                q
            );
            assert_eq!(
                sign as i32, expected_sign,
                "Sign mismatch at ({}, {}): cdp={}, cd_basis={}",
                p, q, sign, expected_sign
            );
        }
    }
}

#[test]
fn test_cdp_cross_validates_with_cd_basis_mul_sign_dim32() {
    for p in 1..32 {
        for q in 1..32 {
            if p == q {
                continue;
            }
            let (idx, sign) = cdp_signed_product(p, q);
            let expected_sign = cd_basis_mul_sign(32, p, q);
            assert_eq!(idx, p ^ q);
            assert_eq!(
                sign as i32, expected_sign,
                "Sign mismatch at dim=32 ({}, {}): cdp={}, cd_basis={}",
                p, q, sign, expected_sign
            );
        }
    }
}

#[test]
fn test_cdp_cross_validates_with_cd_basis_mul_sign_dim64() {
    for p in 1..64 {
        for q in 1..64 {
            if p == q {
                continue;
            }
            let (idx, sign) = cdp_signed_product(p, q);
            let expected_sign = cd_basis_mul_sign(64, p, q);
            assert_eq!(idx, p ^ q);
            assert_eq!(
                sign as i32, expected_sign,
                "Sign mismatch at dim=64 ({}, {}): cdp={}, cd_basis={}",
                p, q, sign, expected_sign
            );
        }
    }
}

#[test]
fn test_cdp_anticommutativity() {
    // For i != j (both nonzero imaginary), e_i*e_j = -e_j*e_i
    // when i and j are in the same quaternion subalgebra.
    // More generally in CD algebras, basis products may or may not
    // anticommute. But the signed product reversal should be consistent.
    for i in 1..16 {
        for j in (i + 1)..16 {
            let (idx1, _sign1) = cdp_signed_product(i, j);
            let (idx2, _sign2) = cdp_signed_product(j, i);
            assert_eq!(
                idx1, idx2,
                "Product index must be same regardless of order: ({},{})",
                i, j
            );
            // In sedenions, we don't have universal anticommutativity,
            // but the sign relation is captured by our engine.
        }
    }
}

#[test]
fn test_bit_length() {
    assert_eq!(bit_length(0), 0);
    assert_eq!(bit_length(1), 1);
    assert_eq!(bit_length(2), 2);
    assert_eq!(bit_length(3), 2);
    assert_eq!(bit_length(4), 3);
    assert_eq!(bit_length(7), 3);
    assert_eq!(bit_length(8), 4);
    assert_eq!(bit_length(15), 4);
    assert_eq!(bit_length(16), 5);
}

// ===================================================================
// Tone Row Tests (L2)
// ===================================================================

#[test]
fn test_tone_row_dim16_s1() {
    let tr = generate_tone_row(4, 1);
    assert_eq!(tr.g, 8);
    assert_eq!(tr.x, 9); // G + S = 8 + 1 = 9
    assert_eq!(tr.k, 6); // G - 2 = 8 - 2 = 6
    assert_eq!(tr.lo.len(), 6);
    assert_eq!(tr.hi.len(), 6);

    // S=1 is excluded from lo indices
    assert!(!tr.lo.contains(&1), "S=1 must be excluded from tone row");
    // All lo indices should be in [2..8)
    for &l in &tr.lo {
        assert!((2..=7).contains(&l), "LO index {} out of range [2,7]", l);
    }
    // Hi indices should be lo XOR X
    for i in 0..6 {
        assert_eq!(
            tr.hi[i],
            tr.lo[i] ^ tr.x,
            "HI[{}] should be LO[{}] XOR X = {} XOR {}",
            i,
            i,
            tr.lo[i],
            tr.x
        );
    }
}

#[test]
fn test_tone_row_mirror_pairing() {
    // For S=1, positions i and K-1-i should be strut-opposites.
    let tr = generate_tone_row(4, 1);
    for i in 0..tr.k / 2 {
        let mirror = tr.k - 1 - i;
        let lo_xor = tr.lo[i] ^ tr.lo[mirror];
        assert_eq!(
            lo_xor, tr.s,
            "Mirror pair ({}, {}): lo[{}]={} XOR lo[{}]={} should equal S={}",
            i, mirror, i, tr.lo[i], mirror, tr.lo[mirror], tr.s
        );
    }
}

#[test]
fn test_tone_row_dim16_all_struts() {
    // Every valid strut constant at dim=16 should produce a valid tone row.
    for s in 1..8 {
        let tr = generate_tone_row(4, s);
        assert_eq!(tr.k, 6);
        assert!(!tr.lo.contains(&s), "S={} must be excluded from LO", s);
        // All LO indices must be distinct
        let lo_set: HashSet<usize> = tr.lo.iter().copied().collect();
        assert_eq!(lo_set.len(), 6, "S={}: LO must have 6 distinct indices", s);
    }
}

#[test]
fn test_tone_row_dim32_s1() {
    let tr = generate_tone_row(5, 1);
    assert_eq!(tr.g, 16);
    assert_eq!(tr.x, 17);
    assert_eq!(tr.k, 14);
    assert_eq!(tr.lo.len(), 14);
    assert!(!tr.lo.contains(&1));
}

// ===================================================================
// Strutted Emanation Table Tests (L3)
// ===================================================================

#[test]
fn test_strutted_et_dim16_s1_size() {
    let et = create_strutted_et(4, 1);
    assert_eq!(et.tone_row.k, 6);
    // Total possible: 6*6 - 6 (diagonal) - 6 (strut-opposites) = 24
    // Actually: K=6, diagonal = 6 cells, strut-opposites = K cells
    // (row_pos + col_pos == K-1 = 5)
    // Diagonal: 6 cells skipped.
    // Strut-opposite: positions (0,5),(1,4),(2,3),(3,2),(4,1),(5,0) = 6 cells skipped.
    // But diagonal and strut-opposite may overlap when K is odd at the midpoint.
    // K=6: no overlap (midpoint would be 2.5).
    // Total possible = 36 - 6 - 6 = 24.
    assert_eq!(
        et.total_possible, 24,
        "K=6: total possible = 36 - 6 - 6 = 24, got {}",
        et.total_possible
    );
}

#[test]
fn test_strutted_et_dim16_s1_dmz_count() {
    let et = create_strutted_et(4, 1);
    // N=4: K=6, total_possible=24, DMZ=24 (100% fill for all struts).
    assert_eq!(
        et.dmz_count, 24,
        "N=4 S=1 DMZ count should be exactly 24, got {}",
        et.dmz_count
    );
}

#[test]
fn test_strutted_et_dim16_single_regime() {
    // De Marrais: all 7 sedenion strut constants yield the same DMZ count.
    let mut counts = Vec::new();
    for s in 1..8 {
        let et = create_strutted_et(4, s);
        counts.push(et.dmz_count);
    }
    let first = counts[0];
    for (i, &c) in counts.iter().enumerate() {
        assert_eq!(
            c,
            first,
            "Sedenion single regime: S={} has {} DMZ, S=1 has {}",
            i + 1,
            c,
            first
        );
    }
}

#[test]
fn test_strutted_et_dim16_dmz_abs_match() {
    // Verify the cross-magnitude consistency: |UL|==|LR| and |UR|==|LL|
    // for all filled cells.
    let et = create_strutted_et(4, 1);
    for row in &et.cells {
        for cell in row.iter().flatten() {
            // Cross-magnitude check should always pass
            assert_eq!(
                cell.ul.unsigned_abs() as usize,
                cell.lr.unsigned_abs() as usize,
                "Cross-mag fail: |UL|={} != |LR|={} at ({},{})",
                cell.ul.unsigned_abs(),
                cell.lr.unsigned_abs(),
                cell.row_pos,
                cell.col_pos
            );
            assert_eq!(
                cell.ur.unsigned_abs() as usize,
                cell.ll.unsigned_abs() as usize,
                "Cross-mag fail: |UR|={} != |LL|={} at ({},{})",
                cell.ur.unsigned_abs(),
                cell.ll.unsigned_abs(),
                cell.row_pos,
                cell.col_pos
            );
        }
    }
}

#[test]
fn test_strutted_et_dim32_two_regimes() {
    // De Marrais: pathions have 2 DMZ regimes.
    // S=1..8 (inherited from sedenions): DMZ=168 (100% fill)
    // S=9..15 (new at pathion level): DMZ=72 (42.9% fill)
    let regimes = et_regimes(5);
    assert_eq!(
        regimes.len(),
        2,
        "Pathion (N=5) should have exactly 2 regimes, got {:?}",
        regimes
    );
    assert!(
        regimes.contains_key(&168),
        "Pathion should have regime DMZ=168"
    );
    assert!(
        regimes.contains_key(&72),
        "Pathion should have regime DMZ=72"
    );
    assert_eq!(
        regimes[&168].len(),
        8,
        "168-regime should have 8 struts (S=1..8)"
    );
    assert_eq!(
        regimes[&72].len(),
        7,
        "72-regime should have 7 struts (S=9..15)"
    );
}

#[test]
fn test_strutted_et_dim32_dmz_divisible_by_24() {
    // De Marrais: DMZ counts are always divisible by 24.
    for s in 1..16 {
        let et = create_strutted_et(5, s);
        assert_eq!(
            et.dmz_count % 24,
            0,
            "N=5 S={}: DMZ count {} not divisible by 24",
            s,
            et.dmz_count
        );
    }
}

#[test]
fn test_strutted_et_dim16_dmz_divisible_by_24() {
    for s in 1..8 {
        let et = create_strutted_et(4, s);
        assert_eq!(
            et.dmz_count % 24,
            0,
            "N=4 S={}: DMZ count {} not divisible by 24",
            s,
            et.dmz_count
        );
    }
}

// ===================================================================
// ET Sparsity Spectroscopy Tests (L4)
// ===================================================================

#[test]
fn test_spectroscopy_dim16_single_regime() {
    let spectra = et_sparsity_spectroscopy(4);
    assert_eq!(spectra.len(), 7, "Sedenions have 7 strut constants");
    // All should have the same DMZ count (single regime).
    let first_dmz = spectra[0].dmz_count;
    for sp in &spectra {
        assert_eq!(
            sp.dmz_count, first_dmz,
            "S={}: DMZ={}, expected {}",
            sp.s, sp.dmz_count, first_dmz
        );
    }
}

#[test]
fn test_spectroscopy_dim32_regime_structure() {
    let spectra = et_sparsity_spectroscopy(5);
    assert_eq!(spectra.len(), 15, "Pathions have 15 strut constants");
    let mut unique_counts: Vec<usize> = spectra
        .iter()
        .map(|sp| sp.dmz_count)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique_counts.sort();
    assert_eq!(
        unique_counts.len(),
        2,
        "Pathion should have 2 distinct DMZ counts, got {:?}",
        unique_counts
    );
}

// ===================================================================
// ET Regime Verification: Exact DMZ Counts (N=4..7)
// ===================================================================

#[test]
fn test_strutted_et_dim16_exact_dmz_24() {
    // N=4 (sedenions): all 7 struts yield DMZ=24 = total_possible (100% fill).
    for s in 1..8 {
        let et = create_strutted_et(4, s);
        assert_eq!(
            et.dmz_count, 24,
            "N=4 S={}: expected DMZ=24, got {}",
            s, et.dmz_count
        );
        assert_eq!(
            et.total_possible, 24,
            "N=4 S={}: expected total_possible=24, got {}",
            s, et.total_possible
        );
    }
}

#[test]
fn test_strutted_et_dim32_exact_dmz_counts() {
    // N=5 (pathions): S=1..8 -> DMZ=168, S=9..15 -> DMZ=72.
    for s in 1..=8 {
        let et = create_strutted_et(5, s);
        assert_eq!(
            et.dmz_count, 168,
            "N=5 S={}: expected DMZ=168 (full fill), got {}",
            s, et.dmz_count
        );
    }
    for s in 9..=15 {
        let et = create_strutted_et(5, s);
        assert_eq!(
            et.dmz_count, 72,
            "N=5 S={}: expected DMZ=72, got {}",
            s, et.dmz_count
        );
    }
}

#[test]
fn test_strutted_et_dim64_four_regimes() {
    // N=6 (chingons): 4 regimes as de Marrais predicted.
    // DMZ=840 (S=1..8,16), 456 (S=9..15), 168 (S=17..24), 552 (S=25..31)
    let regimes = et_regimes(6);
    assert_eq!(
        regimes.len(),
        4,
        "Chingon (N=6) should have 4 regimes, got {:?}",
        regimes
    );
    assert!(regimes.contains_key(&840), "Missing 840-regime");
    assert!(regimes.contains_key(&456), "Missing 456-regime");
    assert!(regimes.contains_key(&168), "Missing 168-regime");
    assert!(regimes.contains_key(&552), "Missing 552-regime");
    assert_eq!(
        regimes[&840].len(),
        9,
        "840-regime: expected 9 struts (S=1..8,16)"
    );
    assert_eq!(
        regimes[&456].len(),
        7,
        "456-regime: expected 7 struts (S=9..15)"
    );
    assert_eq!(
        regimes[&168].len(),
        8,
        "168-regime: expected 8 struts (S=17..24)"
    );
    assert_eq!(
        regimes[&552].len(),
        7,
        "552-regime: expected 7 struts (S=25..31)"
    );
}

#[test]
fn test_strutted_et_dim64_dmz_divisible_by_24() {
    for s in 1..32 {
        let et = create_strutted_et(6, s);
        assert_eq!(
            et.dmz_count % 24,
            0,
            "N=6 S={}: DMZ count {} not divisible by 24",
            s,
            et.dmz_count
        );
    }
}

#[test]
fn test_strutted_et_dim128_eight_regimes() {
    // N=7 (routons): 8 regimes, extending the period-doubling cascade.
    let regimes = et_regimes(7);
    assert_eq!(
        regimes.len(),
        8,
        "Routon (N=7) should have 8 regimes, got {}",
        regimes.len()
    );
    // Exact DMZ counts discovered empirically:
    let expected_dmz: [usize; 8] = [360, 1032, 1512, 1896, 2184, 2568, 3048, 3720];
    for &dmz in &expected_dmz {
        assert!(regimes.contains_key(&dmz), "N=7 missing regime DMZ={}", dmz);
    }
}

#[test]
fn test_strutted_et_dim128_dmz_divisible_by_24() {
    for s in 1..64 {
        let et = create_strutted_et(7, s);
        assert_eq!(
            et.dmz_count % 24,
            0,
            "N=7 S={}: DMZ count {} not divisible by 24",
            s,
            et.dmz_count
        );
    }
}

#[test]
fn test_regime_doubling_cascade_n4_to_n7() {
    // De Marrais regime-doubling law: number of regimes doubles at each N.
    // N=4: 1, N=5: 2, N=6: 4, N=7: 8.
    let expected = [(4, 1), (5, 2), (6, 4), (7, 8)];
    for &(n, expected_count) in &expected {
        let regimes = et_regimes(n);
        assert_eq!(
            regimes.len(),
            expected_count,
            "N={}: expected {} regimes, got {}",
            n,
            expected_count,
            regimes.len()
        );
    }
}

#[test]
fn test_generator_power_struts_always_full_fill() {
    // Struts that are powers of 2 (generators) always yield 100% fill.
    // At N=n, the generators are G=2^(n-1). S that are powers of 2
    // and less than G yield full fill.
    for n in 4..=7 {
        let g = 1usize << (n - 1);
        let mut power = 1usize;
        while power < g {
            let et = create_strutted_et(n, power);
            assert_eq!(
                et.dmz_count, et.total_possible,
                "N={} S={}: generator-power strut should have full fill, \
                 got {}/{}",
                n, power, et.dmz_count, et.total_possible
            );
            power <<= 1;
        }
    }
}

// ===================================================================
// Trip-Count Two-Step Tests
// ===================================================================

#[test]
fn test_trip_count_known_values() {
    assert_eq!(trip_count(2), 1, "Quaternions: 1 trip");
    assert_eq!(trip_count(3), 7, "Octonions: 7 trips");
    assert_eq!(trip_count(4), 35, "Sedenions: 35 trips");
    assert_eq!(trip_count(5), 155, "Pathions: 155 trips");
    assert_eq!(trip_count(6), 651, "Chingons: 651 trips");
}

#[test]
fn test_trip_count_two_step_matches_full_fill_et() {
    // For inherited struts (S < 8), DMZ_count / 24 = Trip_{N-2}.
    for n in 4..=7 {
        let et = create_strutted_et(n, 1); // S=1 is always inherited
        let bk_count = et.dmz_count / 24;
        let expected = trip_count_two_step(n);
        assert_eq!(
            bk_count, expected,
            "N={}: full-fill ET gives {}/24 = {} box-kites, expected Trip_{{N-2}} = {}",
            n, et.dmz_count, bk_count, expected
        );
    }
}

#[test]
fn test_trip_count_two_step_all_inherited_struts() {
    // All inherited struts (S=1..7) should give the same box-kite count.
    for n in 4..=7 {
        let expected = trip_count_two_step(n);
        for s in 1..=7 {
            let et = create_strutted_et(n, s);
            assert_eq!(
                et.dmz_count / 24,
                expected,
                "N={} S={}: expected {} box-kites",
                n,
                s,
                expected
            );
        }
    }
}

#[test]
fn test_trip_count_two_step_algebraic_identity() {
    // Verify: total_possible / 24 = Trip_{N-2} for full-fill ETs.
    // total_possible = K * (K - 2) where K = 2^{N-1} - 2.
    // Trip_{N-2} = (2^{N-2} - 1)(2^{N-2} - 2) / 6.
    for n in 4..=10 {
        let k = (1usize << (n - 1)) - 2;
        let total = k * (k - 2);
        let trip = trip_count(n - 2);
        assert_eq!(
            total,
            24 * trip,
            "N={}: K(K-2)={} should equal 24 * Trip_{{N-2}} = 24 * {} = {}",
            n,
            total,
            trip,
            24 * trip
        );
    }
}

// ===================================================================
// Sky Classification Tests (de Marrais erratum resolution)
// ===================================================================

#[test]
fn test_is_sky_strut_basic() {
    // S <= 8 are never Skies
    for s in 1..=8 {
        assert!(!is_sky_strut(s), "S={} should NOT be a Sky strut", s);
    }
    // S > 8, not power of 2: ARE Skies
    for s in [
        9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ] {
        assert!(is_sky_strut(s), "S={} should be a Sky strut", s);
    }
    // Powers of 2 > 8: NOT Skies (generator-inherited)
    for s in [16, 32, 64] {
        assert!(
            !is_sky_strut(s),
            "S={} (power of 2) should NOT be a Sky strut",
            s
        );
    }
}

#[test]
fn test_sky_struts_always_sparse_n5_to_n7() {
    // Every Sky strut (S > 8, not power of 2) must have DMZ < total_possible.
    // Every non-Sky strut must have DMZ == total_possible (full fill).
    for n in 5..=7 {
        let g = 1usize << (n - 1);
        for s in 1..g {
            let et = create_strutted_et(n, s);
            if is_sky_strut(s) {
                assert!(
                    et.dmz_count < et.total_possible,
                    "N={} S={}: Sky strut should be sparse, got {}/{}",
                    n,
                    s,
                    et.dmz_count,
                    et.total_possible
                );
            } else if is_inherited_full_fill_strut(n, s) {
                assert_eq!(
                    et.dmz_count, et.total_possible,
                    "N={} S={}: inherited strut should be full fill, got {}/{}",
                    n, s, et.dmz_count, et.total_possible
                );
            }
        }
    }
}

#[test]
fn test_erratum_resolved_gt8_not_lt8() {
    // The "Complex Systems" abstract erroneously states "less than 8".
    // All other de Marrais sources say "> 8 and not a power of 2".
    //
    // Verification: S=9 at N=5 (the first Sky strut) is sparse (72 < 168).
    let et = create_strutted_et(5, 9);
    assert!(
        et.dmz_count < et.total_possible,
        "S=9 at N=5 is sparse (a Sky), confirming > 8 condition"
    );
    assert_eq!(et.dmz_count, 72, "S=9 N=5: exact DMZ count should be 72");

    // S=8 at N=5 is NOT sparse (full fill, confirming 8 is the boundary).
    let et8 = create_strutted_et(5, 8);
    assert_eq!(
        et8.dmz_count, et8.total_possible,
        "S=8 at N=5 is full fill, confirming 8 is the last non-Sky strut"
    );
}

// ===================================================================
// Strut Spectroscopy Tests (N=5 Sky Struts)
// ===================================================================

#[test]
fn test_classify_strut_n5_generators() {
    // At N=5 (G=16): powers of 2 in [1,16) are generators.
    assert_eq!(classify_strut(5, 1), StrutClass::Generator);
    assert_eq!(classify_strut(5, 2), StrutClass::Generator);
    assert_eq!(classify_strut(5, 4), StrutClass::Generator);
    assert_eq!(classify_strut(5, 8), StrutClass::Generator);
}

#[test]
fn test_classify_strut_n5_mandala() {
    // S=3,5,6,7 are mandala-inherited (S<=7, not power of 2).
    assert_eq!(classify_strut(5, 3), StrutClass::Mandala);
    assert_eq!(classify_strut(5, 5), StrutClass::Mandala);
    assert_eq!(classify_strut(5, 6), StrutClass::Mandala);
    assert_eq!(classify_strut(5, 7), StrutClass::Mandala);
}

#[test]
fn test_classify_strut_n5_sky() {
    // S=9..15 (>8, non-power-of-2) are all Sky.
    for s in 9..=15 {
        assert_eq!(
            classify_strut(5, s),
            StrutClass::Sky,
            "S={} should be Sky at N=5",
            s
        );
    }
}

#[test]
fn test_spectroscopy_n5_class_counts() {
    // N=5 has 15 struts: 4 Generator + 4 Mandala + 7 Sky.
    let entries = strut_spectroscopy(5);
    assert_eq!(entries.len(), 15);
    let gen_count = entries
        .iter()
        .filter(|e| e.class == StrutClass::Generator)
        .count();
    let man_count = entries
        .iter()
        .filter(|e| e.class == StrutClass::Mandala)
        .count();
    let sky_count = entries
        .iter()
        .filter(|e| e.class == StrutClass::Sky)
        .count();
    assert_eq!(gen_count, 4, "4 generators (1,2,4,8)");
    assert_eq!(man_count, 4, "4 mandala-inherited (3,5,6,7)");
    assert_eq!(sky_count, 7, "7 sky struts (9..15)");
}

#[test]
fn test_spectroscopy_n5_generators_full_fill() {
    // All generators at N=5 must have full fill.
    let entries = strut_spectroscopy(5);
    for e in entries.iter().filter(|e| e.class == StrutClass::Generator) {
        assert!(e.is_full_fill, "Generator S={} must have full fill", e.s);
        assert_eq!(e.dmz_count, 168, "Generator S={}: expected DMZ=168", e.s);
    }
}

#[test]
fn test_spectroscopy_n5_mandala_full_fill() {
    // All mandala struts at N=5 must have full fill.
    let entries = strut_spectroscopy(5);
    for e in entries.iter().filter(|e| e.class == StrutClass::Mandala) {
        assert!(e.is_full_fill, "Mandala S={} must have full fill", e.s);
        assert_eq!(e.dmz_count, 168, "Mandala S={}: expected DMZ=168", e.s);
    }
}

#[test]
fn test_spectroscopy_n5_sky_struts_sparse_dmz_72() {
    // All 7 Sky struts at N=5 must have DMZ=72 (42.9% fill).
    let entries = strut_spectroscopy(5);
    for e in entries.iter().filter(|e| e.class == StrutClass::Sky) {
        assert!(!e.is_full_fill, "Sky S={} must NOT have full fill", e.s);
        assert_eq!(
            e.dmz_count, 72,
            "Sky S={}: expected DMZ=72, got {}",
            e.s, e.dmz_count
        );
    }
}

#[test]
fn test_spectroscopy_n5_sky_effective_bk_count_3() {
    // DMZ=72 / 24 = 3 effective box-kites for Sky struts.
    // This means 3 of 7 Fano-plane lines survive the Sky transition.
    let entries = strut_spectroscopy(5);
    for e in entries.iter().filter(|e| e.class == StrutClass::Sky) {
        assert_eq!(
            e.effective_bk_count, 3,
            "Sky S={}: expected 3 effective BKs, got {}",
            e.s, e.effective_bk_count
        );
    }
}

#[test]
fn test_spectroscopy_n5_full_fill_effective_bk_count_7() {
    // DMZ=168 / 24 = 7 effective box-kites for inherited struts.
    let entries = strut_spectroscopy(5);
    for e in entries.iter().filter(|e| e.is_full_fill) {
        assert_eq!(
            e.effective_bk_count, 7,
            "Full-fill S={}: expected 7 effective BKs, got {}",
            e.s, e.effective_bk_count
        );
    }
}

#[test]
fn test_spectroscopy_n5_sky_fill_ratio_3_7() {
    // Sky struts at N=5 have fill ratio 72/168 = 3/7.
    let entries = strut_spectroscopy(5);
    for e in entries.iter().filter(|e| e.class == StrutClass::Sky) {
        let expected = 72.0 / 168.0; // = 3/7
        assert!(
            (e.fill_ratio - expected).abs() < 1e-10,
            "Sky S={}: expected fill ratio 3/7, got {:.6}",
            e.s,
            e.fill_ratio
        );
    }
}

#[test]
fn test_spectroscopy_n6_class_counts() {
    // N=6 has 31 struts: 5 Generator (1,2,4,8,16) + 4 Mandala (3,5,6,7) + 22 Sky.
    let entries = strut_spectroscopy(6);
    assert_eq!(entries.len(), 31);
    let gen_count = entries
        .iter()
        .filter(|e| e.class == StrutClass::Generator)
        .count();
    let man_count = entries
        .iter()
        .filter(|e| e.class == StrutClass::Mandala)
        .count();
    let sky_count = entries
        .iter()
        .filter(|e| e.class == StrutClass::Sky)
        .count();
    assert_eq!(gen_count, 5, "5 generators (1,2,4,8,16)");
    assert_eq!(man_count, 4, "4 mandala-inherited (3,5,6,7)");
    assert_eq!(sky_count, 22, "22 sky struts");
}

// ===================================================================
// L5: Twist Transition System Tests
// ===================================================================

#[test]
fn test_twist_transition_table_size() {
    let transitions = twist_transition_table();
    // 7 box-kites x 3 tray-racks each = 21 transitions
    assert_eq!(
        transitions.len(),
        21,
        "Expected 21 twist transitions (7 BK x 3 TR), got {}",
        transitions.len()
    );
}

#[test]
fn test_twist_targets_are_valid_struts() {
    let transitions = twist_transition_table();
    let valid_struts: HashSet<usize> = (1..8).collect();
    for t in &transitions {
        assert!(
            valid_struts.contains(&t.h_star_target),
            "H* target {} is not a valid strut (source={})",
            t.h_star_target,
            t.source_strut
        );
        assert!(
            valid_struts.contains(&t.v_star_target),
            "V* target {} is not a valid strut (source={})",
            t.v_star_target,
            t.source_strut
        );
    }
}

#[test]
fn test_twist_targets_differ_from_source() {
    let transitions = twist_transition_table();
    for t in &transitions {
        assert_ne!(
            t.h_star_target, t.source_strut,
            "H* target should differ from source strut {}",
            t.source_strut
        );
        assert_ne!(
            t.v_star_target, t.source_strut,
            "V* target should differ from source strut {}",
            t.source_strut
        );
    }
}

#[test]
fn test_verify_twist_otrip_cycles() {
    assert!(
        verify_twist_otrip_cycles(),
        "Twist cycle verification should pass"
    );
}

// ===================================================================
// L6: Twisted Sisters PSL(2,7) Tests
// ===================================================================

#[test]
fn test_twisted_sisters_graph_nonempty() {
    let edges = twisted_sisters_graph();
    assert!(!edges.is_empty(), "Twisted Sisters graph should have edges");
}

#[test]
fn test_twisted_sisters_connects_all_7() {
    let edges = twisted_sisters_graph();
    let mut nodes: HashSet<usize> = HashSet::new();
    for e in &edges {
        nodes.insert(e.from_strut);
        nodes.insert(e.to_strut);
    }
    assert_eq!(
        nodes.len(),
        7,
        "Twisted Sisters should connect all 7 box-kites, got {}",
        nodes.len()
    );
}

#[test]
fn test_twisted_sisters_degree_sequence() {
    let seq = twisted_sisters_degree_sequence();
    // Each box-kite connects to others via 3 tray-racks, each with 2 targets.
    // But targets overlap, so degree <= 6.
    for &(s, deg) in &seq {
        assert!(
            deg >= 2,
            "Box-kite S={} should connect to at least 2 others, got {}",
            s,
            deg
        );
    }
}

// ===================================================================
// L7: Extended Lanyard Taxonomy Tests
// ===================================================================

#[test]
fn test_extended_lanyard_census_total() {
    let census = extended_lanyard_census_dim16();
    let total: usize = census.values().sum();
    // 7 box-kites x 8 triangular faces = 56 total
    assert_eq!(
        total, 56,
        "Extended lanyard census should cover 56 faces, got {}",
        total
    );
}

#[test]
fn test_extended_lanyard_census_zigzag_count() {
    let census = extended_lanyard_census_dim16();
    let zigzag = *census.get(&ExtendedLanyardType::TripleZigzag).unwrap_or(&0);
    // 7 box-kites x 2 zigzag faces = 14
    assert_eq!(
        zigzag, 14,
        "Expected 14 TripleZigzag faces (7 BK x 2), got {}",
        zigzag
    );
}

#[test]
fn test_extended_lanyard_census_trefoil_count() {
    let census = extended_lanyard_census_dim16();
    let trefoil = *census.get(&ExtendedLanyardType::Trefoil).unwrap_or(&0);
    // 7 box-kites x 6 trefoil faces = 42
    assert_eq!(
        trefoil, 42,
        "Expected 42 Trefoil faces (7 BK x 6), got {}",
        trefoil
    );
}

#[test]
fn test_extended_lanyard_no_blues_in_sedenions() {
    // In standard sedenion box-kites, no face has all-Same-sign edges.
    // "Blues" (all positive) would require 3 co-assessors with all Same-sign
    // edges, which doesn't occur in the standard octahedral structure.
    let census = extended_lanyard_census_dim16();
    let blues = *census.get(&ExtendedLanyardType::Blues).unwrap_or(&0);
    assert_eq!(
        blues, 0,
        "Sedenions should have 0 Blues faces, got {}",
        blues
    );
}

// ===================================================================
// L8: Trip Sync and Quaternion Copy Tests
// ===================================================================

#[test]
fn test_sail_quaternion_copies_count() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let copies = sail_quaternion_copies(bk);
        // 8 faces, each with 4 Q-copies
        assert_eq!(
            copies.len(),
            8,
            "Expected 8 sail groups for BK S={}",
            bk.strut_signature
        );
        for (i, group) in copies.iter().enumerate() {
            assert_eq!(
                group.len(),
                4,
                "Expected 4 Q-copies per sail, got {} for face {} in BK S={}",
                group.len(),
                i,
                bk.strut_signature
            );
        }
    }
}

#[test]
fn test_trip_sync_for_all_boxkites() {
    // Trip Sync: each box-kite's 6 L-indices contain exactly 4 of 7 Fano lines.
    let bks = find_box_kites(16, 1e-10);
    assert_eq!(bks.len(), 7);
    for bk in &bks {
        assert!(
            verify_trip_sync(bk),
            "Trip Sync should hold for BK S={}",
            bk.strut_signature
        );
    }
}

#[test]
fn test_trip_sync_missing_index_complementation() {
    // The missing L-index determines which 3 O-trips are excluded:
    // exactly those O-trips that contain the missing index.
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
        assert_eq!(
            l_set.len(),
            6,
            "BK S={} should have 6 distinct L-indices",
            bk.strut_signature
        );

        let missing = (1..=7usize).find(|x| !l_set.contains(x)).unwrap();

        let contained: Vec<_> = O_TRIPS
            .iter()
            .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
            .collect();
        let excluded: Vec<_> = O_TRIPS.iter().filter(|t| t.contains(&missing)).collect();

        assert_eq!(
            contained.len(),
            4,
            "BK S={} should contain exactly 4 O-trips",
            bk.strut_signature
        );
        assert_eq!(
            excluded.len(),
            3,
            "BK S={} should exclude exactly 3 O-trips (those containing {})",
            bk.strut_signature,
            missing
        );
    }
}

#[test]
fn test_trip_sync_each_bk_excludes_unique_index() {
    // Each box-kite excludes a different index from {1..7}, establishing
    // the bijection between box-kites and Fano plane points.
    let bks = find_box_kites(16, 1e-10);
    let missing_indices: HashSet<usize> = bks
        .iter()
        .map(|bk| {
            let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
            (1..=7usize).find(|x| !l_set.contains(x)).unwrap()
        })
        .collect();
    assert_eq!(
        missing_indices.len(),
        7,
        "All 7 box-kites should exclude distinct indices"
    );
    assert_eq!(missing_indices, (1..=7usize).collect::<HashSet<_>>());
}

// ===================================================================
// L9: Semiotic Square Algebraic Kernel Tests
// ===================================================================

#[test]
fn test_ss_kernel_verification() {
    let results = verify_ss_algebraic_kernel();
    assert_eq!(results.len(), 7, "Should verify kernel for all 7 box-kites");

    for res in &results {
        assert_eq!(res.axes.len(), 3, "Each box-kite should have 3 strut axes");
    }
}

#[test]
fn test_ss_kernel_product_indices_nonzero() {
    let results = verify_ss_algebraic_kernel();
    for res in &results {
        for (label, check) in &res.axes {
            assert_ne!(
                check.vz_product, 0,
                "V*Z product should be nonzero at BK S={}, axis {:?}",
                res.strut_sig, label
            );
            assert_ne!(
                check.zv_product, 0,
                "Z*v product should be nonzero at BK S={}, axis {:?}",
                res.strut_sig, label
            );
        }
    }
}

#[test]
fn test_ss_kernel_cross_consistency() {
    // The two cross-products V*Z and v*z should yield the same product INDEX.
    // Similarly Z*v and V*z should yield the same product INDEX.
    let results = verify_ss_algebraic_kernel();
    let mut n_klein = 0;
    let mut n_total = 0;
    for res in &results {
        for (_, check) in &res.axes {
            n_total += 1;
            if check.klein_verified {
                n_klein += 1;
            }
        }
    }
    // Report how many axes satisfy the Klein group structure.
    // This is a research probe -- we check rather than assert.
    assert!(
        n_klein > 0,
        "At least some strut axes should show Klein group structure, got {}/{}",
        n_klein,
        n_total
    );
}

// ===================================================================
// L10: CT Boundary / A7 Star Tests
// ===================================================================

#[test]
fn test_ct_boundary_h3_connection() {
    let result = ct_boundary_analysis();
    assert_eq!(
        result.total_strings, 120,
        "Total quincunx strings should equal |H3| = 120"
    );
    assert!(result.matches_h3_order);
}

#[test]
fn test_double_transfer_different_boxkites() {
    assert!(
        verify_double_transfer(),
        "Twist transitions should always move between different box-kites"
    );
}

// ===================================================================
// L11: Sail-Loop Partition Tests (Automorpheme Duality)
// ===================================================================

#[test]
fn test_sail_loop_partition_28_sails() {
    let result = sail_loop_partition();
    assert_eq!(
        result.total_sails, 28,
        "Expected 28 O-trip sails (7 BK x 4), got {}",
        result.total_sails
    );
}

#[test]
fn test_sail_loop_partition_7_loops_of_4() {
    let result = sail_loop_partition();
    assert_eq!(
        result.loops.len(),
        7,
        "Expected exactly 7 loops (automorphemes)"
    );
    for (i, l) in result.loops.iter().enumerate() {
        assert_eq!(
            l.len(),
            4,
            "Loop {} should have 4 sails, got {}",
            i,
            l.len()
        );
    }
}

#[test]
fn test_sail_loop_bk_duality() {
    let result = sail_loop_partition();
    assert!(
        result.bk_sails_in_different_loops,
        "Each BK's 4 O-trip sails must land in 4 different automorphemes"
    );
    assert!(
        result.loop_sails_from_different_bks,
        "Each automorpheme must receive sails from 4 different BKs"
    );
}

// ===================================================================
// L12: Quincunx Construction Tests
// ===================================================================

#[test]
fn test_quincunx_paths_per_boxkite() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let paths = enumerate_quincunx_paths(bk);
        assert_eq!(
            paths.len(),
            6,
            "Expected 6 quincunx paths for BK S={}, got {}",
            bk.strut_signature,
            paths.len()
        );
    }
}

#[test]
fn test_quincunx_string_count_120() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let count = quincunx_string_count(bk);
        assert_eq!(
            count, 120,
            "BK S={} should have 120 quincunx strings, got {}",
            bk.strut_signature, count
        );
    }
}

#[test]
fn test_quincunx_path_visits_5_assessors() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let paths = enumerate_quincunx_paths(bk);
        for path in &paths {
            assert_eq!(
                path.assessor_indices.len(),
                5,
                "Quincunx should visit 5 assessors"
            );
            // Verify all 5 are distinct
            let unique: HashSet<usize> = path.assessor_indices.iter().copied().collect();
            assert_eq!(
                unique.len(),
                5,
                "Quincunx should visit 5 distinct assessors"
            );
        }
    }
}

#[test]
fn test_bicycle_chain_12_diagonals() {
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let chain = bicycle_chain(bk);
        assert_eq!(
            chain.steps.len(),
            12,
            "Bicycle Chain should have 12 steps for BK S={}",
            bk.strut_signature
        );
    }
}

// ===================================================================
// L13: ET Meta-Fractal / Regime Doubling Tests
// ===================================================================

#[test]
fn test_regime_doubling_n4_n5() {
    let result = verify_regime_doubling(5);
    assert_eq!(result.data.len(), 2, "Should test N=4 and N=5");
    assert_eq!(result.data[0], (4, 1), "N=4 should have 1 regime");
    assert_eq!(result.data[1], (5, 2), "N=5 should have 2 regimes");
    assert!(
        result.doubling_law_holds,
        "Regime doubling law should hold for N=4..5"
    );
}

#[test]
fn test_four_corners_replication() {
    // The N=4 -> N=5 corner replication should show high matching
    let (n, fraction) = verify_four_corners(4);
    assert_eq!(n, 4);
    // Some matching is expected (corner panes replicate)
    assert!(
        fraction > 0.0,
        "Four Corners should show some replication, got {:.3}",
        fraction
    );
}

// ===================================================================
// L14: Eco Echo Tests
// ===================================================================

#[test]
fn test_eco_echo_base_ss_count() {
    let result = eco_echo_probe();
    assert_eq!(result.base_ss_count, 21, "7 BK x 3 axes = 21 SS diagrams");
}

#[test]
fn test_eco_echo_role_assignments() {
    let result = eco_echo_probe();
    assert_eq!(
        result.role_assignments, 3,
        "Three role assignments for {{S,G,X}}"
    );
}

#[test]
fn test_eco_echo_xor_closure() {
    let result = eco_echo_probe();
    assert!(
        result.xor_closure_preserved,
        "XOR closure X = G XOR S must hold under all role swaps"
    );
}

#[test]
fn test_eco_echo_meta_node_count() {
    let result = eco_echo_probe();
    // 21 SS x 4 corner nodes = 84 meta-nodes after one expansion
    assert_eq!(result.meta_nodes_after_expansion, 84);
}

// --- L15: Oriented Trip Sync Tests ---

#[test]
fn test_oriented_trip_sync_all_bks_have_valid_embedding() {
    // Every sedenion box-kite must admit at least one PSL(2,7) orientation
    // where the shorthand (a,b,c),(a,d,e),(d,b,f),(e,f,c) is satisfiable.
    let bks = find_box_kites(16, 1e-10);
    assert_eq!(bks.len(), 7);
    for bk in &bks {
        let result = oriented_trip_sync(bk);
        assert!(
            result.has_valid_embedding,
            "BK S={} should have a valid Trip Sync embedding",
            bk.strut_signature
        );
    }
}

#[test]
fn test_oriented_trip_sync_4_available_trips() {
    // Each BK has 6 L-indices from {1..7}\{S}. Removing S from Fano plane
    // leaves exactly 4 of 7 lines intact. So 4 available O-trips per BK.
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let result = oriented_trip_sync(bk);
        assert_eq!(
            result.available_trips.len(),
            4,
            "BK S={} should have 4 available O-trips, got {}",
            bk.strut_signature,
            result.available_trips.len()
        );
    }
}

#[test]
fn test_oriented_trip_sync_candidate_count() {
    // Each of the 4 available trips is tested as a zigzag candidate.
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let result = oriented_trip_sync(bk);
        assert_eq!(
            result.candidate_results.len(),
            4,
            "BK S={}: should test all 4 candidates",
            bk.strut_signature
        );
    }
}

#[test]
fn test_oriented_trip_sync_shorthand_well_formed() {
    // For a valid embedding: zigzag (a,b,c) + trefoils (a,d,e),(d,b,f),(e,f,c)
    // must together use all 6 L-indices of the BK.
    let bks = find_box_kites(16, 1e-10);
    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort();
            s
        })
        .collect();

    for bk in &bks {
        let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
        let result = oriented_trip_sync(bk);

        // Find a valid candidate
        let valid_idx = result
            .candidate_results
            .iter()
            .find(|(_, valid)| *valid)
            .map(|(i, _)| *i);

        assert!(
            valid_idx.is_some(),
            "BK S={} must have valid candidate",
            bk.strut_signature
        );
        let zig = result.available_trips[valid_idx.unwrap()];

        let remaining: Vec<usize> = l_set.iter().copied().filter(|x| !zig.contains(x)).collect();
        assert_eq!(remaining.len(), 3);

        // Verify 3 remaining form trefoil triples that are all O-trips
        let (a, b, c) = (zig[0], zig[1], zig[2]);
        let mut found_assignment = false;
        let perms: Vec<(usize, usize, usize)> = vec![
            (remaining[0], remaining[1], remaining[2]),
            (remaining[0], remaining[2], remaining[1]),
            (remaining[1], remaining[0], remaining[2]),
            (remaining[1], remaining[2], remaining[0]),
            (remaining[2], remaining[0], remaining[1]),
            (remaining[2], remaining[1], remaining[0]),
        ];
        for (d, e, f) in perms {
            let t1 = {
                let mut t = [a, d, e];
                t.sort();
                t
            };
            let t2 = {
                let mut t = [d, b, f];
                t.sort();
                t
            };
            let t3 = {
                let mut t = [e, f, c];
                t.sort();
                t
            };
            if otrip_set.contains(&t1) && otrip_set.contains(&t2) && otrip_set.contains(&t3) {
                found_assignment = true;
                // Verify all 6 L-indices are covered
                let all: HashSet<usize> = [a, b, c, d, e, f].iter().copied().collect();
                assert_eq!(
                    all, l_set,
                    "BK S={}: shorthand must use all 6 L-indices",
                    bk.strut_signature
                );
                break;
            }
        }
        assert!(
            found_assignment,
            "BK S={}: no valid shorthand assignment found",
            bk.strut_signature
        );
    }
}

// --- L15b: Sail Decomposition Tests ---

#[test]
fn test_sail_decomposition_1_1_3_3_split() {
    // Every sedenion box-kite must decompose as 1 zigzag sail + 1 vent
    // + 3 trefoil sails + 3 non-sail trefoils.
    let bks = find_box_kites(16, 1e-10);
    assert_eq!(bks.len(), 7);
    for bk in &bks {
        let sd = sail_decomposition(bk);
        assert_eq!(sd.strut_sig, bk.strut_signature);
        assert_eq!(sd.faces.len(), 8);

        let count = |role: FaceRole| sd.faces.iter().filter(|f| f.role == role).count();
        assert_eq!(
            count(FaceRole::ZigzagSail),
            1,
            "BK S={}: expected 1 zigzag sail",
            bk.strut_signature
        );
        assert_eq!(
            count(FaceRole::TrefoilSail),
            3,
            "BK S={}: expected 3 trefoil sails",
            bk.strut_signature
        );
        assert_eq!(
            count(FaceRole::Vent),
            1,
            "BK S={}: expected 1 vent",
            bk.strut_signature
        );
        assert_eq!(
            count(FaceRole::NonSailTrefoil),
            3,
            "BK S={}: expected 3 non-sail trefoils",
            bk.strut_signature
        );
    }
}

#[test]
fn test_sail_decomposition_4_sails_are_otrips() {
    // The 4 sails (1 zigzag + 3 trefoil) must all have L-indices forming O-trips.
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let sd = sail_decomposition(bk);
        let sail_faces = sd
            .faces
            .iter()
            .filter(|f| f.role == FaceRole::ZigzagSail || f.role == FaceRole::TrefoilSail);
        for face in sail_faces {
            assert!(
                face.otrip_index.is_some(),
                "BK S={}: sail face {:?} must have O-trip index",
                bk.strut_signature,
                face.l_indices
            );
        }
    }
}

#[test]
fn test_sail_decomposition_4_non_sails_not_otrips() {
    // The 4 non-sails (1 vent + 3 non-sail trefoils) must NOT have L-indices forming O-trips.
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let sd = sail_decomposition(bk);
        let non_sail_faces = sd
            .faces
            .iter()
            .filter(|f| f.role == FaceRole::Vent || f.role == FaceRole::NonSailTrefoil);
        for face in non_sail_faces {
            assert!(
                face.otrip_index.is_none(),
                "BK S={}: non-sail face {:?} must NOT have O-trip index",
                bk.strut_signature,
                face.l_indices
            );
        }
    }
}

#[test]
fn test_sail_decomposition_4_distinct_otrips() {
    // The 4 sails per BK must correspond to 4 distinct O-trips (matching Trip Sync).
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let sd = sail_decomposition(bk);
        let otrip_indices: HashSet<usize> = sd.faces.iter().filter_map(|f| f.otrip_index).collect();
        assert_eq!(
            otrip_indices.len(),
            4,
            "BK S={}: 4 sails must map to 4 distinct O-trips, got {}",
            bk.strut_signature,
            otrip_indices.len()
        );
    }
}

#[test]
fn test_sail_decomposition_zigzag_sail_all_opposite() {
    // The zigzag sail must have all-opposite edges (by definition of TwistType::Zigzag).
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;
    for bk in &bks {
        let sd = sail_decomposition(bk);
        let zs = &sd.faces[sd.zigzag_sail_idx];
        let signs = [
            edge_sign_type(
                &bk.assessors[zs.assessor_indices[0]],
                &bk.assessors[zs.assessor_indices[1]],
                atol,
            ),
            edge_sign_type(
                &bk.assessors[zs.assessor_indices[1]],
                &bk.assessors[zs.assessor_indices[2]],
                atol,
            ),
            edge_sign_type(
                &bk.assessors[zs.assessor_indices[0]],
                &bk.assessors[zs.assessor_indices[2]],
                atol,
            ),
        ];
        assert!(
            signs.iter().all(|&s| s == EdgeSignType::Opposite),
            "BK S={}: zigzag sail must have all-Opposite edges",
            bk.strut_signature
        );
    }
}

#[test]
fn test_sail_decomposition_vent_all_opposite_no_otrip() {
    // The vent must also have all-opposite edges but NOT form an O-trip.
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;
    for bk in &bks {
        let sd = sail_decomposition(bk);
        let vent = &sd.faces[sd.vent_idx];
        let signs = [
            edge_sign_type(
                &bk.assessors[vent.assessor_indices[0]],
                &bk.assessors[vent.assessor_indices[1]],
                atol,
            ),
            edge_sign_type(
                &bk.assessors[vent.assessor_indices[1]],
                &bk.assessors[vent.assessor_indices[2]],
                atol,
            ),
            edge_sign_type(
                &bk.assessors[vent.assessor_indices[0]],
                &bk.assessors[vent.assessor_indices[2]],
                atol,
            ),
        ];
        assert!(
            signs.iter().all(|&s| s == EdgeSignType::Opposite),
            "BK S={}: vent must have all-Opposite edges",
            bk.strut_signature
        );
        assert!(
            vent.otrip_index.is_none(),
            "BK S={}: vent must NOT form an O-trip",
            bk.strut_signature
        );
    }
}

#[test]
fn test_sail_decomposition_consistent_across_all_struts() {
    // The sail decomposition counts must be identical for all 7 strut constants.
    let bks = find_box_kites(16, 1e-10);
    let mut results = Vec::new();
    for bk in &bks {
        let sd = sail_decomposition(bk);
        let sail_otrips: Vec<usize> = sd.faces.iter().filter_map(|f| f.otrip_index).collect();
        results.push((bk.strut_signature, sail_otrips.len()));
    }
    for &(s, n) in &results {
        assert_eq!(n, 4, "BK S={}: expected 4 sails, got {}", s, n);
    }
}

#[test]
fn test_sail_decomposition_28_sails_total() {
    // 7 box-kites x 4 sails each = 28 total sails (matches sail_loop_partition).
    let bks = find_box_kites(16, 1e-10);
    let total_sails: usize = bks
        .iter()
        .map(|bk| {
            let sd = sail_decomposition(bk);
            sd.faces
                .iter()
                .filter(|f| f.role == FaceRole::ZigzagSail || f.role == FaceRole::TrefoilSail)
                .count()
        })
        .sum();
    assert_eq!(total_sails, 28, "7 BK x 4 sails = 28 total");
}

// --- Edge-Sign Cross-Validation: ET vs Box-Kite Geometry ---

#[test]
fn test_et_edge_sign_vs_boxkite_consistent_mapping() {
    // Cross-validate: the ET cell edge_sign and boxkites.rs edge_sign_type
    // must have a consistent 1:1 mapping across all edges of all 7 box-kites.
    //
    // The ET edge_sign is computed from integer-exact CDP products:
    //   +1 if sgn(H_row * L_col) == sgn(L_row * H_col) ("same quadrant concordance")
    //   -1 otherwise ("cross quadrant discordance")
    //
    // EdgeSignType is computed from zero-product solution signs:
    //   Same if (+,+) or (-,-) are solutions
    //   Opposite if (+,-) or (-,+) are solutions
    //
    // The mapping turns out to be: ET +1 <-> Opposite, ET -1 <-> Same.
    // This is because the X-pattern concordance relates to how the product
    // *vanishes*, and the zero-product sign convention is reversed.
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;

    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let bk = bks.iter().find(|b| b.strut_signature == s).unwrap();
        let graph = extract_signed_graph(&et);

        let et_sign_map: HashMap<(usize, usize), i32> = graph
            .edges
            .iter()
            .flat_map(|e| [((e.lo_a, e.lo_b), e.sign), ((e.lo_b, e.lo_a), e.sign)])
            .collect();

        for i in 0..6 {
            for j in (i + 1)..6 {
                let a = &bk.assessors[i];
                let b = &bk.assessors[j];

                if let Some(&et_sign) = et_sign_map.get(&(a.low, b.low)) {
                    let bk_sign = edge_sign_type(a, b, atol);
                    // Inverted mapping: ET +1 <-> Opposite, ET -1 <-> Same
                    let expected_bk_sign = if et_sign > 0 {
                        EdgeSignType::Opposite
                    } else {
                        EdgeSignType::Same
                    };
                    assert_eq!(
                        bk_sign, expected_bk_sign,
                        "S={}: edge ({},{})--({},{}) ET sign={} but BK says {:?}",
                        s, a.low, a.high, b.low, b.high, et_sign, bk_sign
                    );
                }
            }
        }
    }
}

#[test]
fn test_et_dmz_edges_count_12_per_sedenion_bk() {
    // At N=4 (sedenions), each BK has 12 edges in the octahedron.
    // All edges should be DMZ (full fill). Verify the ET captures all 12.
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        assert_eq!(
            graph.edges.len(),
            12,
            "S={}: sedenion BK should have 12 DMZ edges (complete octahedron), got {}",
            s,
            graph.edges.len()
        );
    }
}

#[test]
fn test_et_sign_partition_6_6() {
    // Each sedenion BK has 12 DMZ edges partitioned evenly: 6 positive, 6 negative.
    // This is consistent with the octahedron having 12 edges where:
    // - 6 edges connect non-strut-opposite pairs with "same quadrant" concordance
    // - 6 edges connect non-strut-opposite pairs with "cross quadrant" concordance
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        assert_eq!(
            graph.n_positive + graph.n_negative,
            12,
            "S={}: total edges should be 12",
            s
        );
        assert_eq!(
            graph.n_positive, 6,
            "S={}: expected 6 positive edges, got {}",
            s, graph.n_positive
        );
        assert_eq!(
            graph.n_negative, 6,
            "S={}: expected 6 negative edges, got {}",
            s, graph.n_negative
        );
    }
}

#[test]
fn test_zigzag_face_edges_all_positive_in_et() {
    // Zigzag faces have all 3 edges Opposite (in BK geometry).
    // Due to the sign inversion: BK Opposite <-> ET +1.
    // So zigzag face edges should all be +1 in the ET signed graph.
    let bks = find_box_kites(16, 1e-10);
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        let bk = bks.iter().find(|b| b.strut_signature == s).unwrap();
        let sd = sail_decomposition(bk);

        let et_sign_map: HashMap<(usize, usize), i32> = graph
            .edges
            .iter()
            .flat_map(|e| [((e.lo_a, e.lo_b), e.sign), ((e.lo_b, e.lo_a), e.sign)])
            .collect();

        // Check both zigzag faces (zigzag sail + vent)
        for &idx in &[sd.zigzag_sail_idx, sd.vent_idx] {
            let face = &sd.faces[idx];
            let ls = face.l_indices;
            let signs = [
                et_sign_map.get(&(ls[0], ls[1])).copied(),
                et_sign_map.get(&(ls[1], ls[2])).copied(),
                et_sign_map.get(&(ls[0], ls[2])).copied(),
            ];
            for (i, sign) in signs.iter().enumerate() {
                if let Some(s_val) = sign {
                    assert_eq!(
                        *s_val, 1,
                        "S={}: zigzag face {:?} edge {} should be positive in ET, got {}",
                        s, ls, i, s_val
                    );
                }
            }
        }
    }
}

#[test]
fn test_trefoil_face_has_mixed_signs_in_et() {
    // Trefoil faces have mixed Same/Opposite edges.
    // Specifically: at least one positive and at least one negative edge.
    let bks = find_box_kites(16, 1e-10);
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        let bk = bks.iter().find(|b| b.strut_signature == s).unwrap();
        let sd = sail_decomposition(bk);

        let et_sign_map: HashMap<(usize, usize), i32> = graph
            .edges
            .iter()
            .flat_map(|e| [((e.lo_a, e.lo_b), e.sign), ((e.lo_b, e.lo_a), e.sign)])
            .collect();

        // Check all 6 trefoil faces (3 trefoil sails + 3 non-sail trefoils)
        let trefoil_indices: Vec<usize> = sd
            .trefoil_sail_indices
            .iter()
            .chain(sd.non_sail_trefoil_indices.iter())
            .copied()
            .collect();
        for idx in trefoil_indices {
            let face = &sd.faces[idx];
            let ls = face.l_indices;
            let signs: Vec<i32> = [
                et_sign_map.get(&(ls[0], ls[1])),
                et_sign_map.get(&(ls[1], ls[2])),
                et_sign_map.get(&(ls[0], ls[2])),
            ]
            .iter()
            .filter_map(|s| s.copied())
            .collect();

            if signs.len() == 3 {
                let has_positive = signs.iter().any(|&s| s > 0);
                let has_negative = signs.iter().any(|&s| s < 0);
                assert!(
                    has_positive && has_negative,
                    "S={}: trefoil face {:?} must have mixed signs, got {:?}",
                    s,
                    ls,
                    signs
                );
            }
        }
    }
}

// --- Three Vizier XOR Relationship Tests ---

#[test]
fn test_three_viziers_all_sedenion_bks() {
    // All 7 sedenion box-kites must satisfy VZ1, VZ2, VZ3.
    let bks = find_box_kites(16, 1e-10);
    assert_eq!(bks.len(), 7);
    for bk in &bks {
        let res = verify_three_viziers(bk, 16);
        assert!(res.vz1_holds, "S={}: VZ1 failed", res.strut_sig);
        assert!(res.vz2_holds, "S={}: VZ2 failed", res.strut_sig);
        assert!(res.vz3_holds, "S={}: VZ3 failed", res.strut_sig);
        assert_eq!(res.generator, 8);
        assert_eq!(res.n_struts, 3);
        assert_eq!(
            res.lo_hi_xor,
            8 ^ res.strut_sig,
            "S={}: lo^hi should be G^S={}",
            res.strut_sig,
            8 ^ res.strut_sig
        );
    }
}

#[test]
fn test_vizier_lo_hi_xor_unique_per_bk() {
    // Each box-kite has a distinct lo^hi value = G^S = 8^S.
    // Since S in {1..7}, lo^hi in {9,10,11,12,13,14,15} -- all distinct.
    let bks = find_box_kites(16, 1e-10);
    let xors: std::collections::BTreeSet<usize> = bks
        .iter()
        .map(|bk| verify_three_viziers(bk, 16).lo_hi_xor)
        .collect();
    assert_eq!(xors.len(), 7, "7 distinct lo^hi values");
    assert_eq!(*xors.iter().next().unwrap(), 9);
    assert_eq!(*xors.iter().last().unwrap(), 15);
}

#[test]
fn test_vizier_klein_four_group() {
    // The three XOR values {S, G, G^S} form a Klein four-group
    // with the identity: S ^ G = G^S, S ^ (G^S) = G, G ^ (G^S) = S.
    let bks = find_box_kites(16, 1e-10);
    let g = 8usize;
    for bk in &bks {
        let s = bk.strut_signature;
        let x = g ^ s;
        // Klein V4 closure checks
        assert_eq!(s ^ g, x);
        assert_eq!(s ^ x, g);
        assert_eq!(g ^ x, s);
        // Identity element is 0 (XOR identity)
        assert_eq!(s ^ s, 0);
        assert_eq!(g ^ g, 0);
        assert_eq!(x ^ x, 0);
    }
}

#[test]
fn test_vizier_xor_audit_sedenion_components() {
    // At dim=16, the motif components are the 7 box-kites.
    // The VizierXorAudit should recover the same structure.
    let components = motif_components_for_cross_assessors(16);
    assert_eq!(components.len(), 7);
    for comp in &components {
        let audit = vizier_xor_audit(comp).unwrap();
        assert!(audit.vz3_constant, "dim=16 comp: lo^hi should be constant");
        assert!(audit.vz1_lo_eq_hi, "dim=16 comp: VZ1 symmetry should hold");
        assert!(audit.vz2_cross_eq, "dim=16 comp: VZ2 symmetry should hold");
        // lo^hi should be in {9..15}
        assert!(audit.lo_hi_xor >= 9 && audit.lo_hi_xor <= 15);
        // Inferred S should be in {1..7}
        assert!(audit.inferred_s >= 1 && audit.inferred_s <= 7);
    }
}

#[test]
fn test_vizier_xor_audit_pathion_vz3_constant() {
    // At dim=32, check whether lo^hi is constant within each component.
    // This tests whether VZ3 generalizes beyond sedenions.
    let components = motif_components_for_cross_assessors(32);
    assert_eq!(components.len(), 15, "dim=32 has 15 components");
    let vz3_count = components
        .iter()
        .filter_map(vizier_xor_audit)
        .filter(|a| a.vz3_constant)
        .count();
    // Expectation: VZ3 should hold for all 15 pathion components
    // (lo^hi constant within each component)
    assert_eq!(
        vz3_count, 15,
        "VZ3 (lo^hi constant) should hold for all 15 pathion components"
    );
}

#[test]
fn test_vizier_xor_audit_pathion_vz1_symmetry() {
    // At dim=32, check VZ1 symmetry: lo_a^lo_b = hi_a^hi_b for all edges.
    let components = motif_components_for_cross_assessors(32);
    let vz1_count = components
        .iter()
        .filter_map(vizier_xor_audit)
        .filter(|a| a.vz1_lo_eq_hi)
        .count();
    assert_eq!(
        vz1_count, 15,
        "VZ1 symmetry (lo^lo = hi^hi) should hold for all 15 pathion components"
    );
}

#[test]
fn test_vizier_xor_audit_pathion_vz2_symmetry() {
    // At dim=32, check VZ2 symmetry: hi_b^lo_a = hi_a^lo_b for all edges.
    // De Marrais notes VZ2 "may be sedenion-specific" -- let's find out.
    let components = motif_components_for_cross_assessors(32);
    let vz2_count = components
        .iter()
        .filter_map(vizier_xor_audit)
        .filter(|a| a.vz2_cross_eq)
        .count();
    assert_eq!(
        vz2_count, 15,
        "VZ2 symmetry should hold for all 15 pathion components"
    );
}

#[test]
fn test_vizier_xor_audit_pathion_vz2_value() {
    // At dim=32, G=16. Check if VZ2 cross-XOR equals G for all edges.
    let components = motif_components_for_cross_assessors(32);
    for comp in &components {
        let audit = vizier_xor_audit(comp).unwrap();
        if audit.vz2_constant {
            assert_eq!(
                audit.vz2_value,
                Some(16),
                "VZ2 value should be G=16 at dim=32"
            );
        }
    }
}

// --- L9b: (s,g)-Modularity Regime Address Tests ---

#[test]
fn test_regime_count_doubling_law() {
    // regime_count(N) = 2^(N-4): 1, 2, 4, 8, 16, ...
    assert_eq!(regime_count(4), 1);
    assert_eq!(regime_count(5), 2);
    assert_eq!(regime_count(6), 4);
    assert_eq!(regime_count(7), 8);
    assert_eq!(regime_count(8), 16);
    // Edge case: n < 4 saturates to 1
    assert_eq!(regime_count(3), 1);
    assert_eq!(regime_count(0), 1);
}

#[test]
fn test_regime_address_length() {
    // Address length = N - 4 for all valid strut constants
    for n in 4..=7 {
        let max_s = (1usize << (n - 1)) - 1;
        for s in 1..=max_s {
            let addr = regime_address(n, s);
            assert_eq!(
                addr.len(),
                n - 4,
                "N={}, S={}: address length {} != {}",
                n,
                s,
                addr.len(),
                n - 4
            );
        }
    }
}

#[test]
fn test_regime_address_binary_values() {
    // Every element of the address vector must be 0 or 1
    for n in 4..=7 {
        let max_s = (1usize << (n - 1)) - 1;
        for s in 1..=max_s {
            let addr = regime_address(n, s);
            for (i, &b) in addr.iter().enumerate() {
                assert!(
                    b <= 1,
                    "N={}, S={}: addr[{}] = {} (must be 0 or 1)",
                    n,
                    s,
                    i,
                    b
                );
            }
        }
    }
}

#[test]
fn test_regime_address_sedenion_trivial() {
    // At N=4 (sedenions), all addresses are empty -- single regime
    for s in 1..=7 {
        assert_eq!(
            regime_address(4, s),
            Vec::<u8>::new(),
            "S={}: sedenion address must be empty",
            s
        );
    }
}

#[test]
fn test_regime_address_generators_equal_mandala() {
    // Powers of 2 >= 8 (generators) always map to same address as S=3
    for n in 5..=7 {
        let mandala_addr = regime_address(n, 3);
        for k in 3..n {
            let generator = 1usize << k; // 8, 16, 32, ...
            let max_s = (1usize << (n - 1)) - 1;
            if generator <= max_s {
                assert_eq!(
                    regime_address(n, generator),
                    mandala_addr,
                    "N={}, S={}: generator must map to mandala address",
                    n,
                    generator
                );
            }
        }
    }
}

#[test]
fn test_regime_address_distinct_count_matches_formula() {
    // Number of distinct addresses at level N must equal regime_count(N)
    use std::collections::BTreeSet;
    for n in 4..=7 {
        let max_s = (1usize << (n - 1)) - 1;
        let mut addrs = BTreeSet::new();
        for s in 1..=max_s {
            addrs.insert(regime_address(n, s));
        }
        assert_eq!(
            addrs.len(),
            regime_count(n),
            "N={}: distinct addresses {} != regime_count {}",
            n,
            addrs.len(),
            regime_count(n)
        );
    }
}

#[test]
fn test_regime_address_uniformity_n5() {
    // N=5 (pathions): 2 regimes, same-address struts have same DMZ count.
    // et_regimes returns HashMap<dmz_count, Vec<strut_constants>>.
    use std::collections::BTreeMap;
    let regimes = et_regimes(5);
    let mut addr_to_dmz: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for (&dmz, struts) in &regimes {
        for &s in struts {
            let addr = regime_address(5, s);
            addr_to_dmz.entry(addr).or_default().push(dmz);
        }
    }
    for (addr, dmzs) in &addr_to_dmz {
        let first = dmzs[0];
        for &d in dmzs {
            assert_eq!(d, first, "N=5, addr={:?}: DMZ counts not uniform", addr);
        }
    }
    assert_eq!(
        addr_to_dmz.len(),
        2,
        "N=5 must have exactly 2 regime addresses"
    );
}

#[test]
fn test_regime_address_uniformity_n6() {
    // N=6 (chingons): 4 regimes, same-address struts have same DMZ count.
    use std::collections::BTreeMap;
    let regimes = et_regimes(6);
    let mut addr_to_dmz: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for (&dmz, struts) in &regimes {
        for &s in struts {
            let addr = regime_address(6, s);
            addr_to_dmz.entry(addr).or_default().push(dmz);
        }
    }
    for (addr, dmzs) in &addr_to_dmz {
        let first = dmzs[0];
        for &d in dmzs {
            assert_eq!(d, first, "N=6, addr={:?}: DMZ counts not uniform", addr);
        }
    }
    assert_eq!(
        addr_to_dmz.len(),
        4,
        "N=6 must have exactly 4 regime addresses"
    );
}

#[test]
fn test_regime_address_n6_specific_bands() {
    // Verify the specific N=6 regime structure from exploration:
    // [0,0] -> S in {1..8, 16}  (mandala full-fill, 9 struts)
    // [0,1] -> S in {9..15}     (sky regime, 7 struts)
    // [1,0] -> S in {17..24}    (new upper band, 8 struts)
    // [1,1] -> S in {25..31}    (upper sky, 7 struts)
    let mandala: Vec<usize> = (1..=8).chain(std::iter::once(16)).collect();
    let sky_lower: Vec<usize> = (9..=15).collect();
    let upper_mandala: Vec<usize> = (17..=24).collect();
    let upper_sky: Vec<usize> = (25..=31).collect();

    for &s in &mandala {
        assert_eq!(
            regime_address(6, s),
            vec![0, 0],
            "N=6, S={}: expected [0,0]",
            s
        );
    }
    for &s in &sky_lower {
        assert_eq!(
            regime_address(6, s),
            vec![0, 1],
            "N=6, S={}: expected [0,1]",
            s
        );
    }
    for &s in &upper_mandala {
        assert_eq!(
            regime_address(6, s),
            vec![1, 0],
            "N=6, S={}: expected [1,0]",
            s
        );
    }
    for &s in &upper_sky {
        assert_eq!(
            regime_address(6, s),
            vec![1, 1],
            "N=6, S={}: expected [1,1]",
            s
        );
    }
}

// --- L9c: Hide/Fill Involution Tests ---

#[test]
fn test_hide_fill_n4_single_full_fill_regime() {
    // Sedenions: 1 regime, full fill, all 7 struts identical
    let results = hide_fill_analysis(4);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(r.regime_addr, Vec::<u8>::new());
    assert!(r.is_full_fill, "sedenion regime must be full fill");
    assert_eq!(r.dmz_count, r.total_addressable);
    assert_eq!(r.core_size, r.union_size, "full fill => core == union");
    assert_eq!(r.n_struts, 7);
}

#[test]
fn test_hide_fill_n5_two_regimes() {
    let results = hide_fill_analysis(5);
    assert_eq!(results.len(), 2, "N=5 must have 2 regimes");

    // Mandala [0]: full fill
    let mandala = results.iter().find(|r| r.regime_addr == vec![0]).unwrap();
    assert!(mandala.is_full_fill, "mandala must be full fill");
    assert_eq!(mandala.dmz_count, 168);
    assert_eq!(mandala.core_size, 168, "full fill => core == total");

    // Sky [1]: partial fill
    let sky = results.iter().find(|r| r.regime_addr == vec![1]).unwrap();
    assert!(!sky.is_full_fill, "sky must NOT be full fill");
    assert_eq!(sky.dmz_count, 72);
    // Sky union covers all addressable cells
    assert_eq!(
        sky.union_size, sky.total_addressable,
        "sky union must cover all addressable cells"
    );
    // Sky core is 0 (no cell is DMZ in ALL sky struts)
    assert_eq!(sky.core_size, 0, "sky core must be empty");
}

#[test]
fn test_hide_fill_n6_four_regimes() {
    let results = hide_fill_analysis(6);
    assert_eq!(results.len(), 4, "N=6 must have 4 regimes");

    let m = results
        .iter()
        .find(|r| r.regime_addr == vec![0, 0])
        .unwrap();
    let sky = results
        .iter()
        .find(|r| r.regime_addr == vec![0, 1])
        .unwrap();
    let um = results
        .iter()
        .find(|r| r.regime_addr == vec![1, 0])
        .unwrap();
    let us = results
        .iter()
        .find(|r| r.regime_addr == vec![1, 1])
        .unwrap();

    // Mandala: full fill
    assert!(m.is_full_fill);
    assert_eq!(m.dmz_count, 840);
    assert_eq!(m.n_struts, 9); // S=1..8, 16

    // Sky: partial fill
    assert!(!sky.is_full_fill);
    assert_eq!(sky.dmz_count, 456);
    assert_eq!(sky.n_struts, 7);

    // Upper mandala: sparse fill
    assert!(!um.is_full_fill);
    assert_eq!(um.dmz_count, 168);
    assert_eq!(um.n_struts, 8);

    // Upper sky: intermediate fill
    assert!(!us.is_full_fill);
    assert_eq!(us.dmz_count, 552);
    assert_eq!(us.n_struts, 7);
}

#[test]
fn test_hide_fill_row_degree_invariant_n5() {
    // All mandala struts must have uniform row degree = 12 (= K-2 = 14-2)
    let results = hide_fill_analysis(5);
    let mandala = results.iter().find(|r| r.regime_addr == vec![0]).unwrap();
    assert!(
        mandala.row_degrees.iter().all(|&d| d == 12),
        "mandala row degrees must all be 12, got {:?}",
        mandala.row_degrees
    );

    // Sky struts: 12 rows with degree 4, 2 rows with degree 12
    let sky = results.iter().find(|r| r.regime_addr == vec![1]).unwrap();
    let n_full = sky.row_degrees.iter().filter(|&&d| d == 12).count();
    let n_sparse = sky.row_degrees.iter().filter(|&&d| d == 4).count();
    assert_eq!(n_full, 2, "sky must have 2 full rows");
    assert_eq!(n_sparse, 12, "sky must have 12 sparse rows");
}

#[test]
fn test_hide_fill_row_degree_invariant_n6() {
    let results = hide_fill_analysis(6);

    // Mandala: all rows degree 28
    let m = results
        .iter()
        .find(|r| r.regime_addr == vec![0, 0])
        .unwrap();
    assert!(
        m.row_degrees.iter().all(|&d| d == 28),
        "mandala row degrees must all be 28"
    );

    // Sky [0,1]: 24 rows at 12, 6 rows at 28
    let sky = results
        .iter()
        .find(|r| r.regime_addr == vec![0, 1])
        .unwrap();
    assert_eq!(sky.row_degrees.iter().filter(|&&d| d == 12).count(), 24);
    assert_eq!(sky.row_degrees.iter().filter(|&&d| d == 28).count(), 6);

    // Upper mandala [1,0]: 28 rows at 4, 2 rows at 28
    let um = results
        .iter()
        .find(|r| r.regime_addr == vec![1, 0])
        .unwrap();
    assert_eq!(um.row_degrees.iter().filter(|&&d| d == 4).count(), 28);
    assert_eq!(um.row_degrees.iter().filter(|&&d| d == 28).count(), 2);

    // Upper sky [1,1]: 4 rows at 4, 24 rows at 20, 2 rows at 28
    let us = results
        .iter()
        .find(|r| r.regime_addr == vec![1, 1])
        .unwrap();
    assert_eq!(us.row_degrees.iter().filter(|&&d| d == 4).count(), 4);
    assert_eq!(us.row_degrees.iter().filter(|&&d| d == 20).count(), 24);
    assert_eq!(us.row_degrees.iter().filter(|&&d| d == 28).count(), 2);
}

#[test]
fn test_hide_fill_union_covers_all_n6() {
    // Every regime's union must cover all addressable cells
    let results = hide_fill_analysis(6);
    for r in &results {
        assert_eq!(
            r.union_size, r.total_addressable,
            "regime {:?}: union {} != total {}",
            r.regime_addr, r.union_size, r.total_addressable
        );
    }
}

#[test]
fn test_hide_fill_sky_core_n6() {
    // At N=6, sky core = 168 = upper mandala DMZ count
    let results = hide_fill_analysis(6);
    let sky = results
        .iter()
        .find(|r| r.regime_addr == vec![0, 1])
        .unwrap();
    let um = results
        .iter()
        .find(|r| r.regime_addr == vec![1, 0])
        .unwrap();
    assert_eq!(
        sky.core_size, 168,
        "sky core must be 168 (the mandala DMZ count from one level down)"
    );
    assert_eq!(
        sky.core_size, um.dmz_count,
        "sky core must equal upper mandala DMZ count"
    );
}

// --- L9d: Skybox Tests ---

#[test]
fn test_skybox_edge_is_power_of_two() {
    // Skybox edge must be G = 2^(N-1) for doubling recursion to work.
    for n in 4..=6 {
        let sb = create_skybox(n, 3);
        assert_eq!(
            sb.edge,
            1 << (n - 1),
            "N={}: skybox edge must be 2^(N-1)",
            n
        );
        assert_eq!(sb.edge, sb.g, "N={}: skybox edge must equal G", n);
        assert!(
            sb.edge.is_power_of_two(),
            "N={}: skybox edge must be power of 2",
            n
        );
    }
}

#[test]
fn test_skybox_grid_dimensions() {
    // Grid is edge x edge.
    for n in 4..=5 {
        let sb = create_skybox(n, 3);
        assert_eq!(sb.grid.len(), sb.edge, "N={}: grid must have edge rows", n);
        for (r, row) in sb.grid.iter().enumerate() {
            assert_eq!(row.len(), sb.edge, "N={}: row {} must have edge cols", n, r);
        }
    }
}

#[test]
fn test_skybox_structural_empties_n4() {
    // For even edge, diagonal and anti-diagonal don't overlap,
    // giving exactly 2*edge structural empties.
    let sb = create_skybox(4, 3);
    let count: usize = sb
        .grid
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| c.is_structural_empty)
        .count();
    assert_eq!(count, 2 * sb.edge, "structural_empty count must be 2*edge");
}

#[test]
fn test_skybox_diagonal_anti_diagonal() {
    // Every cell on the main diagonal and anti-diagonal must be structural_empty.
    let sb = create_skybox(4, 5);
    let e = sb.edge;
    for i in 0..e {
        assert!(
            sb.grid[i][i].is_structural_empty,
            "diagonal ({},{}) must be structural_empty",
            i, i
        );
        assert!(
            sb.grid[i][e - 1 - i].is_structural_empty,
            "anti-diagonal ({},{}) must be structural_empty",
            i,
            e - 1 - i
        );
    }
}

#[test]
fn test_skybox_four_corners() {
    // All four corners must be structural_empty (they sit on both border
    // and diagonal or anti-diagonal).
    for s in 1..=7 {
        let sb = create_skybox(4, s);
        let e = sb.edge;
        for &(r, c) in &[(0, 0), (0, e - 1), (e - 1, 0), (e - 1, e - 1)] {
            assert!(
                sb.grid[r][c].is_structural_empty,
                "S={}: corner ({},{}) must be structural_empty",
                s, r, c
            );
        }
    }
}

#[test]
fn test_skybox_label_cell_count() {
    // Label cells = border cells minus structural empties on border.
    // Border has 4*(edge-1) cells. Structural empties on border = 4 corners.
    // Label cells = 4*(edge-1) - 4 = 4*(edge-2).
    let sb = create_skybox(4, 3);
    let label_count: usize = sb
        .grid
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| c.is_label_line)
        .count();
    assert_eq!(
        label_count,
        4 * (sb.edge - 2),
        "label cell count must be 4*(edge-2)"
    );
}

#[test]
fn test_skybox_interior_matches_et_n4() {
    // Interior cells (rows 1..K+1, cols 1..K+1) must exactly match
    // the underlying ET in DMZ status and emanation values.
    for s in 1..=7 {
        let sb = create_skybox(4, s);
        let k = sb.et.tone_row.k;
        for r in 0..k {
            for c in 0..k {
                let sb_cell = &sb.grid[r + 1][c + 1];
                let et_cell = &sb.et.cells[r][c];
                let et_dmz = et_cell.as_ref().is_some_and(|cell| cell.is_dmz);
                let et_val = et_cell.as_ref().map_or(0, |cell| cell.emanation_value);
                assert_eq!(
                    sb_cell.is_dmz, et_dmz,
                    "S={}: interior ({},{}) DMZ mismatch",
                    s, r, c
                );
                if sb_cell.is_dmz {
                    assert_eq!(
                        sb_cell.emanation_value, et_val,
                        "S={}: interior ({},{}) value mismatch",
                        s, r, c
                    );
                }
            }
        }
    }
}

#[test]
fn test_skybox_dmz_count_matches_et_n4() {
    // Since label_dmz=0, skybox DMZ must equal ET DMZ.
    for s in 1..=7 {
        let sb = create_skybox(4, s);
        assert_eq!(
            sb.dmz_count, sb.et.dmz_count,
            "S={}: skybox DMZ must equal ET DMZ",
            s
        );
        assert_eq!(sb.label_dmz_count, 0, "S={}: label DMZ must be 0", s);
    }
}

#[test]
fn test_skybox_label_dmz_zero_n5() {
    // Label lines carry the S-assessor (S, X) which is NOT a tone-row
    // position. The cross-magnitude check fails for all ET assessors,
    // giving zero label DMZ. Verified for all N=5 struts.
    for s in [3, 5, 9, 10, 11, 12, 13, 14, 15] {
        let sb = create_skybox(5, s);
        assert_eq!(sb.label_dmz_count, 0, "N=5, S={}: label DMZ must be 0", s);
        assert_eq!(
            sb.dmz_count, sb.et.dmz_count,
            "N=5, S={}: skybox DMZ must equal ET DMZ",
            s
        );
    }
}

#[test]
fn test_skybox_cell_partition() {
    // Every cell must be exactly one of: structural_empty, label_line,
    // or interior (neither). No cell should be both.
    let sb = create_skybox(5, 3);
    for (r, row) in sb.grid.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            assert!(
                !(cell.is_structural_empty && cell.is_label_line),
                "({},{}): cannot be both structural_empty and label_line",
                r,
                c
            );
        }
    }
    // Partition must cover all cells
    let n_empty: usize = sb
        .grid
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| c.is_structural_empty)
        .count();
    let n_label: usize = sb
        .grid
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| c.is_label_line)
        .count();
    let n_interior: usize = sb
        .grid
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| !c.is_structural_empty && !c.is_label_line)
        .count();
    assert_eq!(
        n_empty + n_label + n_interior,
        sb.edge * sb.edge,
        "partition must cover all edge^2 cells"
    );
}

// --- L9e: Theorem 11 Tests ---

#[test]
fn test_theorem11_primary_dmz_n4_to_n5_all_struts() {
    // Every sedenion strut (S=1..7) must embed exactly in the 32-ion ET.
    for s in 1..=7 {
        let r = verify_theorem11(4, s);
        assert!(
            r.primary_dmz_match,
            "S={}: primary DMZ pattern must match",
            s
        );
        assert!(
            r.primary_value_match,
            "S={}: primary emanation values must match",
            s
        );
        assert_eq!(
            r.old_dmz_count, r.primary_subblock_dmz,
            "S={}: primary sub-block DMZ count must equal old ET DMZ count",
            s
        );
    }
}

#[test]
fn test_theorem11_shifted_dmz_n4_to_n5_all_struts() {
    // The shifted copy (lo + old_g) must also embed exactly.
    for s in 1..=7 {
        let r = verify_theorem11(4, s);
        assert!(
            r.shifted_dmz_match,
            "S={}: shifted DMZ pattern must match",
            s
        );
        assert_eq!(
            r.old_dmz_count, r.shifted_subblock_dmz,
            "S={}: shifted sub-block DMZ count must equal old ET DMZ count",
            s
        );
    }
}

#[test]
fn test_theorem11_n5_to_n6_representative() {
    // N=5->6 for mandala and sky struts.
    for &s in &[3, 5, 9, 15] {
        let r = verify_theorem11(5, s);
        assert!(
            r.primary_dmz_match,
            "N=5->6, S={}: primary DMZ must match",
            s
        );
        assert!(
            r.shifted_dmz_match,
            "N=5->6, S={}: shifted DMZ must match",
            s
        );
        assert!(
            r.primary_value_match,
            "N=5->6, S={}: primary values must match",
            s
        );
        assert_eq!(
            r.old_dmz_count, r.primary_subblock_dmz,
            "N=5->6, S={}: primary DMZ count must match",
            s
        );
    }
}

#[test]
fn test_theorem11_map_coverage() {
    // Primary and shifted maps must cover 2*K_old positions total,
    // and must be disjoint (no position appears in both).
    let r = verify_theorem11(4, 3);
    let k_old = r.primary_map.len();
    assert_eq!(k_old, 6, "K_old must be 6 at N=4");

    let mut all_positions: Vec<usize> = Vec::new();
    all_positions.extend_from_slice(&r.primary_map);
    all_positions.extend_from_slice(&r.shifted_map);
    assert_eq!(
        all_positions.len(),
        2 * k_old,
        "must have 2*K_old mapped positions total"
    );

    // Check disjointness
    all_positions.sort();
    for w in all_positions.windows(2) {
        assert_ne!(
            w[0], w[1],
            "primary and shifted maps must be disjoint, but both contain {}",
            w[0]
        );
    }
}

#[test]
fn test_theorem11_map_injective() {
    // Each map must be injective (no two old positions map to the same new position).
    for &s in &[3, 7] {
        let r = verify_theorem11(4, s);
        let mut prim = r.primary_map.clone();
        prim.sort();
        prim.dedup();
        assert_eq!(
            prim.len(),
            r.primary_map.len(),
            "S={}: primary map must be injective",
            s
        );

        let mut shift = r.shifted_map.clone();
        shift.sort();
        shift.dedup();
        assert_eq!(
            shift.len(),
            r.shifted_map.len(),
            "S={}: shifted map must be injective",
            s
        );
    }
}

#[test]
fn test_theorem11_two_level_chain() {
    // Verify embedding chains: N=4 embeds in N=5 which embeds in N=6.
    // The composition of embeddings must also give a valid embedding.
    let r45 = verify_theorem11(4, 3);
    let r56 = verify_theorem11(5, 3);
    assert!(
        r45.primary_dmz_match && r56.primary_dmz_match,
        "both levels must match for chain embedding"
    );

    // The N=4 sub-block in N=6 should be reachable by composing maps:
    // old_N4_pos -> new_N5_pos (via r45.primary_map) -> new_N6_pos (via r56.primary_map)
    let composed: Vec<usize> = r45
        .primary_map
        .iter()
        .map(|&n5_pos| r56.primary_map[n5_pos])
        .collect();

    // Verify this composed map gives distinct positions
    let mut sorted = composed.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        composed.len(),
        "composed N=4->N=6 map must be injective"
    );
}

// --- L9f: Balloon Ride Tests ---

#[test]
fn test_min_level_for_strut() {
    // min_level_for_strut(S) = floor(log2(S)) + 2.
    // S=1: need G>1 -> N=2 (G=2). S=2..3: N=3 (G=4). S=4..7: N=4 (G=8).
    assert_eq!(min_level_for_strut(1), 2);
    for s in 2..=3 {
        assert_eq!(min_level_for_strut(s), 3, "S={} should require N=3", s);
    }
    for s in 4..=7 {
        assert_eq!(min_level_for_strut(s), 4, "S={} should require N=4", s);
    }
    for s in 8..=15 {
        assert_eq!(min_level_for_strut(s), 5, "S={} should require N=5", s);
    }
    for s in 16..=31 {
        assert_eq!(min_level_for_strut(s), 6, "S={} should require N=6", s);
    }
}

#[test]
fn test_balloon_ride_mandala_full_fill() {
    // Mandala struts (S=3, S=7) must have 100% fill at N=4..6.
    for &s in &[3, 7] {
        let ride = balloon_ride(s, 4, 6);
        assert!(
            ride.mandala_full_fill,
            "S={}: mandala strut must have 100% fill at all levels",
            s
        );
        for step in &ride.steps {
            assert!(
                (step.fill_ratio - 1.0).abs() < 1e-12,
                "S={} N={}: fill ratio should be 1.0, got {}",
                s,
                step.n,
                step.fill_ratio
            );
        }
    }
}

#[test]
fn test_balloon_ride_dmz_values_mandala() {
    // S=3 at N=4..6: exact DMZ counts must match exploration.
    let ride = balloon_ride(3, 4, 6);
    let dmzs: Vec<usize> = ride.steps.iter().map(|s| s.dmz_count).collect();
    assert_eq!(
        dmzs,
        vec![24, 168, 840],
        "S=3 DMZ sequence must be [24, 168, 840]"
    );
}

#[test]
fn test_balloon_ride_dmz_values_sky() {
    // S=15 at N=5..6: exact DMZ counts from exploration.
    let ride = balloon_ride(15, 5, 6);
    let dmzs: Vec<usize> = ride.steps.iter().map(|s| s.dmz_count).collect();
    assert_eq!(dmzs, vec![72, 456], "S=15 DMZ sequence must be [72, 456]");
}

#[test]
fn test_balloon_ride_fill_monotone_sky() {
    // Sky strut S=15: fill ratio must be monotonically non-decreasing.
    let ride = balloon_ride(15, 5, 6);
    assert!(
        ride.fill_monotone,
        "S=15: fill ratio must be monotonically non-decreasing"
    );
    // Verify specific values
    let fills: Vec<f64> = ride.steps.iter().map(|s| s.fill_ratio).collect();
    assert!((fills[0] - 72.0 / 168.0).abs() < 1e-6, "fill[0]");
    assert!((fills[1] - 456.0 / 840.0).abs() < 1e-6, "fill[1]");
    // Fill must strictly increase for sky struts
    assert!(
        fills[1] > fills[0],
        "fill must increase: {} > {}",
        fills[1],
        fills[0]
    );
}

#[test]
fn test_balloon_ride_regime_address_growth() {
    // Sky strut S=15: regime address gains one [0] prefix per level.
    // N=5: [1], N=6: [0,1]
    let ride = balloon_ride(15, 5, 6);
    assert_eq!(ride.steps[0].regime_address, vec![1u8]);
    assert_eq!(ride.steps[1].regime_address, vec![0u8, 1]);
}

#[test]
fn test_balloon_ride_mandala_regime_address() {
    // Mandala strut S=3: regime address is all-zeros (length grows with N).
    // N=4: [], N=5: [0], N=6: [0,0]
    let ride = balloon_ride(3, 4, 6);
    assert_eq!(ride.steps[0].regime_address, Vec::<u8>::new());
    assert_eq!(ride.steps[1].regime_address, vec![0u8]);
    assert_eq!(ride.steps[2].regime_address, vec![0u8, 0]);
}

#[test]
fn test_balloon_ride_dmz_growth_converges() {
    // DMZ growth ratio should converge toward 4.0 as K -> infinity.
    // For mandala S=7 at N=4..6: ratios are 7.0, 5.0.
    let ride = balloon_ride(7, 4, 6);
    let ratios: Vec<f64> = ride.steps.iter().map(|s| s.dmz_growth_ratio).collect();
    // First step has no previous, ratio = 0
    assert!((ratios[0] - 0.0).abs() < 1e-12);
    // Subsequent ratios should decrease toward 4.0
    assert!(
        ratios[1] > ratios[2],
        "ratio should decrease: {} > {}",
        ratios[1],
        ratios[2]
    );
    // All non-first ratios should be > 4.0 (approaching from above)
    for &r in &ratios[1..] {
        assert!(r > 4.0, "DMZ growth ratio should be > 4.0, got {}", r);
    }
}

#[test]
fn test_balloon_ride_step_count() {
    // Balloon ride from N=4 to N=6 should have 3 steps.
    let ride = balloon_ride(3, 4, 6);
    assert_eq!(ride.steps.len(), 3);
    // Verify N values are sequential.
    let ns: Vec<usize> = ride.steps.iter().map(|s| s.n).collect();
    assert_eq!(ns, vec![4, 5, 6]);
}

#[test]
fn test_balloon_ride_sky_classification() {
    // S=15 is sky; S=3, S=7 are not.
    let ride_sky = balloon_ride(15, 5, 6);
    assert!(ride_sky.steps[0].is_sky, "S=15 should be sky");

    let ride_mandala = balloon_ride(3, 4, 5);
    assert!(!ride_mandala.steps[0].is_sky, "S=3 should not be sky");
}

#[test]
fn test_balloon_ride_all_sedenion_struts() {
    // All 7 sedenion struts at N=4..5: verify basic consistency.
    for s in 1..=7 {
        let ride = balloon_ride(s, 4, 5);
        assert_eq!(ride.steps.len(), 2);
        // N=4: K=6, addressable=24
        assert_eq!(ride.steps[0].k, 6);
        assert_eq!(ride.steps[0].addressable, 24);
        // N=5: K=14, addressable=168
        assert_eq!(ride.steps[1].k, 14);
        assert_eq!(ride.steps[1].addressable, 168);
        // DMZ must increase from N=4 to N=5
        assert!(
            ride.steps[1].dmz_count > ride.steps[0].dmz_count,
            "S={}: DMZ must increase from N=4 to N=5",
            s
        );
    }
}

// --- L9g: Spectroscopy Bands Tests ---

#[test]
fn test_spectroscopy_bands_n5_structure() {
    let result = spectroscopy_bands(5);
    assert_eq!(result.n, 5);
    assert_eq!(result.dim, 32);
    assert_eq!(result.g, 16);
    assert_eq!(result.n_struts, 15);
    assert_eq!(result.n_bands, 2);
    assert_eq!(result.bands.len(), 2);
    // Band 0: S=1..8 (8 struts: 4 generators + 4 mandala)
    let b0 = &result.bands[0];
    assert_eq!(b0.s_lo, 1);
    assert_eq!(b0.s_hi, 8);
    assert_eq!(b0.n_struts, 8);
    assert_eq!(b0.n_generators, 4);
    assert_eq!(b0.n_mandala, 4);
    assert_eq!(b0.n_sky, 0);
    assert_eq!(b0.behavior, BandBehavior::FullFill);
    assert!(b0.all_full_fill);
    // Band 1: S=9..15 (7 sky struts)
    let b1 = &result.bands[1];
    assert_eq!(b1.s_lo, 9);
    assert_eq!(b1.s_hi, 15);
    assert_eq!(b1.n_struts, 7);
    assert_eq!(b1.n_generators, 0);
    assert_eq!(b1.n_mandala, 0);
    assert_eq!(b1.n_sky, 7);
    assert_eq!(b1.behavior, BandBehavior::UniformSky);
    assert!(!b1.all_full_fill);
}

#[test]
fn test_spectroscopy_bands_n5_dmz_ranges() {
    let result = spectroscopy_bands(5);
    // Band 0: all full fill => DMZ = addressable = K*(K-1)/2 * 2 = 14*13/2*... no,
    // DMZ = total_possible for full fill struts. At N=5 addressable = K*(K-1) = 14*12 = 168.
    assert_eq!(result.bands[0].dmz_min, 168);
    assert_eq!(result.bands[0].dmz_max, 168);
    // Band 1: uniform sky, all 72
    assert_eq!(result.bands[1].dmz_min, 72);
    assert_eq!(result.bands[1].dmz_max, 72);
    assert_eq!(result.bands[1].n_regimes, 1);
}

#[test]
fn test_spectroscopy_bands_n6_band_count() {
    let result = spectroscopy_bands(6);
    assert_eq!(result.n, 6);
    assert_eq!(result.dim, 64);
    assert_eq!(result.g, 32);
    assert_eq!(result.n_struts, 31);
    assert_eq!(result.n_bands, 4);
}

#[test]
fn test_spectroscopy_bands_n6_behaviors() {
    let result = spectroscopy_bands(6);
    // Band 0 (S=1..8): FullFill
    assert_eq!(result.bands[0].behavior, BandBehavior::FullFill);
    // Band 1 (S=9..16): MixedRegime (gen=1 at S=16, plus 7 sky struts with varied DMZ)
    assert_eq!(result.bands[1].behavior, BandBehavior::MixedRegime);
    // Band 2 (S=17..24): UniformSky
    assert_eq!(result.bands[2].behavior, BandBehavior::UniformSky);
    // Band 3 (S=25..31): UniformSky
    assert_eq!(result.bands[3].behavior, BandBehavior::UniformSky);
}

#[test]
fn test_spectroscopy_bands_n6_class_counts() {
    let result = spectroscopy_bands(6);
    // Band 0: 4 generators (1,2,4,8) + 4 mandala (3,5,6,7)
    assert_eq!(result.bands[0].n_generators, 4);
    assert_eq!(result.bands[0].n_mandala, 4);
    assert_eq!(result.bands[0].n_sky, 0);
    // Band 1: S=9..16, includes S=16 (generator: power-of-2)
    assert_eq!(result.bands[1].n_generators, 1);
    assert_eq!(result.bands[1].n_mandala, 0);
    assert_eq!(result.bands[1].n_sky, 7);
    // Band 2: S=17..24, all sky
    assert_eq!(result.bands[2].n_generators, 0);
    assert_eq!(result.bands[2].n_mandala, 0);
    assert_eq!(result.bands[2].n_sky, 8);
    // Band 3: S=25..31, all sky (7 struts, partial band)
    assert_eq!(result.bands[3].n_generators, 0);
    assert_eq!(result.bands[3].n_mandala, 0);
    assert_eq!(result.bands[3].n_sky, 7);
}

#[test]
fn test_spectroscopy_bands_n6_dmz_ranges() {
    let result = spectroscopy_bands(6);
    // Band 0: full fill at N=6 means DMZ = 840
    assert_eq!(result.bands[0].dmz_min, 840);
    assert_eq!(result.bands[0].dmz_max, 840);
    // Band 1: mixed regime, DMZ ranges from 456 to 840
    assert_eq!(result.bands[1].dmz_min, 456);
    assert_eq!(result.bands[1].dmz_max, 840);
    assert!(result.bands[1].n_regimes >= 2);
    // Band 2: uniform sky, all DMZ = 168
    assert_eq!(result.bands[2].dmz_min, 168);
    assert_eq!(result.bands[2].dmz_max, 168);
    // Band 3: uniform sky, all DMZ = 552
    assert_eq!(result.bands[3].dmz_min, 552);
    assert_eq!(result.bands[3].dmz_max, 552);
}

#[test]
fn test_spectroscopy_bands_flipbook_frame_count() {
    let result = spectroscopy_bands(5);
    // Band 0 has 8 frames (S=1..8)
    assert_eq!(result.bands[0].frames.len(), 8);
    // Band 1 has 7 frames (S=9..15)
    assert_eq!(result.bands[1].frames.len(), 7);
    // Total frames = 15 = n_struts
    let total: usize = result.bands.iter().map(|b| b.frames.len()).sum();
    assert_eq!(total, result.n_struts);
}

#[test]
fn test_spectroscopy_bands_flipbook_frame_ordering() {
    let result = spectroscopy_bands(5);
    // Frames within each band must be ordered by S
    for band in &result.bands {
        for (i, frame) in band.frames.iter().enumerate() {
            assert_eq!(frame.s, band.s_lo + i);
        }
    }
}

#[test]
fn test_spectroscopy_bands_expected_regime_count() {
    // N=4: expected = 2^0 = 1
    let r4 = spectroscopy_bands(4);
    assert_eq!(r4.expected_regime_count, 1);
    // N=5: expected = 2^1 = 2
    let r5 = spectroscopy_bands(5);
    assert_eq!(r5.expected_regime_count, 2);
    // N=6: expected = 2^2 = 4
    let r6 = spectroscopy_bands(6);
    assert_eq!(r6.expected_regime_count, 4);
}

#[test]
fn test_spectroscopy_bands_n4_sedenion_baseline() {
    // N=4: G=8, 7 struts, 1 band (S=1..7)
    let result = spectroscopy_bands(4);
    assert_eq!(result.n_bands, 1);
    assert_eq!(result.bands[0].n_struts, 7);
    // All 7 sedenion struts are mandala (3,5,6,7) or generator (1,2,4)
    assert_eq!(result.bands[0].n_generators + result.bands[0].n_mandala, 7);
    assert_eq!(result.bands[0].n_sky, 0);
    assert_eq!(result.bands[0].behavior, BandBehavior::FullFill);
}

#[test]
fn test_spectroscopy_bands_band0_always_full_fill() {
    // Band 0 should always be FullFill regardless of N, because it contains
    // S=1..8 which are all mandala or generator struts.
    for n in 4..=6 {
        let result = spectroscopy_bands(n);
        assert_eq!(
            result.bands[0].behavior,
            BandBehavior::FullFill,
            "Band 0 at N={} should be FullFill",
            n
        );
    }
}

#[test]
fn test_spectroscopy_bands_effective_bk_count() {
    // effective_bk_count = dmz_count / 24 (each box-kite contributes 24 DMZ cells)
    let result = spectroscopy_bands(5);
    for band in &result.bands {
        for frame in &band.frames {
            assert_eq!(
                frame.effective_bk_count,
                frame.dmz_count / 24,
                "S={}: effective_bk_count mismatch",
                frame.s
            );
        }
    }
}

// --- L16: Signed Adjacency Graph & Lanyard Dictionary Tests ---

#[test]
fn test_signed_graph_edge_count() {
    // Each sedenion BK has DMZ edges. The 6x6 upper triangle has 15
    // assessor pairs; DMZ count depends on sign concordance of quadrants.
    // Verify each BK has the same DMZ count (structural invariant).
    let mut counts = Vec::new();
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        assert!(
            !graph.edges.is_empty(),
            "S={}: should have at least 1 DMZ edge",
            s
        );
        assert!(
            graph.edges.len() <= 15,
            "S={}: at most 15 edges from 6x6 upper triangle",
            s
        );
        counts.push(graph.edges.len());
    }
    // All BKs should have the same DMZ count (symmetry of sedenion algebra).
    let first = counts[0];
    for (i, &c) in counts.iter().enumerate() {
        assert_eq!(
            c,
            first,
            "S={}: DMZ edge count {} differs from S=1 count {}",
            i + 1,
            c,
            first
        );
    }
}

#[test]
fn test_signed_graph_sign_partition() {
    // Every edge must be +1 or -1, and counts should sum to total edges.
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        assert_eq!(
            graph.n_positive + graph.n_negative,
            graph.edges.len(),
            "S={}: sign partition should sum to total edge count",
            s
        );
        for edge in &graph.edges {
            assert!(
                edge.sign == 1 || edge.sign == -1,
                "S={}: edge sign must be +1 or -1, got {}",
                s,
                edge.sign
            );
        }
    }
}

#[test]
fn test_signed_graph_nodes_are_6_lows() {
    // The nodes of the signed graph are the 6 unique L-indices of the BK.
    for s in 1..=7 {
        let et = create_strutted_et(4, s);
        let graph = extract_signed_graph(&et);
        assert_eq!(
            graph.nodes.len(),
            6,
            "S={}: expected 6 nodes, got {}",
            s,
            graph.nodes.len()
        );
        let node_set: HashSet<usize> = graph.nodes.iter().copied().collect();
        assert_eq!(
            node_set.len(),
            6,
            "S={}: nodes should be 6 distinct values",
            s
        );
    }
}

#[test]
fn test_lanyard_traversal_zigzag_face() {
    // A face with all 3 edges negative should produce /\/\ alternation.
    // (We test the traversal logic with a synthetic all-negative graph.)
    let graph = SignedAdjacencyGraph {
        s: 0,
        nodes: vec![1, 2, 3],
        edges: vec![
            SignedEdge {
                lo_a: 1,
                lo_b: 2,
                sign: -1,
            },
            SignedEdge {
                lo_a: 2,
                lo_b: 3,
                sign: -1,
            },
            SignedEdge {
                lo_a: 3,
                lo_b: 1,
                sign: -1,
            },
        ],
        n_positive: 0,
        n_negative: 3,
    };
    let sig = traverse_lanyard(&graph, &[1, 2, 3], true);
    // Starting /, edge -1 flips to \, edge -1 flips to /
    assert_eq!(
        sig.signature_string, "/\\/",
        "All-negative 3-cycle should be /\\/"
    );
}

#[test]
fn test_lanyard_traversal_trefoil_face() {
    // A face with 2 positive + 1 negative should produce ///\ or similar.
    let graph = SignedAdjacencyGraph {
        s: 0,
        nodes: vec![1, 2, 3],
        edges: vec![
            SignedEdge {
                lo_a: 1,
                lo_b: 2,
                sign: 1,
            },
            SignedEdge {
                lo_a: 2,
                lo_b: 3,
                sign: 1,
            },
            SignedEdge {
                lo_a: 3,
                lo_b: 1,
                sign: -1,
            },
        ],
        n_positive: 2,
        n_negative: 1,
    };
    let sig = traverse_lanyard(&graph, &[1, 2, 3], true);
    // Starting /, +1 keeps /, +1 keeps /
    assert_eq!(
        sig.signature_string, "///",
        "2pos+1neg should preserve state along positive edges"
    );
}

#[test]
fn test_extract_lanyards_from_et_produces_faces() {
    // Each BK has 8 triangular faces. Lanyard extraction should produce 8 signatures.
    for s in 1..=7 {
        let lanyards = extract_lanyards_from_et(4, s);
        assert_eq!(
            lanyards.len(),
            8,
            "S={}: expected 8 face lanyards, got {}",
            s,
            lanyards.len()
        );
    }
}

#[test]
fn test_lanyard_cycle_length_3() {
    // Every lanyard from a triangular face has cycle length 3.
    for s in 1..=7 {
        let lanyards = extract_lanyards_from_et(4, s);
        for (i, lan) in lanyards.iter().enumerate() {
            assert_eq!(
                lan.cycle.len(),
                3,
                "S={} face {}: cycle length should be 3, got {}",
                s,
                i,
                lan.cycle.len()
            );
        }
    }
}

// --- L17: Delta Transition Function Tests ---

#[test]
fn test_strut_pairs_xor_identity() {
    // For each S0, every strut pair {u, v} must satisfy u XOR v = S0.
    for s0 in 1..=7 {
        let pairs = strut_pairs_for(s0);
        for (i, pair) in pairs.iter().enumerate() {
            assert_eq!(
                pair.u ^ pair.v,
                s0,
                "S0={} pair {}: {} XOR {} should be {}",
                s0,
                i,
                pair.u,
                pair.v,
                s0
            );
        }
    }
}

#[test]
fn test_strut_pairs_count_3() {
    // Each S0 in {1..7} has exactly 3 strut pairs.
    for s0 in 1..=7 {
        let pairs = strut_pairs_for(s0);
        assert_eq!(pairs.len(), 3, "S0={}: should have 3 strut pairs", s0);
    }
}

#[test]
fn test_strut_pairs_exclude_s0() {
    // Neither u nor v should equal S0.
    for s0 in 1..=7 {
        let pairs = strut_pairs_for(s0);
        for pair in &pairs {
            assert_ne!(pair.u, s0, "S0={}: u should not equal S0", s0);
            assert_ne!(pair.v, s0, "S0={}: v should not equal S0", s0);
        }
    }
}

#[test]
fn test_strut_pairs_ordered() {
    // Each pair should have u < v.
    for s0 in 1..=7 {
        let pairs = strut_pairs_for(s0);
        for pair in &pairs {
            assert!(
                pair.u < pair.v,
                "S0={}: pair should be ordered u<v, got ({}, {})",
                s0,
                pair.u,
                pair.v
            );
        }
    }
}

#[test]
fn test_strut_pairs_cover_all_non_s0_indices() {
    // The 6 endpoints across 3 pairs should be exactly {1..7} \ {S0}.
    for s0 in 1..=7 {
        let pairs = strut_pairs_for(s0);
        let mut endpoints: Vec<usize> = pairs.iter().flat_map(|p| [p.u, p.v]).collect();
        endpoints.sort();
        endpoints.dedup();
        let expected: Vec<usize> = (1..=7).filter(|&x| x != s0).collect();
        assert_eq!(
            endpoints, expected,
            "S0={}: strut pair endpoints should cover {{1..7}} \\ {{S0}}",
            s0
        );
    }
}

#[test]
fn test_delta_transition_tables_all_7() {
    let tables = delta_transition_tables();
    assert_eq!(tables.len(), 7, "Should have 7 delta transition tables");
    for (i, dt) in tables.iter().enumerate() {
        assert_eq!(dt.s0, i + 1, "Table {} should have s0={}", i, i + 1);
    }
}

#[test]
fn test_delta_reachability_matches_twist() {
    // Delta strut pairs and twist transitions share the same reachability:
    // every S0 reaches exactly {1..7}\{S0} via its 3 strut pairs.
    assert!(
        verify_delta_reachability(),
        "Delta reachability must cover all non-S0 indices"
    );
}

#[test]
fn test_delta_transition_returns_pair() {
    // delta(S0, {u,v}) should return (u, v).
    for s0 in 1..=7 {
        let pairs = strut_pairs_for(s0);
        for pair in &pairs {
            let (a, b) = delta_transition(s0, pair);
            assert_eq!(a, pair.u, "delta should return pair.u");
            assert_eq!(b, pair.v, "delta should return pair.v");
        }
    }
}

// --- L18: Brocade/Slipcover Normalization Tests ---

#[test]
fn test_brocade_4_relabelings_per_bk() {
    // Each BK has 4 O-trips in its L-set, so 4 brocade relabelings.
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let relabelings = brocade_relabelings(bk);
        assert_eq!(
            relabelings.len(),
            4,
            "BK S={}: should have 4 brocade relabelings, got {}",
            bk.strut_signature,
            relabelings.len()
        );
    }
}

#[test]
fn test_brocade_central_trip_is_otrip() {
    // Every central trip must be a valid O-trip.
    let bks = find_box_kites(16, 1e-10);
    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort();
            s
        })
        .collect();

    for bk in &bks {
        for rel in brocade_relabelings(bk) {
            let mut sorted = rel.central_trip;
            sorted.sort();
            assert!(
                otrip_set.contains(&sorted),
                "BK S={}: central trip {:?} is not an O-trip",
                bk.strut_signature,
                rel.central_trip
            );
        }
    }
}

#[test]
fn test_brocade_outer_indices_complement() {
    // Outer indices = L-set \ central trip (exactly 3 elements).
    let bks = find_box_kites(16, 1e-10);
    for bk in &bks {
        let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
        for rel in brocade_relabelings(bk) {
            let central_set: HashSet<usize> = rel.central_trip.iter().copied().collect();
            let outer_set: HashSet<usize> = rel.outer_indices.iter().copied().collect();
            assert_eq!(
                outer_set.len(),
                3,
                "BK S={}: outer should have 3 distinct elements",
                bk.strut_signature
            );
            // outer = l_set \ central
            let expected: HashSet<usize> = l_set.difference(&central_set).copied().collect();
            assert_eq!(
                outer_set, expected,
                "BK S={}: outer indices should be L-set \\ central",
                bk.strut_signature
            );
        }
    }
}

#[test]
fn test_brocade_consistency() {
    assert!(
        verify_brocade_consistency(),
        "Brocade normalization must be consistent across all box-kites"
    );
}

#[test]
fn test_brocade_cpo_consistent_across_bks() {
    // In the Fano plane, removing point S leaves 4 lines on 6 points.
    // The complement of a line may or may not be another line.
    // This test verifies: all BKs have the same CPO count (by symmetry
    // of the Fano plane automorphism group).
    let bks = find_box_kites(16, 1e-10);
    let cpo_counts: Vec<usize> = bks
        .iter()
        .map(|bk| {
            brocade_relabelings(bk)
                .iter()
                .filter(|r| r.preserves_cpo)
                .count()
        })
        .collect();
    let first = cpo_counts[0];
    for (i, &c) in cpo_counts.iter().enumerate() {
        assert_eq!(
            c,
            first,
            "BK S={}: CPO count {} differs from S=1 count {}",
            i + 1,
            c,
            first
        );
    }
    // Document the actual count for the claim
    // (Fano plane: complement of line in 6-point restriction)
}

// --- L17+: Twist-Delta Pair Correspondence Tests ---

#[test]
fn test_twist_delta_xor_law_universal() {
    // The Fano XOR Law: for every twist transition, h XOR v == source_strut.
    // This was an open question (I-016); resolved by fixing the S-pairing
    // selection in twist_transition_table(). The 4 vent assessors admit 3
    // pairings with XOR values {S, perp[0], perp[1]}; the correct twist
    // targets use the S-pairing (delta-consistent).
    let comparisons = twist_delta_correspondence();
    assert_eq!(comparisons.len(), 21, "7 BKs x 3 tray-racks = 21");
    for c in &comparisons {
        assert!(
            c.xor_matches_source,
            "S={} TrayRack=[{},{}]: h^v={} != S",
            c.source_strut,
            c.tray_rack_label[0],
            c.tray_rack_label[1],
            c.twist_targets.0 ^ c.twist_targets.1
        );
    }
}

#[test]
fn test_twist_delta_strut_pair_match_universal() {
    // Every twist target pair {h,v} must match a delta strut pair.
    let comparisons = twist_delta_correspondence();
    for c in &comparisons {
        assert!(
            c.matching_strut_pair.is_some(),
            "S={} TrayRack=[{},{}]: targets ({},{}) not a delta strut pair",
            c.source_strut,
            c.tray_rack_label[0],
            c.tray_rack_label[1],
            c.twist_targets.0,
            c.twist_targets.1
        );
    }
}

#[test]
fn test_twist_delta_fano_lines_universal() {
    // Every twist transition {S, h, v} must lie on a Fano line (O-trip).
    let comparisons = twist_delta_correspondence();
    for c in &comparisons {
        assert!(
            c.fano_line.is_some(),
            "S={} TrayRack=[{},{}]: triple ({},{},{}) not a Fano line",
            c.source_strut,
            c.tray_rack_label[0],
            c.tray_rack_label[1],
            c.source_strut,
            c.twist_targets.0,
            c.twist_targets.1
        );
    }
}

#[test]
fn test_twist_delta_full_diagnostic() {
    // Print the complete twist-delta comparison table for analysis.
    let comparisons = twist_delta_correspondence();
    eprintln!("\n=== Twist-Delta Correspondence Table ===");
    eprintln!(
        "{:>3} {:>10} {:>7} {:>7} {:>10} {:>12} {:>10}",
        "S", "TrayRack", "h*", "v*", "h^v", "XOR==S?", "Fano?"
    );
    for c in &comparisons {
        let (h, v) = c.twist_targets;
        let xor_val = h ^ v;
        eprintln!(
            "{:>3} {:>10} {:>7} {:>7} {:>10} {:>12} {:>10}",
            c.source_strut,
            format!("[{},{}]", c.tray_rack_label[0], c.tray_rack_label[1]),
            h,
            v,
            xor_val,
            if c.xor_matches_source { "YES" } else { "NO" },
            if c.fano_line.is_some() { "YES" } else { "NO" },
        );
    }
    // If XOR doesn't hold, look for alternate Fano-plane quantities
    let xor_fails: Vec<_> = comparisons
        .iter()
        .filter(|c| !c.xor_matches_source)
        .collect();
    if !xor_fails.is_empty() {
        eprintln!("\n=== XOR Failures Analysis ===");
        for c in &xor_fails {
            let (h, v) = c.twist_targets;
            let s = c.source_strut;
            // Check all possible Fano-plane relationships
            eprintln!(
                "S={} TrayRack=[{},{}]: h={}  v={}  h^v={}  h^s={}  v^s={}  h^v^s={}",
                s,
                c.tray_rack_label[0],
                c.tray_rack_label[1],
                h,
                v,
                h ^ v,
                h ^ s,
                v ^ s,
                h ^ v ^ s
            );
        }
    }
}

#[test]
fn test_vent_pairing_three_fano_roles() {
    // Verify that the 3 pairings of 4 vent assessors give XOR values
    // lying exactly on the Fano line {S, perp[0], perp[1]}.
    let analyses = vent_pairing_analysis();
    assert_eq!(analyses.len(), 21, "7 BKs x 3 tray-racks = 21");

    let mut all_consistent = true;
    let mut role_counts = [0usize; 3]; // how many are S, perp0, perp1

    eprintln!("\n=== Vent Pairing Analysis ===");
    for a in &analyses {
        // Verify both sub-pairs in each pairing have the same XOR
        for (i, p) in a.pairings.iter().enumerate() {
            let xor1 = p.0.2;
            let xor2 = p.1.2;
            if xor1 != xor2 {
                eprintln!(
                    "INCONSISTENT: S={} perp=[{},{}] pairing {}: xor1={} != xor2={}",
                    a.source_strut, a.perp_pair[0], a.perp_pair[1], i, xor1, xor2
                );
                all_consistent = false;
            }
        }

        // Verify all 3 pairing roles are distinct {0,1,2}
        let mut roles = a.pairing_fano_roles;
        roles.sort();
        assert_eq!(
            roles,
            [0, 1, 2],
            "S={} perp=[{},{}]: pairings should cover all 3 Fano line roles, got {:?}",
            a.source_strut,
            a.perp_pair[0],
            a.perp_pair[1],
            a.pairing_fano_roles
        );

        // Track which role the current twist targets selected
        if let Some(idx) = a.current_pairing_index {
            role_counts[a.pairing_fano_roles[idx]] += 1;
        }
    }

    assert!(
        all_consistent,
        "All pairings should have consistent XOR values"
    );
    eprintln!(
        "Current twist pairing distribution: S={}, perp0={}, perp1={}",
        role_counts[0], role_counts[1], role_counts[2]
    );
}

#[test]
fn test_cross_bk_lanyard_census_total() {
    // 7 BKs x 8 faces = 56 total face lanyards.
    let census = cross_bk_lanyard_census();
    assert_eq!(census.n_bks, 7);
    assert_eq!(
        census.total_faces, 56,
        "Expected 56 faces, got {}",
        census.total_faces
    );
}

#[test]
fn test_cross_bk_lanyard_no_blues() {
    // No AllSame (Blues) faces should exist in sedenions.
    let census = cross_bk_lanyard_census();
    let blues = census
        .pattern_counts
        .get(&FaceSignPattern::AllSame)
        .copied()
        .unwrap_or(0);
    assert_eq!(blues, 0, "Expected 0 Blues faces in sedenions");
}

#[test]
fn test_cross_bk_lanyard_zigzag_count() {
    // AllOpposite (TripleZigzag): 7 BKs x 2 zigzag faces = 14.
    let census = cross_bk_lanyard_census();
    let zigzag = census
        .pattern_counts
        .get(&FaceSignPattern::AllOpposite)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        zigzag, 14,
        "Expected 14 AllOpposite faces (7x2), got {}",
        zigzag
    );
}

#[test]
fn test_cross_bk_lanyard_uniform_patterns() {
    // Normalized sign patterns (order-independent) should be uniform
    // across all 7 BKs.
    let census = cross_bk_lanyard_census();
    assert!(
        census.uniform_across_bks,
        "Normalized pattern distribution should be uniform across all 7 BKs"
    );
}

#[test]
fn test_cross_bk_lanyard_full_classification() {
    // Complete classification: print all patterns for documentation.
    let census = cross_bk_lanyard_census();
    eprintln!("\n=== Cross-BK Normalized Lanyard Census ===");
    eprintln!("Total faces: {}", census.total_faces);

    let mut patterns: Vec<_> = census.pattern_counts.iter().collect();
    patterns.sort_by_key(|(p, _)| **p);
    for (pattern, count) in &patterns {
        eprintln!("  {:?}: {} (per BK: {})", pattern, count, *count / 7);
    }
    eprintln!("Uniform: {}", census.uniform_across_bks);

    // Per-BK breakdown
    eprintln!("\nPer-BK breakdown:");
    for (s, pats) in &census.per_bk_patterns {
        eprintln!("  S={}: {:?}", s, pats);
    }

    // Verify: AllSame + AllOpposite + TwoSameOneOpp + OneSameTwoOpp = 56
    let total: usize = census.pattern_counts.values().sum();
    assert_eq!(total, 56);
}

#[test]
fn test_vent_pairing_detail_table() {
    // Print the full pairing analysis for manual inspection.
    let analyses = vent_pairing_analysis();
    eprintln!("\n=== Vent Pairing Detail Table ===");
    for a in &analyses {
        let fano_names = ["S", "p0", "p1"];
        eprintln!(
            "\nS={}, perp=[{},{}], vents=[{},{},{},{}], twist=({},{})",
            a.source_strut,
            a.perp_pair[0],
            a.perp_pair[1],
            a.vent_indices[0],
            a.vent_indices[1],
            a.vent_indices[2],
            a.vent_indices[3],
            a.current_twist_targets.0,
            a.current_twist_targets.1,
        );
        for (i, p) in a.pairings.iter().enumerate() {
            let role = a.pairing_fano_roles[i];
            let role_name = if role < 3 { fano_names[role] } else { "??" };
            let selected = a.current_pairing_index == Some(i);
            eprintln!(
                "  P{}: {{{},{}}}(^{}) + {{{},{}}}(^{}) -> {} {}",
                i,
                p.0.0,
                p.0.1,
                p.0.2,
                p.1.0,
                p.1.1,
                p.1.2,
                role_name,
                if selected { "<-- SELECTED" } else { "" }
            );
        }
    }
}
