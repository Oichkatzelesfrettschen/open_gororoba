//! algebra_core: Cayley-Dickson algebras, Clifford algebras, and algebraic structures.
//!
//! This crate provides high-performance implementations of:
//! - Cayley-Dickson multiplication for any power-of-2 dimension
//! - Associator computation and batch operations
//! - Zero-divisor search algorithms
//! - Clifford algebra Cl(8) for particle physics
//! - E8 lattice and root system computations
//! - Kac-Moody algebras (E9, E10, E11 extensions)
//! - Box-kite symmetry structures (de Marrais)
//! - Monstrous Moonshine (j-function and Monster group)
//!
//! # Literature
//! - de Marrais (2000): Box-kite structure of sedenion zero-divisors
//! - Furey et al. (2024): Cl(8) -> 3 generations
//! - Reggiani (2024): Geometry of sedenion zero divisors
//! - Kac (1990): Infinite-Dimensional Lie Algebras
//! - Damour, Henneaux, Nicolai (2002): E10 and M-theory

pub mod cayley_dickson;
pub mod clifford;
pub mod zd_graphs;
pub mod e8_lattice;
pub mod boxkites;
pub mod annihilator;
pub mod reggiani;
pub mod m3;
pub mod octonion_field;
pub mod wheels;
pub mod padic;
pub mod group_theory;
pub mod fractal_analysis;
pub mod nilpotent_orbits;
pub mod hypercomplex;
pub mod moonshine;
pub mod kac_moody;
pub mod homotopy_algebra;
pub mod stochastic;
pub mod grassmannian;
pub mod so7_drift;
pub mod projective_geometry;
pub mod mult_table;
pub mod cd_external;
pub mod emanation;
pub mod e10_octonion;
pub mod billiard_stats;

// Re-export core algebra functions
pub use cayley_dickson::{
    cd_multiply, cd_conjugate, cd_norm_sq, cd_associator, cd_associator_norm,
    cd_multiply_simd, cd_norm_sq_simd,  // SIMD-accelerated versions
    batch_associator_norms, batch_associator_norms_sq, batch_associator_norms_parallel,
    left_mult_operator, find_zero_divisors, measure_associator_density,
    zd_spectrum_analysis, count_pathion_zero_divisors,
    cd_basis_mul_sign,  // Integer-exact basis product sign
};

pub use clifford::{
    pauli_matrices, gamma_matrices_cl8, verify_clifford_relation,
    GammaMatrix, CliffordAlgebra,
};

pub use zd_graphs::{
    build_zd_interaction_graph, analyze_zd_graph, analyze_basis_participation,
    build_associator_graph, analyze_associator_graph, zd_shortest_path,
    zd_graph_diameter, ZdGraphAnalysis, BasisParticipationResult,
    AssociatorGraphResult,
    // XOR heuristics (cd_xor_heuristics port)
    xor_key, xor_bucket_necessary_for_two_blade, xor_balanced_four_tuple,
    xor_pairing_buckets, xor_bucket_necessary_2v4,
    // XOR-balanced search extension (CX-003)
    enumerate_xor_balanced_4tuples, even_parity_sign_vectors,
    zero_product_2blade_x_4blade, build_mixed_blade_graph,
    xor_necessity_statistics,
    BladeNode, MixedBladeGraphResult, TwoBladeSpec, FourBladeSpec,
};

pub use e8_lattice::{
    E8Lattice, E8Root, generate_e8_roots, e8_cartan_matrix,
    e8_weyl_group_order, compute_e8_inner_products,
    // Atlas-E8 integration
    AtlasE8CrossValidation, AtlasEmbeddingInfo, ExceptionalGroupsFromAtlas,
    cross_validate_with_atlas, get_atlas_embedding_info,
    verify_cartan_matrix_with_atlas, exceptional_groups_from_atlas,
    // Freudenthal-Tits magic square
    DivisionAlgebra, MagicSquareLieAlgebra, FreudenthalTitsMagicSquare,
    magic_square_entry,
};

pub use boxkites::{
    Assessor, BoxKite, find_box_kites, analyze_box_kite_symmetry,
    BoxKiteSymmetryResult, primitive_assessors, are_coassessors,
    diagonal_zero_product, build_coassessor_graph, find_connected_components,
    compute_strut_signature,
    // Production rules and automorphemes (de Marrais 2000, 2004)
    all_diagonal_zero_products, EdgeSignType, edge_sign_type,
    O_TRIPS, production_rule_1, production_rule_2, production_rule_3,
    automorpheme_assessors, automorphemes, automorphemes_containing_assessor,
    StrutTable, canonical_strut_table,
    // Generalized motif census (cd_motif_census port)
    CrossPair, cross_assessors, diagonal_zero_products_exact,
    MotifComponent, motif_components_for_cross_assessors,
};

pub use annihilator::{
    AnnihilatorInfo, left_multiplication_matrix, right_multiplication_matrix,
    nullspace_basis, annihilator_info, is_zero_divisor, is_reggiani_zd,
    find_left_annihilator_vector,
};

pub use reggiani::{
    StandardZeroDivisor, standard_zero_divisors,
    standard_zero_divisor_partners, assert_standard_zero_divisor_annihilators,
};

pub use m3::{
    OctonionTable, compute_m3_octonion_basis,
    M3Classification, classify_m3,
};

pub use octonion_field::{
    Octonion, FieldParams, EvolutionResult, DispersionResult,
    FANO_TRIPLES, build_structure_constants,
    oct_multiply, oct_conjugate, oct_norm_sq,
    hamiltonian, force, stormer_verlet_step, noether_charges,
    evolve, gaussian_wave_packet, standing_wave, measure_dispersion,
};

pub use grassmannian::{
    Subspace, subspace_from_vectors, subspace_from_orthonormal,
    principal_angles, geodesic_distance, chordal_distance,
    pairwise_geodesic_distances, count_distinct_distances, orthonormality_error,
};

pub use wheels::{
    WheelQ, verify_carlstrom_axioms, canonical_test_set,
};

pub use padic::{
    Rational, CantorDigits,
    vp_int, vp, abs_p, is_power_of_two, is_dyadic,
    ternary_digits_power3, cantor_function_on_cantor,
    padic_distance, check_ultrametric,
};

pub use group_theory::{
    order_psl2_q, order_symmetric, order_alternating,
    order_gl, order_sl, is_prime, prime_power,
    PSL_2_7_ORDER, exceptional,
};

pub use fractal_analysis::{
    HurstResult, RescaledRangeResult, DfaResult, HurstClassification,
    MultiSeriesHurstResult,
    calculate_hurst, hurst_rs_analysis, dfa_analysis, classify_hurst,
    analyze_multiple_series, generate_fgn, generate_fbm,
};

pub use nilpotent_orbits::{
    JordanType, NilpotentAnalysis,
    nilpotency_index, jordan_type_nilpotent, jordan_block,
    matrix_from_jordan_type, enumerate_partitions, partition_count,
    dominance_order,
};

pub use hypercomplex::{
    AlgebraDim, ZeroSearchConfig, ZeroDivisorResults,
    HypercomplexAlgebra, OctonionFieldDynamics, PathionAlgebra,
};

pub use moonshine::{
    J_COEFFICIENTS, J_COEFFICIENTS_VALID,
    MONSTER_REP_DIMENSIONS, MONSTER_REPS_VALID,
    MONSTER_CONJUGACY_CLASSES, LEECH_LATTICE_DIMENSION, NIEMEIER_LATTICE_COUNT,
    monster_group_order, verify_monster_order_factorization,
    verify_moonshine_c1, verify_moonshine_c2,
    compute_j_coefficients, known_moonshine_decompositions,
    j_constant_term_e8_relation, moonshine_dimensions,
    j_as_hauptmodul, mckay_e8_observation,
    MoonshineDecomposition, MoonshineDimensions, HauptmodulProperty,
};

pub use kac_moody::{
    // Core types
    GeneralizedCartanMatrix, CartanEntry, KacMoodyType, LieAlgebraType,
    // Dynkin diagrams
    DynkinDiagram, DynkinNode, DynkinEdge,
    // E-series Cartan matrices
    e8_cartan, e9_cartan, e10_cartan, e11_cartan,
    // Classical series
    a_n_cartan, d_n_cartan,
    // Weyl groups and root systems
    WeylGroupInfo, KacMoodyRootSystem,
    // Extended E-series root systems
    KacMoodyRoot, RootType,
    E9RootSystem, E10RootSystem, E11RootSystem, ESeriesRootSystem,
};

pub use homotopy_algebra::{
    // Core types
    Degree, GradedElement, HomotopyAlgebraType, HomotopyOperation,
    // A-infinity structures
    AInfinityAlgebra, MinimalAInfinity, MasseyProduct,
    // L-infinity structures
    LInfinityAlgebra, BVInfinityAlgebra, FormalityMorphism,
    // Combinatorics
    Associahedron, catalan_number, cyclohedron_vertices,
    // Sign computations
    koszul_sign, a_infinity_sign, l_infinity_sign,
    // String field theory
    StringFieldTheory, StringType,
};

pub use projective_geometry::{
    PGPoint, PGLine, ProjectiveGeometry,
    pg, pg_from_cd_dim, incidence_matrix,
    component_xor_label, map_components_to_pg, verify_pg_line_structure,
    find_linear_class_predicate, find_affine_class_predicate,
    find_boolean_class_predicate,
    sign_twist_signature, verify_signature_determines_solutions,
};

pub use stochastic::{
    // Ornstein-Uhlenbeck process
    OUParams, generate_ou_process, fit_ou_parameters, MeanReversionResult,
    // Geometric Brownian Motion
    GBMParams, generate_gbm,
    // Levy flights
    LevyParams, generate_levy_flight,
    // Anomalous diffusion analysis
    AnomalousDiffusionResult, DiffusionType, analyze_anomalous_diffusion,
};

pub use e10_octonion::{
    fano_complement, fano_complement_table, extract_3windows,
    fano_completion_rate, optimal_fano_mapping, exact_pvalue,
    fano_enrichment_zscore, describe_fano_structure,
    NULL_FANO_RATE_UNIFORM,
    // Cayley integer bridge
    CayleyBasis, default_cayley_basis, verify_cayley_integer_norms,
    simple_root_products, dynkin_fano_correspondence,
    optimal_cayley_basis, dynkin_fano_null_summary,
};

pub use billiard_stats::{
    // Constants
    E10_ADJACENCY, N_WALLS, N_E8, NULL_R_E8_UNIFORM,
    // Locality metrics
    LocalityMetrics, compute_locality_metrics,
    SectorMetrics, compute_sector_metrics,
    // Null models
    NullModel, generate_null_sequence,
    // Permutation tests
    PermutationTestResult, permutation_test_r_e8, permutation_test_mi,
    // Transition matrix analysis
    transition_matrix, stationary_distribution,
};

pub use emanation::{
    // Core ET types
    ToneRow, StruttedEtCell, StruttedEmanationTable, StrutSpectrum,
    // ET construction
    generate_tone_row, create_strutted_et,
    // Regime spectroscopy
    et_sparsity_spectroscopy, et_regimes,
    // CDP signed product (integer-exact)
    cdp_signed_product,
    // Trip-Count Two-Step
    trip_count, trip_count_two_step,
    // Sky classification (de Marrais erratum resolved)
    is_sky_strut, is_inherited_full_fill_strut,
    // Sail decomposition (face classification)
    FaceRole, ClassifiedFace, SailDecomposition, sail_decomposition,
    // Strut spectroscopy (detailed classification)
    StrutClass, StrutSpectroscopyEntry, classify_strut, strut_spectroscopy,
    // (s,g)-Modularity -- recursive regime address
    regime_address, regime_count,
    // Hide/Fill involution -- row-degree invariance
    RowDegreeDistribution, row_degree_distribution,
    HideFillResult, hide_fill_analysis,
    // Skybox -- label-line extension for recursion
    SkyboxCell, Skybox, create_skybox,
    // Theorem 11 -- recursive ET embedding
    Theorem11Result, verify_theorem11,
    // Balloon ride -- fixed-S, increasing-N sequence
    BalloonRideStep, BalloonRide, min_level_for_strut, balloon_ride,
    // Spectroscopy bands -- fixed-N, all-S band structure
    BandBehavior, FlipBookFrame, SpectroscopyBand, SpectroscopyResult,
    spectroscopy_bands,
    // Three Vizier XOR relationships (de Marrais 2007)
    ThreeVizierResult, verify_three_viziers,
    VizierXorAudit, vizier_xor_audit,
};

// Re-export external algebra crates for convenience
pub use wheel as ext_wheel;
pub use padic as ext_padic;
