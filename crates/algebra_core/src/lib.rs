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

pub mod analysis;
pub mod construction;
pub mod error;
pub mod experimental;
pub mod lie;
pub mod physics;
pub mod universal_algebra;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use error::{AlgebraError, AlgebraResult};
pub use universal_algebra::UniversalAlgebra;

// Re-export core algebra functions from construction
pub use construction::cayley_dickson::{
    batch_associator_norms,
    batch_associator_norms_parallel,
    batch_associator_norms_sq,
    cd_associator,
    cd_associator_norm,
    cd_basis_mul_sign, // Integer-exact basis product sign
    cd_conjugate,
    cd_multiply,
    cd_multiply_simd,
    cd_norm_sq,
    cd_norm_sq_simd, // SIMD-accelerated versions
    count_pathion_zero_divisors,
    find_zero_divisors,
    left_mult_operator,
    measure_associator_density,
    zd_spectrum_analysis,
};

pub use construction::wheels::{canonical_test_set, verify_carlstrom_axioms, WheelQ};

pub use construction::padic::{
    abs_p, cantor_function_on_cantor, check_ultrametric, is_dyadic, is_power_of_two,
    padic_distance, ternary_digits_power3, vp, vp_int, CantorDigits, Rational,
};

pub use construction::hypercomplex::{
    AlgebraDim, HypercomplexAlgebra, OctonionFieldDynamics, PathionAlgebra, ZeroDivisorResults,
    ZeroSearchConfig,
};

// Re-export from physics
pub use physics::clifford::{
    gamma_matrices_cl8, pauli_matrices, verify_clifford_relation, CliffordAlgebra, GammaMatrix,
};

pub use physics::octonion_field::{
    build_structure_constants, evolve, force, gaussian_wave_packet, hamiltonian,
    measure_dispersion, noether_charges, oct_conjugate, oct_multiply, oct_norm_sq, standing_wave,
    stormer_verlet_step, DispersionResult, EvolutionResult, FieldParams, Octonion, FANO_TRIPLES,
};

pub use physics::m3::{classify_m3, compute_m3_octonion_basis, M3Classification, OctonionTable};

pub use physics::billiard_sim::{
    BilliardConfig, BilliardState, BounceResult, ConstraintDiagnostics, HyperbolicBilliard,
    LorentzVec,
};

// Re-export from analysis
pub use analysis::zd_graphs::{
    analyze_associator_graph,
    analyze_basis_participation,
    analyze_zd_graph,
    build_associator_graph,
    build_mixed_blade_graph,
    build_zd_interaction_graph,
    // XOR-balanced search extension (CX-003)
    enumerate_xor_balanced_4tuples,
    even_parity_sign_vectors,
    xor_balanced_four_tuple,
    xor_bucket_necessary_2v4,
    xor_bucket_necessary_for_two_blade,
    // XOR heuristics (cd_xor_heuristics port)
    xor_key,
    xor_necessity_statistics,
    xor_pairing_buckets,
    zd_graph_diameter,
    zd_shortest_path,
    zero_product_2blade_x_4blade,
    AssociatorGraphResult,
    BasisParticipationResult,
    BladeNode,
    FourBladeSpec,
    MixedBladeGraphResult,
    TwoBladeSpec,
    ZdGraphAnalysis,
};

pub use analysis::boxkites::{
    // Production rules and automorphemes (de Marrais 2000, 2004)
    all_diagonal_zero_products,
    analyze_box_kite_symmetry,
    are_coassessors,
    automorpheme_assessors,
    automorphemes,
    automorphemes_containing_assessor,
    build_coassessor_graph,
    canonical_strut_table,
    compute_strut_signature,
    cross_assessors,
    diagonal_zero_product,
    diagonal_zero_products_exact,
    edge_sign_type,
    find_box_kites,
    find_connected_components,
    motif_components_for_cross_assessors,
    primitive_assessors,
    production_rule_1,
    production_rule_2,
    production_rule_3,
    Assessor,
    BoxKite,
    BoxKiteSymmetryResult,
    // Generalized motif census (cd_motif_census port)
    CrossPair,
    EdgeSignType,
    MotifComponent,
    StrutTable,
    O_TRIPS,
};

pub use analysis::annihilator::{
    annihilator_info, find_left_annihilator_vector, is_reggiani_zd, is_zero_divisor,
    left_multiplication_matrix, nullspace_basis, right_multiplication_matrix, AnnihilatorInfo,
};

pub use analysis::reggiani::{
    assert_standard_zero_divisor_annihilators, standard_zero_divisor_partners,
    standard_zero_divisors, StandardZeroDivisor,
};

pub use analysis::grassmannian::{
    chordal_distance, count_distinct_distances, geodesic_distance, orthonormality_error,
    pairwise_geodesic_distances, principal_angles, subspace_from_orthonormal,
    subspace_from_vectors, Subspace,
};

pub use analysis::fractal_analysis::{
    analyze_multiple_series, calculate_hurst, classify_hurst, dfa_analysis, generate_fbm,
    generate_fgn, hurst_rs_analysis, DfaResult, HurstClassification, HurstResult,
    MultiSeriesHurstResult, RescaledRangeResult,
};

pub use analysis::stochastic::{
    analyze_anomalous_diffusion,
    fit_ou_parameters,
    generate_gbm,
    generate_levy_flight,
    generate_ou_process,
    // Anomalous diffusion analysis
    AnomalousDiffusionResult,
    DiffusionType,
    // Geometric Brownian Motion
    GBMParams,
    // Levy flights
    LevyParams,
    MeanReversionResult,
    // Ornstein-Uhlenbeck process
    OUParams,
};

pub use analysis::homotopy_algebra::{
    a_infinity_sign,
    catalan_number,
    cyclohedron_vertices,
    // Sign computations
    koszul_sign,
    l_infinity_sign,
    // A-infinity structures
    AInfinityAlgebra,
    // Combinatorics
    Associahedron,
    BVInfinityAlgebra,
    // Core types
    Degree,
    FormalityMorphism,
    GradedElement,
    HomotopyAlgebraType,
    HomotopyOperation,
    // L-infinity structures
    LInfinityAlgebra,
    MasseyProduct,
    MinimalAInfinity,
    // String field theory
    StringFieldTheory,
    StringType,
};

pub use analysis::projective_geometry::{
    component_xor_label,
    find_affine_class_predicate,
    find_boolean_class_predicate,
    find_linear_class_predicate,
    incidence_matrix,
    map_components_to_pg,
    pg,
    pg_correspondence_summary,
    pg_from_cd_dim,
    sign_twist_signature,
    verify_pg_correspondence,
    verify_pg_line_structure,
    verify_signature_determines_solutions,
    // C-444 correspondence verification
    PGCorrespondenceResult,
    PGLine,
    PGPoint,
    ProjectiveGeometry,
};

// Re-export from lie
pub use lie::e8_lattice::{
    compute_e8_inner_products,
    cross_validate_with_atlas,
    e8_cartan_matrix,
    e8_weyl_group_order,
    exceptional_groups_from_atlas,
    generate_e8_roots,
    get_atlas_embedding_info,
    magic_square_entry,
    verify_cartan_matrix_with_atlas,
    // Atlas-E8 integration
    AtlasE8CrossValidation,
    AtlasEmbeddingInfo,
    // Freudenthal-Tits magic square
    DivisionAlgebra,
    E8Lattice,
    E8Root,
    ExceptionalGroupsFromAtlas,
    FreudenthalTitsMagicSquare,
    MagicSquareLieAlgebra,
};

pub use lie::group_theory::{
    exceptional, is_prime, order_alternating, order_gl, order_psl2_q, order_sl, order_symmetric,
    prime_power, PSL_2_7_ORDER,
};

pub use lie::nilpotent_orbits::{
    dominance_order, enumerate_partitions, jordan_block, jordan_type_nilpotent,
    matrix_from_jordan_type, nilpotency_index, partition_count, JordanType, NilpotentAnalysis,
};

pub use lie::kac_moody::{
    // Classical series
    a_n_cartan,
    d_n_cartan,
    e10_cartan,
    e11_cartan,
    // E-series Cartan matrices
    e8_cartan,
    e9_cartan,
    CartanEntry,
    // Dynkin diagrams
    DynkinDiagram,
    DynkinEdge,
    DynkinNode,
    E10RootSystem,
    E11RootSystem,
    E9RootSystem,
    ESeriesRootSystem,
    // Core types
    GeneralizedCartanMatrix,
    // Extended E-series root systems
    KacMoodyRoot,
    KacMoodyRootSystem,
    KacMoodyType,
    LieAlgebraType,
    RootType,
    // Weyl groups and root systems
    WeylGroupInfo,
};

// Re-export from experimental
pub use experimental::moonshine::{
    compute_j_coefficients, j_as_hauptmodul, j_constant_term_e8_relation,
    known_moonshine_decompositions, mckay_e8_observation, monster_group_order,
    moonshine_dimensions, verify_monster_order_factorization, verify_moonshine_c1,
    verify_moonshine_c2, HauptmodulProperty, MoonshineDecomposition, MoonshineDimensions,
    J_COEFFICIENTS, J_COEFFICIENTS_VALID, LEECH_LATTICE_DIMENSION, MONSTER_CONJUGACY_CLASSES,
    MONSTER_REPS_VALID, MONSTER_REP_DIMENSIONS, NIEMEIER_LATTICE_COUNT,
};

pub use experimental::e10_octonion::{
    build_e8_transition_graph,
    build_fano_overlap_graph,
    claim4_summary,
    compare_8x8_graphs,
    compare_fano_dynkin,
    compare_fano_transitions,
    default_cayley_basis,
    describe_fano_structure,
    dynkin_fano_correspondence,
    dynkin_fano_null_summary,
    e8_dynkin_adjacency,
    exact_pvalue,
    extract_3windows,
    fano_complement,
    fano_complement_table,
    fano_completion_rate,
    fano_enrichment_zscore,
    optimal_cayley_basis,
    optimal_fano_mapping,
    optimal_fano_overlap_basis,
    simple_root_products,
    symmetrize_transition_graph,
    verify_cayley_integer_norms,
    verify_claim4,
    // Cayley integer bridge
    CayleyBasis,
    // Claim 4 verification (Engine B)
    Claim4Result,
    // Fano overlap graph (Engine B)
    FanoOverlapGraph,
    GraphEdgeComparison,
    NULL_FANO_RATE_UNIFORM,
};

pub use experimental::billiard_stats::{
    chi_squared_e8_transitions,
    claim1_summary,
    compute_locality_metrics,
    compute_sector_metrics,
    fano_analysis_report,
    fano_structure_analysis,
    fano_structure_analysis_from_sequence,
    generate_null_sequence,
    permutation_test_mi,
    permutation_test_r_e8,
    stationary_distribution,
    // Transition matrix analysis
    transition_matrix,
    verify_claim1,
    // Chi-squared test
    ChiSquaredResult,
    // Claim 1 verification
    Claim1Result,
    // Fano structure analysis
    FanoStructureAnalysis,
    // Locality metrics
    LocalityMetrics,
    // Null models
    NullModel,
    // Permutation tests
    PermutationTestResult,
    SectorMetrics,
    // Constants
    E10_ADJACENCY,
    NULL_R_E8_UNIFORM,
    N_E8,
    N_WALLS,
};

pub use experimental::emanation::{
    balloon_ride,
    // CDP signed product (integer-exact)
    cdp_signed_product,
    classify_strut,
    create_skybox,
    create_strutted_et,
    et_regimes,
    // Regime spectroscopy
    et_sparsity_spectroscopy,
    // ET construction
    generate_tone_row,
    hide_fill_analysis,
    is_inherited_full_fill_strut,
    // Sky classification (de Marrais erratum resolved)
    is_sky_strut,
    min_level_for_strut,
    // (s,g)-Modularity -- recursive regime address
    regime_address,
    regime_count,
    row_degree_distribution,
    sail_decomposition,
    spectroscopy_bands,
    strut_spectroscopy,
    // Trip-Count Two-Step
    trip_count,
    trip_count_two_step,
    verify_theorem11,
    verify_three_viziers,
    vizier_xor_audit,
    BalloonRide,
    // Balloon ride -- fixed-S, increasing-N sequence
    BalloonRideStep,
    // Spectroscopy bands -- fixed-N, all-S band structure
    BandBehavior,
    ClassifiedFace,
    // Sail decomposition (face classification)
    FaceRole,
    FlipBookFrame,
    HideFillResult,
    // Hide/Fill involution -- row-degree invariance
    RowDegreeDistribution,
    SailDecomposition,
    Skybox,
    // Skybox -- label-line extension for recursion
    SkyboxCell,
    SpectroscopyBand,
    SpectroscopyResult,
    // Strut spectroscopy (detailed classification)
    StrutClass,
    StrutSpectroscopyEntry,
    StrutSpectrum,
    StruttedEmanationTable,
    StruttedEtCell,
    // Theorem 11 -- recursive ET embedding
    Theorem11Result,
    // Three Vizier XOR relationships (de Marrais 2007)
    ThreeVizierResult,
    // Core ET types
    ToneRow,
    VizierXorAudit,
};

pub use experimental::cd_external;
pub use experimental::so7_drift;

// Re-export external algebra crates for convenience
pub use padic as ext_padic;
pub use wheel as ext_wheel;
