//! # Extracted text: idris2_linear_types.pdf
//!
//! - source_root: `/home/eirikr/Documents/AGL_Library/Theorem_Provers_Documentation`
//! - source_relpath: `idris2_linear_types.pdf`
//! - source_abs: `/home/eirikr/Documents/AGL_Library/Theorem_Provers_Documentation/idris2_linear_types.pdf`
//! - detected_kind: `pdf`
//! - extracted_at_utc: `2026-01-02T17:31:31+00:00`
//! - pages: `11`
//! - title: ``
//! - author: ``
//! - subject: ``
//! - keywords: ``
//! - creation_date: `Wed Jun 23 17:40:05 2021 PDT`
//! - mod_date: `Wed Jun 23 17:40:05 2021 PDT`
//! - encrypted: `no`
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~text
//! Epicyclic frequencies in static and spherically symmetric wormhole geometries
//! Vittorio De Falco1,2 ,∗ Mariafelicia De Laurentis3,2,4 ,† and Salvatore Capozziello2,3,4,5‡
//! 1
//!
//! arXiv:2106.12564v1 [gr-qc] 23 Jun 2021
//!
//! Department of Mathematics and Applications “R. Caccioppoli”,
//! University of Naples Federico II, Via Cintia, 80126 Naples, Italy,
//! 2
//! Istituto Nazionale di Fisica Nucleare, Sezione di Napoli,
//! Complesso Universitario di Monte S. Angelo, Via Cintia Edificio 6, 80126 Napoli, Italy
//! 3
//! Università degli studi di Napoli “Federico II”, Dipartimento di Fisica “Ettore Pancini”,
//! Complesso Universitario di Monte S. Angelo, Via Cintia Edificio 6, 80126 Napoli, Italy
//! 4
//! Lab.Theor.Cosmology,Tomsk State University of Control Systems and Radioelectronics(TUSUR), 634050 Tomsk, Russia
//! 5
//! Scuola Superiore Meridionale, Università di Napoli “Federico II”, Largo San Marcellino 10, 80138 Napoli, Italy
//! (Dated: June 24, 2021)
//! The measurement of the epicyclic frequencies is a widely used astrophysical technique to infer
//! information on a given self-gravitating system and on the related gravity background. We derive
//! their explicit expressions in static and spherically symmetric wormhole spacetimes. We discuss how
//! these theoretical results can be applied to: (1) detect the presence of a wormhole, distinguishing it
//! by a black hole; (2) reconstruct wormhole solutions through the fit of the observational data, once
//! we have them. Finally, we discuss the physical implications of our proposed epicyclic method.
//! PACS numbers: 04.20.Dw, 04.70as, 04.25.dg
//! Keywords: Physics of black holes, alternative gravity, wormhole.
//!
//! I.
//!
//! INTRODUCTION
//!
//! A wormhole (WH) can be intuitively seen as a topological shortcut-structure, which capable of connecting
//! two distinct spacetime points. A visual representation
//! of a WH is obtained through the example of drawing
//! two separate points on a paper sheet, and then considering as the shortest connecting-trajectory not the joining straight line, but the bent paper which brings the
//! two points one over the other. In Fig. 1, we sketch
//! UPPER REGION
//!
//! THROAT
//!
//! {
//!
//! NECK
//!
//! WORMHOLE
//!
//! LOWER REGION
//!
//! FIG. 1. Geometrical representation of a WH.
//!
//! the intuitive picture reported before. A WH is conceptually defined as a compact object characterized by no
//! horizons and physical singularities, and endowed with a
//! traversable bridge, dubbed WH neck, connecting two universes or two different regions of the same spacetime [1].
//! These objects have been extensively studied in the literature, indeed there are many authors, who not only
//!
//! ∗ vittorio.defalco@physics.cz
//! † mariafelicia.delaurentis@unina.it
//! ‡ capozziello@unina.it
//!
//! built up new WH solutions, both in General Relativity
//! (GR) and in Alternative Gravity, but they were also interested in analysing their properties [2–8]. On the other
//! hand, there is also a major research effort in conceiving original astrophysical strategies to search for WH
//! observational signatures [9–19]. This research topics is
//! strongly motivated not only by the presence of several
//! complementary data, but also because there is the great
//! opportunity to perform, now and in the near-future,
//! highly-precise observations in strong field regimes. This
//! is a very crucial point, because the missed detection of
//! WHs can be so far explained by the fact that gravity has
//! not been investigated in extreme regimes. This justifies
//! also the possibility to find a particular subclass of WH solutions, known in the literature as black hole (BH) mimickers, which perfectly mimic all observational properties
//! of a BH with arbitrary accuracy [20] and their signature
//! may be likely revealed by strong gravity experiments.
//! In this respect, an important role can be played by the
//! epicyclic frequencies. Let us assume that a particle moving in a closed orbit is disturbed by small perturbations in
//! the radial, azimuthal, and polar directions. The particle
//! oscillation frequencies, along the above-mentioned directions, correspond to the epicyclic frequencies {νr , νϕ , νθ },
//! respectively. These quantities reveal some useful features
//! [21, 22]: (1) strong dependence on the underlying spacetime geometry; (2) production of observational effects in
//! strong field regime; (3) direct possibility in measuring
//! them with actual and near future data. At the best of
//! our knowledge, there are only two papers on this subject
//! applied to WHs, which are: (1) Chakraborty and collaborators, who studied the behaviour of a test gyroscope
//! moving towards a Teo rotating traversable WH [23]; (2)
//! Deligianni and coauthors, who focused their attention on
//! quasi-periodic oscillations (QPOs) from an accretion disk
//!
//!
//! --- PAGE BREAK ---
//! 2
//! around Teo rotating traversable WHs [24].
//! In this work, we adopt the above-cited advantages of
//! the epicyclic frequencies to elaborate an astrophysical
//! procedure to both observationally unearth WHs and to
//! identify the most appropriate WH solution/s to fit the
//! observational data. Our analysis concentrates on static
//! and spherically symmetric WH metrics. The article is organized as follows: in Sec. II, we summarize our modelindependent approach framed in generic static and spherically symmetric WH geometries and then derive the formulas of the epicyclic frequencies; in Sec. III we apply
//! the approach to both observationally detect the presence
//! of a WH and to reconstruct the WH solutions through
//! the fit of the observational data; finally, in Sec. IV, we
//! discuss the obtained results and draw the conclusions.
//!
//! II.
//!
//! WORMHOLE EPICYCLIC FREQUENCIES
//!
//! In this section, we first describe the general theory in
//! which the WH solutions are framed (see Sec. II A). As
//! second step, we consider the timelike geodesic equations
//! (see Sec. II B), and then finally derive the general expressions of the epicyclic frequencies (see Sec. II C).
//!
//! A.
//!
//! The Morris-Thorne-like wormhole metrics
//!
//! A generic static and spherically symmetric WH can be
//! described by the Morris-Thorne-like metric [25], ds2 =
//! gαβ dxα dxβ , which in geometrical units (G = c = 1), in
//! spherical coordinates (t, r, θ, ϕ), and set in the equatorial plane θ = π/2 (without loss of generality due to the
//! spherical symmetry hypothesis) reads as
//! ds2 = −e2Φ(r) dt2 +
//!
//! dr2
//! + r2 dϕ2 ,
//! 1 − b(r)/r
//!
//! (1)
//!
//! where Φ(r) and b(r) are the redshift and shape functions, respectively. Eq. (1) is a two-parameters family of
//! metrics, fully determined once Φ(r) and b(r) are known,
//! and it represents also a class of solutions valid both in
//! GR and in several Alternative Gravity theories, with the
//! further request to be traversable and stable. Φ(r) and
//! b(r) can be assigned in two ways: (1) a-priori, meaning that a new theoretical WH solution has been found;
//! (2) a-posteriori, referring to the fact that they can be
//! reconstructed through the fit of the observational data
//! (approach followed in this paper).
//! We require that these WH solutions satisfy the following properties [25]: (1) Φ(r) and b(r) are real smooth
//! functions, and Φ(r) is everywhere finite, since there is
//! the absence of horizons and essential singularities; to
//! define a finite proper radial distance l, we have that
//! (1 − b(r)/r) ≥ 0; (3) the flaring outward condition [26–
//! 29] requires that b0 (r) < b(r)/r near and on the throat.
//! It defines the minimum radius such that rmin = b0 and
//! b(rmin ) = b0 ; (4) asymptotic flatness, namely b(r)/r → 0
//!
//! and Φ(r) → 0 for r → +∞; (5) the WH traversability,
//! which depending on the gravity framework, can be obtained by considering exotic matter (especially in GR)
//! [30–32] or topological defects (mainly in alternative and
//! extended theories of gravity) [33–35]; (6) the mass M is
//! defined, according to the Arnowitt, Deser, Misner (ADM)
//! formalism, as the total mass of the system contained in
//! the whole spacetime [1]:
//! Z ∞
//! c2 b0
//! 2
//! M ≡ lim m(r) =
//! ρ(x)x2 dx.
//! + 4πc
//! (2)
//! r→+∞
//! 2G
//! b0
//! B.
//!
//! Timelike geodesic equations
//!
//! A test particle moving around any self-gravitating object, in particular a WH, and affected only by gravity
//! follows a timelike geodesic equation, which is
//! dxβ dxγ
//! d2 xα
//! + Γα
//! = 0,
//! βγ
//! 2
//! dτ
//! dτ dτ
//!
//! (3)
//!
//! where τ is the affine parameter (proper time) along the
//! test particle trajectory, and Γα
//! βγ are the Christoffel symbols. We prefer to adopt the relativity of observer splitting formalism, which permits to clearly distinguish between gravitational and inertial contributions (see [36–
//! 41], for more details). This approach is equivalent to
//! Eq. (3), but it has the great advantage that has a direct
//! connection with the classical description, allowing to understand the physics behind the symbols we algebraically
//! manipulate. We are also aware that there are more direct
//! formulas to calculate the epicyclic frequencies (see Refs.
//! [23, 24], for details), but we deem however pedagogical
//! to present the full derivation.
//! The approach can be formulated considering the presence of two observers: (1) one static located at infinity,
//! corresponding to our telescopes and detectors with which
//! we normally perform observations, and measurements in
//! astrophysics; (2) local static observers (LSOs), in which
//! it is more easy performing the calculations. A proper
//! reference frame adapted to the LSOs is given by the orthonormal basis of vectors [17, 25]
//! r
//! ∂t
//! ∂ϕ
//! b(r)
//! (4)
//! et̂ = Φ(r) , er̂ = ∂r 1 −
//! , eϕ̂ =
//! .
//! r
//! r
//! e
//! We will denote throughout the paper, vector, and tensor
//! indices (e.g., vα ; Tαβ ) evaluated in the LSO frame by a
//! hat (e.g., vα̂ ; Tα̂β̂ ), whereas scalar quantities (e.g., f ) are
//! followed by n (e.g., f (n)). A test particle moves with
//! a timelike four-velocity U and a spatial velocity ν(U, n)
//! with respect to the LSO frames, which both read as [17]
//! U = γ[et̂ + ν],
//! ν = ν(sin αer̂ + cos αeϕ̂ ),
//! (5)
//! √
//! where γ = 1/ 1 − ν 2 is the Lorentz factor, ν = ||ν|| is
//! the magnitude of the test particle spatial velocity, and α
//! is the azimuthal angle of the vector ν measured clockwise
//! from the positive ϕ̂ direction in the LSO frame.
//!
//!
//! --- PAGE BREAK ---
//! 3
//! An important role is played by the LSO kinematical quantities, which are: the acceleration a(n)r̂ , being
//! the general relativistic gravitational attraction along the
//! radial direction, and the relative Lie curvature vector
//! k(Lie) (n)r̂ , corresponding to the general relativistic centrifugal force along the radial direction. Their explicit
//! expressions are [17]
//! r
//! b(r)
//! r̂
//! 0
//! ,
//! (6)
//! a(n) = Φ (r) 1 −
//! r
//! r
//! b(r)
//! 1
//! 1−
//! k(Lie) (n)r̂ = −
//! .
//! (7)
//! r
//! r
//!
//! is the Keplerian angular velocity. It is easy to check
//! that X0 is an equilibrium configuration of Eqs. (8) –
//! (10), namely f (X0 ) = 0. Therefore, we consider a small
//! perturbation ε  1 around X0 given by
//!
//! where 0 = d/dr. The components of the test particle
//! acceleration a(U ) can be calculated as [17, 37, 38, 41]
//!
//! Therefore, we explicitly obtain
//! s
//! b(r0 ) Φ(r0 ) 0
//! dν1
//! = α1 1 −
//! e
//! Φ (r0 ) [r0 Φ0 (r0 ) − 1] , (18)
//! dt
//! r0
//! q
//! (
//! 0 ) Φ(r0 )
//! 1 − b(r
//! dα1
//! r0 e
//! 2ν1
//! (19)
//! =
//! dt
//! r0
//! )
//! [Φ0 (r0 ) + r0 Φ00 (r0 )]
//! p
//! ,
//! −r1
//! r0 Φ0 (r0 )
//! s
//! dr1
//! b(r0 ) Φ(r0 ) p
//! = α1 1 −
//! e
//! r0 Φ0 (r0 ).
//! (20)
//! dt
//! r0
//!
//! dν
//! ,
//! dτ
//! a(U )r̂ = γ 2 [a(n)r̂ + k(Lie) (n)r̂ ν 2 cos2 α]
//! 
//! 
//! dν
//! dα
//! 2
//! +γ γ sin α
//! + ν cos α
//! ,
//! dτ
//! dτ
//! a(U )t̂ = γ 2 ν sin α a(n)r̂ + γ 3 ν
//!
//! a(U )ϕ̂ = −γ 2 ν 2 sin α cos αk(Lie) (n)r̂
//! 
//! 
//! dα
//! dν
//! 2
//! − ν sin α
//! +γ γ cos α
//! .
//! dτ
//! dτ
//!
//! (8)
//!
//! (9)
//!
//! (10)
//!
//! The geodesics equations (3) corresponds to a(U ) = 0.
//! Using Eqs. (8) – (9) together with the radial component
//! of Eq. (5), we obtain the test particle equations of motion, described in terms of the following set of coupled
//! ordinary differential equations of first order [17]
//! eΦ(r) sin α
//! dν
//! =−
//! a(n)r̂ ,
//! dt
//! γ2
//! dα
//! eΦ(r) cos α
//! =−
//! [a(n)r̂ + k(Lie) (n)r̂ ν 2 ],
//! dt
//! ν
//! r
//! b(r)
//! dr
//! Φ(r)
//! =e
//! ν sin α 1 −
//! ,
//! dt
//! r
//!
//! (11)
//! (12)
//! (13)
//!
//! where we have written them in terms of the coordinate
//! time t by using the time component of Eq. (5)
//! dt
//! γ
//! = Φ(r) .
//! dτ
//! e
//! C.
//!
//! (14)
//!
//! ν = νK + εν1 ,
//!
//! α = εα1 ,
//!
//! r = r0 + εr1 ,
//!
//! (16)
//!
//! or also X = X0 + εX1 , with X1 = (ν1 , α1 , r1 ). Linearizing the dynamical system, we obtain
//! 
//! 
//! dX1
//! ∂fi
//! = A · X1 ,
//! Aij =
//! .
//! (17)
//! dt
//! ∂Xj X=X0
//!
//! To calculate the radial epicyclic angular velocity, we must
//! differentiate Eq. (19) and then use Eqs. (18) – (20),
//! which implies the following harmonic oscillator equation
//! d2 α1
//! + Ω2r α1 = 0,
//! dt2
//!
//! (21)
//!
//! where the radial epicyclic angular velocity Ωr is
//! 
//! 
//! 3Φ0 (r0 )
//! 2
//! 2Φ(r0 ) b(r0 ) − r0
//! 00
//! 02
//! Ωr = e
//! − Φ (r0 ) .
//! 2Φ (r0 ) −
//! r0
//! r0
//! (22)
//! Since the equations of motion are rotationally invariant due to the spherical symmetry, we have that the
//! azimuthal epicyclic angular velocity Ωϕ is equal to the
//! Keplerian angular velocity ΩK , given by [18]
//! s
//! dϕ
//! eΦ(r0 ) νK
//! Φ0 (r0 )
//! ΩK ≡
//! =
//! = eΦ(r0 )
//! .
//! (23)
//! dt
//! r0
//! r0
//!
//! The epicyclic frequencies
//!
//! The explicit formulas of the epicyclic frequencies
//! {νr = Ωr /(2π), νϕ = Ωϕ /(2π)} can be calculated in
//! terms of the epicyclic angular velocities {Ωr , Ωϕ }. Defining X = (ν, α, r), the dynamical system given by Eqs.
//! (8) – (10) can be written as dX/dt = f (X). We consider a stable circular orbit X0 = (νK , 0, r0 ), where
//! s
//! νK ≡
//!
//! −
//!
//! p
//! a(n)r̂
//! = rΦ0 (r),
//! r̂
//! k(Lie) (n)
//!
//! (15)
//!
//! III.
//!
//! APPLICATIONS
//!
//! The epicyclic frequencies assume a prominent role
//! in X-ray binaries, which are double systems typically
//! formed by a BH (or a neutron star) which is gravitationally bounded to its companion star. They are usually characterized by two distinctive features [42]: (1) the
//! presence of an accretion disk formed around the compact
//! object, which emits in all energy bands of the electromagnetic spectrum, especially with more brightness in
//!
//!
//! --- PAGE BREAK ---
//! 4
//! the X-rays owed to the radiation coming from the matter inflow in the innermost regions; (2) the appearance
//! of significant flux variabilities on long and much shorter
//! times-scales. The former can be appreciated on long-term
//! light-curves and imply significant changes in the energy
//! spectra as reported in the X-ray hardness-intensity diagrams; whereas the latter cannot be studied by investigating the light-curve, and, for this reason, the Fourier
//! analysis is commonly employed through power-density
//! spectra to reveal very fast aperiodic and quasi-periodic
//! variabilities. About last point, a feature observed in almost all kinds of accreting systems is the existence of narrow peaks with a distinct centroid frequency, well known
//! in the literature as QPOs (see Refs. [21, 22], for reviews).
//! QPOs are usually associated with accretion-related
//! time-scales and to certain effects of strong gravity on
//! the motion of matter around massive compact objects.
//! Their study is extremely relevant, because they represent an astrophysical mean to explore the accretion flow
//! around BHs in an alternative approach not accessible via
//! energy spectra alone and can provide also indirect tests
//! of gravity within/without GR theory [21]. Although they
//! are strong and easily measurable signals, their physical
//! origin remains still matter of debate. However, many
//! models have been proposed so far to explain the origin
//! and the evolution of QPOs in X-ray binaries, which contributed thus to increase our understanding toward their
//! observational and theoretical characteristics.
//! An interesting aspect of all QPO models relies on the
//! fact that they share an extensive use of the epicyclic
//! frequencies through disparate theoretical treatments to
//! describe the matter motions in the vicinity of BHs.
//! Therefore, we can generally state that the observations
//! of the epicyclic frequencies can be associated with QPO
//! measurements. The detection of this phenomenon involves also general relativistic light bending effects in the
//! strongly curved BH spacetime, and polarization measurements, which allows to distinguish between the different
//! proposed QPO models [43]. The acquisition of the observational data can be performed by the actual telescopes,
//! like Rossi X-ray Timing Explorer (RXTE) [44], and by
//! near-future space-missions, like LOFT (Large Observatory for X-ray Timing) [45], eXTP (Enhanced X-ray Timing and Polarization mission) [46], IXPE (Imaging X-ray
//! Polarimetry Explorer) [47].
//! The aim of this digression is critical to raise the awareness on the strong observational power of the epicyclic
//! frequencies and on the consequent great possibility to
//! experimentally achieve the objectives proposed in this
//! paper. This section is dedicated to the applications of
//! the results obtained in Sec. II. We first show how to detect WHs through epicyclic frequencies (see Sec. III A)
//! and then we illustrate how to reconstruct a WH solution
//! through the observational data (see Sec. III B).
//!
//! A.
//!
//! Wormhole’s detection: deviations from a
//! Schwarzschild black hole
//!
//! To determine the presence of a WH, we should be able
//! to detect metric-departures from the Schwarzschild BH
//! geometry. A direct and simple approach to achieve this
//! goal can be performed by comparing the epicyclic frequencies of the Schwarzschild spacetime with those detected. Since we do not have yet observational data, in
//! Table I we selected, from the literature, different WH
//! solutions framed both in GR and in some Alternative
//! Theories of Gravity, which can be considered straightforward extensions of GR [35, 48–52]. Below, we sketch
//! their main features in view of WH detection.
//! (i) Metric Theories: In this class of theories, the variable describing the gravitational field is the metric
//! tensor. A purely metric Lagrangian, linearly depending on the Ricci scalar (the Einstein-Hilbert
//! Larangian), is necessary to have a second order
//! dynamics. Considering non-linear combinations of
//! curvature invariants give higher–order Lagrangians
//! with fourth-order field equations as, e.g., the socalled f (R) gravity. Adopting different forms of
//! f , it is possible to address a wide range of significant phenomena at infrared scales, like: clustering
//! of structures and accelerated expansion of the Hubble flow. The key-feature of this approach relies in
//! solving the dark side problem through geometry. It
//! exploits the more degrees of freedom of the gravitational field to model the constituents of dark energy
//! and dark matter, without searching for new exotic
//! material components, but standard perfect fluids
//! can be notwithstanding employed.
//! (ii) Metric–Affine Theories: This class of theories concerns a generalisation of the metric approach, because it considers metric and connection as independent fields, allowing thus the matter to couple
//! not only with metric, but also with connection. In
//! addition, some of these theories can be formulated
//! relaxing the hypothesis of metricity, which practically means considering the Equivalence Principle at the foundation of gravitational interaction
//! and then the coincidence of the causal and geodesic
//! structures of the spacetime. An example of this formalism is represented by the Palatini formulation,
//! where metric and affine connections are not necessarily related through the Levi-Civita connection.
//! (iii) Teleparallel Theories: The central features of this
//! class of theories are: (i) the Lagrangian of the gravitational field is a function f of the torsion scalar T ;
//! (ii) the Weitzenböck connection, is adopted instead
//! of the Levi-Civita connection; (iii) a geodesic structure is not necessary but dynamics and kinematic
//! are ruled by affinities. This formulation shares a
//! deep analogy with GR, because the field equations,
//! written in terms of T , can be rearranged in the same
//!
//!
//! --- PAGE BREAK ---
//! 5
//! way of those expressed in terms of R. However, despite of the fact that Teleparallel Equivalent General Relativity (TEGR) practically coincide with
//! GR from the point of view of the field equations,
//! discrepancies emerges for the metric f (R) formulation and f (T ) gravity. They are: (1) the f (R) field
//! equations are of fourth-order, whereas the f (T )
//! field equations remain of second-order (as it also
//! occurs in GR); (2) the dynamical variables in the
//! teleparallel gravity are the tetrad fields eµα , while,
//! in the metric formulation, this role is fulfilled by
//! the metric tensor gµν . This means the breaking of
//! the Lorentz invariance and other differences which
//! emerge in the two formulations.
//! It is important to note that the procedure and calculations we have performed in Secs. II B and II C to derive
//! the epicyclic frequencies in generic WH spacetimes are
//! valid not only for metric theories of gravity, but also for
//! metric-affine and teleparallel models. Indeed, the latter
//! two theories can be reduced to equivalent metric theories,
//! where the further degrees of freedom of the geometrical
//! background are encoded in the metric potentials Φ(r)
//! and b(r) (see Refs. [48, 51], for more details).
//! Before showing the method, we recall that the
//! Schwarzschild metric can be defined by Eq. (1) through
//! 
//! 
//! 1
//! 2M
//! Φ(r) = log 1 −
//! ,
//! b(r) = 2M.
//! (24)
//! 2
//! r
//! In a Schwarzschild BH spacetime, the innermost stable
//! circular orbit (ISCO) radius is rISCO = 6M , while the
//! two epicyclic angular velocities are [53]
//! s 
//! s
//! 
//! M
//! 6M
//! M
//! G
//! G
//! Ωr =
//! 1−
//! .
//! (25)
//! ,
//! Ωϕ =
//! 3
//! r0
//! r0
//! r03
//!
//! 6M (see #1 and #2). These preliminary crossed observations could reveal already at the beginning whether
//! indications of WH existence can be present. To better
//! analyse the selected WH solutions, we consider both the
//! difference and the ratio of the epicyclic frequencies with
//! respect to the Schwarzschild known profiles, see Fig. 3.
//! We take into account both possibilities to quantify how
//! much the data approach closer to the BH physics and to
//! have an estimate of their order of magnitude.
//! From Fig. 3, we can deduce: (1) there exists some
//! WH geometries that does not admit either one or both
//! epicyclic frequencies (see #6, #7, #8, #10), or even they
//! admit constant zero values (see #3, #5, because they
//! have constant redshift function); (2) there are some WH
//! solutions, which perfectly mimic the BH trends (see
//! #1, #2, #9 in the lower panels of Fig. 3), and others
//! get closer to the BH profiles at larger radii (see #7 in
//! the lower panels of Fig. 3). For the last cases, it would
//! be important to detect the epicyclic frequencies closer to
//! the BH ISCO, where their discrepancies are higher.
//! Another way to infer information consists in considering the precession frequency νper = νϕ − νr [56]. In Fig.
//! 4, we plot these frequencies only for those WH geometries admitting both epicyclic frequencies. These measurements allow to set tighter constraints on the collected
//! data, and robustly understanding the presence of a WH.
//!
//! B.
//!
//! Procedure for reconstructing the WH solution
//! from the fit of the observational data
//!
//! In this section we propose a method to reconstruct the
//! WH solution through the fit of the data. We first present
//! the theoretical strategy (see Sec. III B 1) and then applies
//! it to some simulated data (see Sec. III B 2).
//!
//! Regarding static and spherically symmetric WHs, we
//! already calculated the epicyclic angular velocities (22)
//! and (23). The ISCO radius can be calculated through
//! the formula (see Ref. [18], for details)
//! L2z [Φ0 (r)r − 1] + Φ0 (r)r3 = 0,
//!
//! (26)
//!
//! where Lz is the conserved angular momentum with respect to z-axis, orthogonal to the equatorial plane, along
//! the test particle trajectory. Equation (26), solved for
//! the lowest value of Lz , permits to determine rISCO . In
//! Fig. 2 we show the trend of the epicyclic angular velocities of some WH solutions reported in Table I, namely
//! only those that admit real values. We include also the
//! Schwarzschild values in order to show how some WH solutions can closely emulate the BH behaviour.
//! The practical way to disclose possible departures from
//! BH geometries through the acquired data is by first measuring the mass of the compact object M (see Refs.
//! [54, 55], for details), which then implies where the ISCO
//! radius is located for a BH. As shown in Table I there are
//! some WH solutions having an ISCO radius different from
//!
//! 1.
//!
//! Theoretical strategy
//!
//! We assume that the epicyclic frequencies {νr , νϕ } (or
//! {Ωr , Ωϕ }) are measured in the range r ∈ [r1 , r2 ]. We note
//! that Eq. (23) can be arranged in the following way
//! Ω2ϕ (r) = e2Φ(r)
//!
//! Φ0 (r)
//! r
//!
//! ⇒
//!
//! h
//!
//! e2Φ(r)
//!
//! i0
//!
//! = 2rΩ2ϕ (r). (27)
//!
//! Integrating both members of the above equation between
//! [r1 , r] with r ∈ [r1 , r2 ], we obtain
//! 
//! 
//! Z r
//! 1
//! 2Φ(r1 )
//! 2
//! Φ(r) = log e
//! +
//! 2xΩϕ (x)dx .
//! (28)
//! 2
//! r1
//! Since the term e2Φ(r1 ) is unknown, it can be calculated exploiting different (and complementary) techniques, like: measuring the photon impact parameter
//! bph (r1 ), the photon emission angle αE , and the position
//! r1 at which the photon has been emitted it is possible
//!
//!
//! --- PAGE BREAK ---
//! 6
//!
//! Legend
//!
//! 0.05
//!
//! #0
//!
//! #1
//!
//! #2
//!
//! #4
//!
//! #9
//!
//! #10
//!
//! 1
//!
//! 0.04
//!
//! 0.03
//!
//! ΩK (M-1)
//!
//! Ωr (M-1)
//!
//! Legend
//!
//! 0.500
//!
//! 0.02
//!
//! #0
//!
//! #1
//!
//! #2
//!
//! #4
//!
//! #6
//!
//! #7
//!
//! #9
//!
//! 0.100
//! 0.050
//!
//! 0.010
//! 0.005
//!
//! 0.01
//!
//! 0.001
//!
//! 0.00
//! 5
//!
//! 10
//! r (M)
//!
//! 15
//!
//! 20
//!
//! 5
//!
//! 10
//! r (M)
//!
//! 15
//!
//! 20
//!
//! FIG. 2. Plots of the radial (left panel ) and azimuthal (right panel ) epicyclic angular frequencies in terms of the radius r of
//! some WH solutions, whose numbers in the legends refer to those reported in Table I. The dashed black line represents the
//! Schwarzschild BH case.
//! TABLE I. We report some examples of WH solutions in different gravity frameworks. In the first row, the parameters related
//! to a Schwarzschild BH are reported, where b0 , only in this case, stays for the event horizon radius.
//! Theory
//! Schwarzschild
//!
//! #
//! 0
//! 1
//!
//! GR
//! 2
//!
//! Φ(r)
//! 1
//! 2 log
//!
//! b0
//!
//! − 3M
//! r
//!
//! 5
//! 6
//!
//! 9M 2
//! r
//!
//! 0
//!
//! 4
//!
//! − 12 log
//!
//! 0
//! h
//!
//! rISCO
//!
//! 
//!
//! 2M
//! 2M
//! 
//! q 
//! 
//! ( √
//!  1 log 1 − 1
//! M ≤ r ≤ 3M r 3
//! M
//! ≤
//! r
//! ≤
//! 3M
//! 2
//! 3
//! 
//! 
//! √
//! √
//! M
//!  1 log 1 − 3M
//! 3M 3M ≤ r
//! 3M ≤ r
//! 2
//! r
//! (
//! (
//! 1
//! 7
//! 9
//! 3M ≤ r ≤ 4M
//! 3M ≤ r ≤ 4M
//! 2 log 16
//! 16 r
//! 
//! 3M
//! 1
//! 9M
//! 9
//! log
//! 1
//! −
//! 4M
//! ≤
//! r
//! 4M ≤ r
//! 2
//! 4r
//! 4M
//!
//! 3
//!
//! METRIC
//!
//! b(r)
//!
//! 1 − 2M
//! r
//!
//! r
//! er/M −1
//! 3
//!
//! M
//! r2
//!
//! 
//! M 2.5
//! r
//!
//! i
//!
//! r
//!
//! 0.83
//!
//! 3M
//!
//! Ωr (r)
//! q
//!
//! 6M
//!
//! NOa
//!
//! M
//!
//! 6M
//!
//! M
//!
//! NO
//!
//! M
//! M
//!
//! NO
//!
//! 1
//! c
//! 2 log[A(r)]
//!
//! r[1 − A(r)Ω2+ (r/M )]
//!
//! M
//!
//! NO
//!
//! 10
//!
//! −M
//! r2
//! M2
//! r
//!
//! M
//!
//! (∗)
//! (∗)
//!
//! 0.71
//!
//! 3
//!
//! (
//!
//! r
//!
//! 6M 
//!
//! e− M − r
//!
//! 3e
//!
//! q
//!
//! r
//!
//! NO
//!
//! 0.79
//! √
//!
//! 3
//! − M3
//! r
//!
//! q
//!
//! 3≤r
//!
//! 0. q
//! 1.06 M
//! r3
//!
//! 4≤r
//!
//! √
//!
//! 3
//!
//! 3M
//!
//! 3e− r
//!
//! M
//! r3
//!
//! √
//!
//! (∗)
//!
//! 6M
//! e− r M (r−6M )(r 3 +M 3 )
//! r7
//!
//! M4
//! √
//! √
//! r 3 r 2 −M 2 [2r ( r 2 −M 2 +r )−M 2 ]
//!
//! (∗)
//!
//! r
//!
//! 3
//!
//! −M
//! r3
//!
//! [59]
//! [60]
//!
//! 1.12r 0.25
//! M 1.25
//!
//! −M 3 (r−M )(r+M )(r 3 +6M 3 )
//! r5
//!
//! [57]
//! [58]
//!
//! q
//!
//! 0
//!
//! r 0.33 (M 0.17 −r 0.17 )
//! M 2.5
//!
//! [57]
//!
//! 3≤r≤4
//!
//! 0
//!
//! 
//! r
//! e M −e M (r−6M )
//!
//! Ω+ (r)2 (A(r)(3.A0 (r)+1.rA00 (r))−2.rA0 (r)2 )
//! r
//!
//! q
//!
//! [53]
//!
//! 1≤r≤3
//!
//! 0. q
//! 0.93 M
//! r3
//!
//! r4
//!
//! √
//!
//! 6M
//!
//! M
//!
//! r
//!
//! (∗)b
//! √
//!
//! 8
//!
//! − 3M
//! r
//!  
//! 
//! q
//! 2
//! 1
//! 1
//! +
//! 1− M
//! 2
//! r2
//!
//! (
//!
//! 0
//!
//! NO
//!
//! 7
//!
//! − 12 log
//!
//! 1≤r≤3
//!
//! 0
//! √
//!
//! Ref.
//!
//! M
//! r3
//!
//! r (r−2.25M )
//!
//! METRIC-AFFINE
//!
//! 9
//!
//! ΩK (r)
//! q
//!
//! 
//!
//! 3≤r
//! 
//! 3≤r≤4
//! 0. r
//! 9M
//! 1.06 M (1−3 4r )(r−6.75M ) 4 ≤ r
//!
//! 6.75M
//!
//! 2
//! − Mr
//!
//! TELEPARALLEL
//!
//! 1 − 6M
//! r
//!
//! (
//! 0. q
//! )
//! 0.93 M (r−5.2M
//! r4
//!
//! 5.20M
//!
//! 3
//! −M
//! r3
//!
//! 3
//!
//! M
//! r3
//!
//! q
//!
//! [61]
//! M3
//! r5
//!
//! 3e
//! q 0
//! A (r)
//! 0.71A(r)1.5 rA(r)
//! √ − 3M q M
//! 3e r
//! r3
//!
//! [35]
//! [62]
//! [63]
//!
//! 2
//! − 2 √ 2 M2 2
//! r ( r −M +r ) −M 2
//!
//! [4]
//!
//! a “NO” means that does not exist the ISCO radius.
//! b The symbol (∗) means that the marked functions have not real values.
//! c We have that A(r) =
//!
//! 1
//! Ω+ (r/M )
//!
//! 
//! 1−
//!
//! 1+0.25G(r/M )
//! √
//! 0.5 Ω− (r/M )r/M
//!
//! 
//! √
//! 
//! 
//! , where Ω± (z) = 1 ± z −4 and G(z) = −0.57 + 0.5 z 4 − 1 f3/4 (z) + f7/4 (z)
//!
//! with fλ (z) = 2 F1 (1/2, λ, 3/2, 1 − z 4 ) being the hypergeometric function. We indicate with 0 = d/dr.
//!
//! to determine eΦ(r1 ) = r1 sin αE /bph (r1 ), [18, 64]; invoking gravitational redshift effects, which permit to have
//! 1 + z = eΦ(r1 ) [65–67]; in the case where r2  r1 , it
//! would be reasonable to have e2Φ(r1 ) ≈ 1, and so recast
//! Eq. (28) in a way that e2Φ(r2 ) appears, instead of e2Φ(r1 ) .
//! On the other hand, the integral term can be exactly
//! calculated depending on how Ωϕ is sampled in the region
//! [r1 , r2 ]. Let us assume N + 1 points r1 ≡ x0 < x1 · · · <
//! xN −1 < xN ≡ r2 , in correspondence of which we have our
//! measured samples {Ωϕ (x̄i )}i=1,...,N , where x̄i ∈ [xi−1 , xi ]
//!
//! for i = 1, . . . , N . Therefore, we arrive to the formula
//! Z r2
//! r1
//!
//! 2xΩ2ϕ (x)dx =
//!
//! N
//! X
//!
//! 2x̄i Ω2ϕ (x̄i )(xi − xi−1 ),
//!
//! (29)
//!
//! i=1
//!
//! which allows to obtain the N nodes {x̄i , Φ(x̄i )}i=1,...,N to
//! be fitted. We finally reconstruct the explicit expression
//! of Φ(r), and we can also calculate Φ0 (r) and Φ00 (r).
//! Now instead, using Eq. (22) we can determine the
//!
//!
//! --- PAGE BREAK ---
//! 7
//!
//! 3.5
//! #1
//!
//! #2
//!
//! #4
//!
//! #9
//!
//! #10
//!
//! #1
//!
//! 3.0
//!
//! #2
//!
//! #4
//!
//! #9
//!
//! #10
//!
//! 2.5
//!
//! 0.01
//! Ωr (WH)/Ωr (Schw)
//!
//! Ωr (WH)-Ωr (Schw) (M-1)
//!
//! Legend
//!
//! Legend
//!
//! 0.02
//!
//! 0.00
//!
//! 2.0
//! 1.5
//! 1.0
//!
//! -0.01
//!
//! 0.5
//! -0.02
//!
//! 0.0
//! 6
//!
//! 8
//!
//! 10
//!
//! 12
//!
//! 14
//!
//! 16
//!
//! 18
//!
//! 20
//!
//! 6
//!
//! r (M)
//!
//! 10
//!
//! 12
//!
//! 14
//!
//! 16
//!
//! 18
//!
//! 20
//!
//! r (M)
//!
//! 1
//!
//! 100
//!
//! Legend
//! #1
//!
//! #2
//!
//! #4
//!
//! #6
//!
//! #7
//!
//! 50
//!
//! #9
//!
//! Legend
//! #1
//!
//! ΩK (WH)/ΩK (Schw)
//!
//! ΩK (WH)-ΩK (Schw) (M-1)
//!
//! 8
//!
//! 0.100
//!
//! 0.010
//!
//! #2
//!
//! #4
//!
//! #6
//!
//! #7
//!
//! #9
//!
//! 10
//! 5
//!
//! 1
//! 0.5
//!
//! 0.001
//!
//! 0.1
//! 6
//!
//! 8
//!
//! 10
//!
//! 12
//!
//! 14
//!
//! 16
//!
//! 18
//!
//! 20
//!
//! r (M)
//!
//! 6
//!
//! 8
//!
//! 10
//!
//! 12
//!
//! 14
//!
//! 16
//!
//! 18
//!
//! 20
//!
//! r (M)
//!
//! FIG. 3. Plots of difference
//! n (left panels)oand ratio (right panels) between the radial and azimuthal epicyclic angular frequencies
//! (WH)
//!
//! (WH)
//!
//! (only those admitting real values, see Table I and Fig. 3), and the Schwarzschild BH
//! for some WH solutions Ωr
//! , ΩK
//! o
//! n
//! (Schw)
//! (Schw)
//! (represented by dashed black lines).
//! values Ωr
//! , ΩK
//!
//! shape function b(r) in [r1 , r2 ] through the formula
//! 
//! 
//! 2
//! Ω
//! (r)
//! r
//! h
//! i + 1 r. (30)
//! b(r) = 
//! 0
//! e2Φ(r) 2Φ02 (r) − 3Φr(r) − Φ00 (r)
//! We then discretize the interval [r1 , r2 ] and use the observed samples {Ωr (x̄i )}i=1,...,N . In this way, we straightforwardly obtain the nodes {x̄i , b(x̄i )}i=1,...,N , which in
//! turn can be fitted to reconstruct also b(r).
//!
//! 2.
//!
//! A test-example
//!
//! This section aims at exhibiting a practical example to
//! further clarify how the above outlined theoretical procedure works. We provide some simulated data to compensate for the actual absence of real data on WHs.
//! Let us assume to measure the radial νr and azimuthal
//!
//! νϕ epicyclic frequencies related to a WH of mass M =
//! 105 M in N = 10 points {x̄i }i=1,...,10 , included in the
//! radial interval [r1 = 7M, r2 = 10M ], see Table II. Let
//! us then calculate the related epicyclic angular velocities
//! {Ωr , Ωϕ }, and, since we know the WH mass M , we can
//! further convert them into geometrical units, see Table
//! II. Furthermore, we divide uniformly the interval [r1 , r2 ]
//! in bins of amplitude ∆r = (r2 − r1 )/10 = 0.3M . This
//! means that there exists N + 1 = 11 points such that
//! r1 = x0 < x1 < · · · < x9 < x10 = r2 and xi − xi−1 = ∆r
//! for all i = 1, . . . , 10.
//! We need first to determine the redshift function by exploiting Eq. (28). Let us assume, we are able to provide
//! the estimation eΦ(r1 ) = 1.15 through some observational
//! technique. Calculating the integral (29) through the data
//! of Table II, we are finally able to obtain the nodes for the
//! redshift function {x̄i , Φ(x̄i )}i=1,...,10 . In Fig. 5, the nodes
//! and the best fit function are represented (calculations are
//!
//!
//! --- PAGE BREAK ---
//! 8
//! -0.30
//!
//! ●
//! ●
//!
//! #0
//!
//! #1
//!
//! #2
//!
//! ●
//!
//! -0.32
//!
//! Legend
//!
//! 0.04
//!
//! #4
//!
//! ●
//!
//! #9
//!
//! Φ
//!
//! -0.34
//!
//! Ωpre (M-1)
//!
//! 0.03
//!
//! ●
//! ●
//!
//! -0.36
//!
//! ●
//!
//! -0.38
//!
//! 0.02
//!
//! ●
//! ●
//!
//! -0.40
//! ●
//! -0.42
//!
//! 0.01
//!
//! Error
//!
//! 0.015
//!
//! 0.00
//! 6
//!
//! 8
//!
//! 10
//!
//! 12
//!
//! 14
//!
//! 16
//!
//! 18
//!
//! 20
//!
//! r (M)
//!
//! FIG. 4. Precession frequency Ωper in terms of the radius r
//! for some WH solutions admitting real values of both epicyclic
//! angular velocities (see Table I). The dashed black line is the
//! Schwarzschild BH case. The lower tight panel reports the
//! relative errors with respect to the Schwarzschild geometry.
//!
//! TABLE II. We report the N = 10 points x̄i ∈ [r1 = 7M, r2 =
//! 10M ] in correspondence of which we have the sampled values
//! of the radial and azimuthal epicyclic frequencies {νr , νϕ }, and
//! angular velocities {Ωr , Ωϕ } (both in dimensional and geometrical units). We stress that these are not real data.
//! x̄i
//!
//! νr
//! −3
//!
//! (M) (10
//!
//! Ωr
//! Hz) (rad/s) (10
//!
//! Ωr
//! −3
//!
//! νϕ
//! −1
//!
//! M
//!
//! −3
//!
//! ) (10
//!
//! Ωϕ
//!
//! Ωϕ
//!
//! Hz) (rad/s) (M−1 )
//!
//! ▲
//!
//! ▲
//!
//! 0.010
//!
//! ▲
//!
//! ▲
//!
//! ▲
//!
//! 0.005
//!
//! ▲
//!
//! ▲
//!
//! ▲
//!
//! 0.000
//! 7.0
//!
//! ▲
//!
//! ▲
//!
//! 7.5
//!
//! 8.0
//!
//! 8.5
//! r (M)
//!
//! 9.0
//!
//! 9.5
//!
//! 10.0
//!
//! FIG. 5. Upper panel: the blue points represent the nodes
//! {x̄i , Φ(x̄i )}i=1,...,10 , and the black line is the best fit Φ(r) =
//! −2.95168/r. Lower panel: there are the relative fit-errors and
//! the dashed blue line is set at the mean error 0.007.
//!
//! mean, and maximum values are 0.001, 0.007, 0.015, respectively, attesting thus the good agreement of the fit.
//! Once we have the fitted Φ(r), we can analytically calculate Φ0 (r), Φ00 (r), which permits to determine the shape
//! function through Eq. (30). Using also the values reported
//! in Table II, we can calculate the nodes {x̄i , b(x̄i )}i=1,...,10 .
//! In Fig. 6, nodes are fitted with the following function
//!
//! 7.17
//!
//! 7.07
//!
//! 0.044
//!
//! 2.40
//!
//! 1.75
//!
//! 0.110
//!
//! 0.059
//!
//! b(r) = re1.355−0.557r ,
//!
//! 7.52
//!
//! 7.46
//!
//! 0.047
//!
//! 2.53
//!
//! 1.66
//!
//! 0.104
//!
//! 0.056
//!
//! 7.84
//!
//! 7.68
//!
//! 0.048
//!
//! 2.61
//!
//! 1.59
//!
//! 0.100
//!
//! 0.054
//!
//! 8.04
//!
//! 7.76
//!
//! 0.049
//!
//! 2.63
//!
//! 1.54
//!
//! 0.097
//!
//! 0.052
//!
//! whose minimum, mean, and maximum relative fit-error
//! values are 0.0018, 0.0460, 0.1164, respectively, confirming
//! thus again an excellent agreement of the fit.
//!
//! 8.44
//!
//! 7.84
//!
//! 0.049
//!
//! 2.66
//!
//! 1.46
//!
//! 0.092
//!
//! 0.049
//!
//! 8.55
//!
//! 7.85
//!
//! 0.049
//!
//! 2.66
//!
//! 1.44
//!
//! 0.090
//!
//! 0.049
//!
//! 8.91
//!
//! 7.83
//!
//! 0.049
//!
//! 2.66
//!
//! 1.37
//!
//! 0.086
//!
//! 0.047
//!
//! 9.11
//!
//! 7.80
//!
//! 0.049
//!
//! 2.65
//!
//! 1.34
//!
//! 0.084
//!
//! 0.045
//!
//! 9.57
//!
//! 7.70
//!
//! 0.048
//!
//! 2.61
//!
//! 1.26
//!
//! 0.079
//!
//! 0.043
//!
//! 9.82
//!
//! 7.62
//!
//! 0.048
//!
//! 2.59
//!
//! 1.22
//!
//! 0.077
//!
//! 0.042
//!
//! IV.
//!
//! performed in the Mathematica 12 environment)
//! Φ(r) = −
//!
//! 2.95168
//! ,
//! r
//!
//! (32)
//!
//! DISCUSSION AND CONCLUSIONS
//!
//! In this paper, we considered static and spherically symmetric WH geometries described by the Morris-Thornelike metric (1), defined in terms of the redshift and shape
//! functions. This theory-independent formalism provides a
//! general approach to investigate several possible WH solutions within different gravity frameworks. We explore the
//! inverse problem, namely the reconstruction of the WH
//! solution through the fit of possible observational data.
//! In our approach, we used the observer splitting formalism (see Sec. II), which is very useful to disentangle among gravitational and inertial effects, and it also
//!
//! (31)
//!
//! together with the relative fit-errors 1 , whose minimum,
//! n o
//! Φ̃i
//!
//! 1 Denoted with {Φ }
//! i i=1,...,10 the real-measured values, and with
//!
//! i=1,...,10
//!
//! those values obtained through the fitting function
//!
//! (31) evaluated at {x̄i }i=1,...,10 , then the relative fit-errors are
//! n
//! o
//! |Φi − Φ̃i |/|Φi |
//! .
//!
//! calculated as follows
//!
//! i=1,...,10
//!
//!
//! --- PAGE BREAK ---
//! 9
//! ●
//! 0.5
//! ●
//! 0.4
//! ●
//! b (M)
//!
//! ●
//! 0.3
//!
//! ● ●
//! ●
//!
//! ●
//!
//! 0.2
//!
//! ●
//!
//! ●
//!
//! Error
//!
//! 0.1
//!
//! 0.0
//! 0.12
//! 0.10
//! 0.08
//! 0.06
//! 0.04
//! 0.02
//! 0.00
//! 7.0
//!
//! ▲
//! ▲
//! ▲
//!
//! ▲
//!
//! ▲
//!
//! ▲
//!
//! 7.5
//!
//! 8.0
//!
//! ▲
//!
//! ▲
//!
//! 8.5
//! r (M)
//!
//! ▲
//! ▲
//!
//! 9.0
//!
//! 9.5
//!
//! 10.0
//!
//! FIG. 6. Upper panel: the red points represent the nodes
//! {x̄i , b(x̄i )}i=1,...,10 , and the black line is the best fit b(r) =
//! re1.355−0.557r . Lower panel: there are the relative fit-errors
//! and the dashed red line is set at the mean error 0.0460.
//!
//! permits to have a direct connection with the classical description, assigning therefore a precise physical meaning
//! to the quantities theoretically manipulated, see Eqs. (6)
//! and (7). There is a direct way to obtain the epicyclic
//! frequencies through the formulas [23, 24]:
//! s
//! ∂r gtt
//! Ωϕ =
//! ,
//! (33)
//! ∂r gϕϕ
//! 
//! 1  2 2 tt
//! gtt ∂r g + (Ωϕ gϕϕ )2 ∂r2 g ϕϕ ,
//! (34)
//! Ωr =
//! 2grr
//! which can be checked that are equivalent to Eqs. (23)
//! and (22), respectively. In addition, Equs. (33) and (34)
//! can be very useful for numerical implementations.
//! We have stressed the observational aspect of the
//! epicyclic frequencies, since they are extensively used as
//! a fundamental ingredient for the development of QPO
//! models, which are easy to be detected and are a common
//! feature in several X-ray binaries. The study of QPOs requires theoretically a general-relativistic ray-tracing code
//! to inquire their X-ray timing spectroscopy and polarization properties, and experimentally simultaneous observations through first-generation X-ray polarimeters and
//! LOFT-type missions. After having derived the epicyclic
//! frequencies (22) and (23), which include a combination
//! of the redshift, together with its derivatives, and shape
//! functions, see Sec. II A, we then applied our formulas
//! for achieving two goals: (1) detecting the presence of a
//! WH, distinguishing it from a BH (see Sec. III A); (2)
//!
//! [1] M. Visser, Lorentzian wormholes:
//! Hawking (1995).
//!
//! From Einstein to
//!
//! exhibiting a procedure for reconstructing a WH solution
//! through the fit of observational data (see Sec. III B).
//! The first point is timely and essential, because it provides a further astrophysical strategy to reveal the observational existence of a WH. The method is very simple,
//! because it relies on comparing the observed data with the
//! BH information in order to see whether there are some
//! relevant discrepancies. This would mean that metricchanges may occur and a WH may exist. There are some
//! WH solutions, which closely mimic the BH observational
//! proprieties, therefore this procedure alone is not enough
//! sometimes and it must be complemented with other approaches presented in the literature to extract more information and tighter constraint on the theoretical models.
//! Once there would be available data on WHs, we should
//! be able to reconstruct the WH solution. Specifically, exploiting Eqs. (28) and (30), it would be possible to reconstruct the redshift and shape functions. We provide
//! also a test-example based on some simulated data, see
//! Sec. III B 2. This section has only the aim to better clarify how the outlined procedure works practically with
//! the data. Therefore, the following remarks are in order:
//! (i) the data in Table II may be in principle not observable; (ii) the data in Table II are listed without detection
//! errors (depending mainly on the instrument sensitivity
//! used to perform the measurements), so they can be interpreted as the mean values of the detection; (iii) the
//! fit of the data and the related fit-errors can be performed
//! with more advanced statistical methods (see e.g., [68]);
//! (iv) the sampled epicyclic frequencies and radial extent
//! [r1 , r2 ] may be different from those observed.
//! This paper is part of a series of works aiming at providing both observational evidences of WH existence and
//! different techniques to reconstruct them through potential future observational data. This model-independent
//! approach allows not only to determine the WH solutions,
//! but also to provide indirect observational tests of gravity within GR theory or towards Alternative Theories of
//! Gravity. In addition, all these procedures can be adapted
//! and extended also to study other classes of compact objects different from WHs. As near-future projects we aim
//! at complementing this approach with other astrophysical
//! techniques. In particular, this work can be also extended
//! and improved for axially symmetric WHs.
//!
//! ACKNOWLEDGEMENTS
//!
//! V.D.F. thanks Gruppo Nazionale di Fisica Matematica
//! of Istituto Nazionale di Alta Matematica for the support.
//! V.D.F., M.D.L., and S.C. acknowledge the support of
//! INFN sez. di Napoli, iniziative specifiche TEONGRAV,
//! QGSKY, and MOONLIGHT2.
//!
//! [2] M. Visser, Nuclear Physics B 328, 203 (1989),
//! arXiv:0809.0927 [gr-qc].
//!
//!
//! --- PAGE BREAK ---
//! 10
//! [3] C. Barcelo and M. Visser, Physics Letters B 466, 127
//! (1999), arXiv:gr-qc/9908029 [gr-qc].
//! [4] C. G. Böhmer, T. Harko, and F. S. N. Lobo, Phys. Rev.
//! D 85, 044033 (2012), arXiv:1110.5756 [gr-qc].
//! [5] L. A. Anchordoqui, S. Capozziello, G. Lambiase, and
//! D. F. Torres, Mod. Phys. Lett. A15, 2219 (2000),
//! arXiv:gr-qc/0011097 [gr-qc].
//! [6] S. Bahamonde, U. Camci, S. Capozziello, and M. Jamil,
//! Phys. Rev. D94, 084042 (2016), arXiv:1608.03918 [grqc].
//! [7] S. Capozziello, R. D’Agostino, and D. Gregoris, Phys.
//! Dark Univ. 28, 100513 (2020), arXiv:2002.04875 [gr-qc].
//! [8] S. Capozziello, O. Luongo, and L. Mauro, Eur. Phys. J.
//! Plus 136, 167 (2021), arXiv:2012.13908 [gr-qc].
//! [9] Z. Li and C. Bambi, Phys. Rev. D 90, 024071 (2014),
//! arXiv:1405.1883 [gr-qc].
//! [10] V. Cardoso, E. Franzin, and P. Pani, PRL 117, 089902
//! (2016).
//! [11] R. A. Konoplya and A. Zhidenko, JCAP 2016, 043
//! (2016), arXiv:1606.00517 [gr-qc].
//! [12] S. Paul, R. Shaikh, P. Banerjee, and T. Sarkar, arXiv
//! e-prints , arXiv:1911.05525 (2019), arXiv:1911.05525 [grqc].
//! [13] D.-C. Dai and D. Stojkovic, PRD 100, 083513 (2019),
//! arXiv:1910.00429 [gr-qc].
//! [14] P. Banerjee, S. Paul, R. Shaikh, and T. Sarkar, arXiv
//! e-prints , arXiv:1912.01184 (2019), arXiv:1912.01184
//! [astro-ph.HE].
//! [15] K. Hashimoto and N. Tanahashi, PRD 95, 024007 (2017),
//! arXiv:1610.06070 [hep-th].
//! [16] S. Dalui, B. R. Majhi, and P. Mishra, Physics Letters B
//! 788, 486 (2019), arXiv:1803.06527 [gr-qc].
//! [17] V. De Falco, E. Battista, S. Capozziello,
//! and
//! M. De Laurentis, Phys. Rev. D 101, 104037 (2020),
//! arXiv:2004.14849 [gr-qc].
//! [18] V. De Falco, E. Battista, S. Capozziello,
//! and
//! M. De Laurentis, Phys. Rev. D 103, 044007 (2021),
//! arXiv:2101.04960 [gr-qc].
//! [19] V. De Falco, E. Battista, S. Capozziello, and M. De
//! Laurentis, European Physical Journal C 81, 157 (2021),
//! arXiv:2102.01123 [gr-qc].
//! [20] V. Cardoso and P. Pani, Living Reviews in Relativity 22,
//! 4 (2019), arXiv:1904.05363 [gr-qc].
//! [21] S. E. Motta, Astronomische Nachrichten 337, 398 (2016),
//! arXiv:1603.07885 [astro-ph.HE].
//! [22] A. Ingram and S. Motta, arXiv e-prints ,
//! arXiv:2001.08758 (2020), arXiv:2001.08758 [astroph.HE].
//! [23] C. Chakraborty and P. Pradhan, J. Cosmol. Astropart.
//! Phys. 2017, 035 (2017), arXiv:1603.09683 [gr-qc].
//! [24] E. Deligianni, J. Kunz, P. Nedkova, S. Yazadjiev, and
//! R. Zheleva, arXiv e-prints , arXiv:2103.13504 (2021),
//! arXiv:2103.13504 [gr-qc].
//! [25] M. S. Morris and K. S. Thorne, American Journal of
//! Physics 56, 395 (1988).
//! [26] D. Hochberg and M. Visser, Phys. Rev. D 58, 044021
//! (1998), arXiv:gr-qc/9802046 [gr-qc].
//! [27] D. Hochberg and M. Visser, Phys. Rev. Letters 81, 746
//! (1998), arXiv:gr-qc/9802048 [gr-qc].
//! [28] S. Capozziello, F. S. N. Lobo, and J. P. Mimoso, Phys.
//! Rev. D91, 124019 (2015), arXiv:1407.7293 [gr-qc].
//! [29] S. Capozziello, F. S. N. Lobo, and J. P. Mimoso, Phys.
//! Lett. B730, 280 (2014), arXiv:1312.0784 [gr-qc].
//!
//! [30] D. Hochberg, A. Popov, and S. V. Sushkov, Phys. Rev.
//! L. 78, 2050 (1997), arXiv:gr-qc/9701064 [gr-qc].
//! [31] K. A. Bronnikov, L. N. Lipatova, I. D. Novikov, and
//! A. A. Shatskiy, Gravitation and Cosmology 19, 269
//! (2013), arXiv:1312.6929 [gr-qc].
//! [32] R. Garattini, European Physical Journal C 79, 951
//! (2019), arXiv:1907.03623 [gr-qc].
//! [33] F. S. N. Lobo and M. A. Oliveira, Phys. Rev. D 80,
//! 104012 (2009), arXiv:0909.5539 [gr-qc].
//! [34] T. Harko, F. S. N. Lobo, M. K. Mak, and S. V. Sushkov,
//! Phys. Rev. D 87, 067504 (2013), arXiv:1301.6878 [gr-qc].
//! [35] S. Capozziello, T. Harko, T. S. Koivisto, F. S. N.
//! Lobo, and G. J. Olmo, Phys. Rev. D86, 127504 (2012),
//! arXiv:1209.5862 [gr-qc].
//! [36] R. T. Jantzen, P. Carini, and D. Bini, in Marcel Grossmann Meeting on General Relativity, edited by F. Satō
//! and T. Nakamura (1992).
//! [37] D. Bini, P. Carini, and R. T. Jantzen, International Journal of Modern Physics D 06, 1 (1997), gr-qc/0106013.
//! [38] D. Bini, P. Carini, and R. T. Jantzen, International Journal of Modern Physics D 06, 143 (1997), gr-qc/0106014.
//! [39] D. Bini, P. Carini, and R. T. Jantzen, in Recent Developments in Theoretical and Experimental General Relativity, Gravitation, and Relativistic Field Theories, edited
//! by T. Piran and R. Ruffini (1999) p. 376.
//! [40] D. Bini, F. de Felice, and R. T. Jantzen, Classical and
//! Quantum Gravity 16, 2105 (1999).
//! [41] V. De Falco, E. Battista, and M. Falanga, Phys. Rev. D
//! D97, 084048 (2018), arXiv:1804.00519 [gr-qc].
//! [42] W. H. G. Lewin, J. van Paradijs, and E. P. J. van den
//! Heuvel, X-ray Binaries (1997).
//! [43] B. Beheshtipour, J. K. Hoormann, and H. Krawczynski,
//! ApJ 826, 203 (2016), arXiv:1605.09756 [astro-ph.HE].
//! [44] R. A. Remillard and J. E. McClintock, ARA&A 44, 49
//! (2006), arXiv:astro-ph/0606352 [astro-ph].
//! [45] M. Feroci, E. Bozzo, S. Brandt, M. Hernanz, M. van der
//! Klis, L. P. Liu, P. Orleanski, M. Pohl, A. Santangelo,
//! S. Schanne, and et al., in Space Telescopes and Instrumentation 2016: Ultraviolet to Gamma Ray, Society of
//! Photo-Optical Instrumentation Engineers (SPIE) Conference Series, Vol. 9905, edited by J.-W. A. den Herder,
//! T. Takahashi, and M. Bautz (2016) p. 99051R.
//! [46] S. Zhang, M. Feroci, A. Santangelo, Y. Dong, H. Feng,
//! et al.
//! [47] P. Soffitta, X. Barcons, R. Bellazzini, J. Braga, E. Costa,
//! G. W. Fraser, S. Gburek, J. Huovelin, G. Matt,
//! M. Pearce, J. Poutanen, V. Reglero, A. Santangelo,
//! R. A. Sunyaev, G. Tagliaferri, M. Weisskopf, R. Aloisio,
//! E. Amato, P. Attiná, M. Axelsson, L. Baldini, S. Basso,
//! S. Bianchi, P. Blasi, J. Bregeon, A. Brez, N. Bucciantini, L. Burderi, V. Burwitz, P. Casella, E. Churazov, M. Civitani, S. Covino, R. M. Curado da Silva,
//! G. Cusumano, M. Dadina, F. D’Amico, A. De Rosa,
//! S. Di Cosimo, G. Di Persio, T. Di Salvo, M. Dovciak,
//! R. Elsner, C. J. Eyles, A. C. Fabian, S. Fabiani, H. Feng,
//! S. Giarrusso, R. W. Goosmann, P. Grandi, N. Grosso,
//! G. Israel, M. Jackson, P. Kaaret, V. Karas, M. Kuss,
//! D. Lai, G. La Rosa, J. Larsson, S. Larsson, L. Latronico, A. Maggio, J. Maia, F. Marin, M. M. Massai, T. Mineo, M. Minuti, E. Moretti, F. Muleri, S. L.
//! O’Dell, G. Pareschi, G. Peres, M. Pesce, P.-O. Petrucci,
//! M. Pinchera, D. Porquet, B. Ramsey, N. Rea, F. Reale,
//! J. M. Rodrigo, A. Różańska, A. Rubini, P. Rudawy,
//! F. Ryde, M. Salvati, V. A. r. de Santiago, S. Sazonov,
//!
//!
//! --- PAGE BREAK ---
//! 11
//! C. Sgró, E. Silver, G. Spandre, D. Spiga, L. Stella,
//! T. Tamagawa, F. Tamborra, F. Tavecchio, T. Teixeira
//! Dias, M. van Adelsberg, K. Wu, and S. Zane, Experimental Astronomy 36, 523 (2013), arXiv:1309.6995 [astroph.HE].
//! [48] Y.-F. Cai, S. Capozziello, M. De Laurentis,
//! and
//! E. N. Saridakis, Rept. Prog. Phys. 79, 106901 (2016),
//! arXiv:1511.07586 [gr-qc].
//! [49] S. Capozziello, M. De Laurentis, and V. Faraoni, Open
//! Astron. J. 3, 49 (2010), arXiv:0909.4672 [gr-qc].
//! [50] G. J. Olmo, Int. J. Mod. Phys. D 20, 413 (2011),
//! arXiv:1101.3864 [gr-qc].
//! [51] S. Capozziello and M. De Laurentis, Phys. Rept. 509,
//! 167 (2011), arXiv:1108.6266 [gr-qc].
//! [52] T. Clifton, P. G. Ferreira, A. Padilla, and C. Skordis,
//! Phys. Rep. 513, 1 (2012), arXiv:1106.2476 [astro-ph.CO].
//! [53] M. A. Abramowicz and W. Kluźniak, Astrophysics and
//! Space Science 300, 127 (2005), arXiv:astro-ph/0411709
//! [astro-ph].
//! [54] M. Falanga, T. Belloni, P. Casella, M. Gilfanov,
//! P. Jonker, and A. King, eds., The Physics of Accretion
//! onto Black Holes (Springer-Verlag New York, 2015).
//! [55] C. Bambi, Astrophysics of Black Holes, Vol. 440 (2016).
//! [56] L. Stella and M. Vietri, PRL 82, 17 (1999), arXiv:astroph/9812124 [astro-ph].
//! [57] J. P. S. Lemos, F. S. N. Lobo, and S. Q. de Oliveira,
//! Phys. Rev. D 68, 064004 (2003).
//! [58] R. Myrzakulov, L. Sebastiani, S. Vagnozzi,
//! and
//! S. Zerbini, Classical and Quantum Gravity 33, 125005
//!
//! (2016), arXiv:1510.02284 [gr-qc].
//! [59] N. Godani and G. C. Samanta, New Astron. 80, 101399
//! (2020), arXiv:2004.14209 [gr-qc].
//! [60] M. Sharif and Z. Zahra, Astrophys Space Sci. 348, 275
//! (2013).
//! [61] M. Calzà, M. Rinaldi, and L. Sebastiani, European Physical Journal C 78, 178 (2018), arXiv:1802.00329 [gr-qc].
//! [62] C. Bejarano, F. S. N. Lobo, G. J. Olmo, and D. RubieraGarcia, European Physical Journal C 77, 776 (2017),
//! arXiv:1607.01259 [gr-qc].
//! [63] M. Sharif and K. Nazir, Annals of Physics 393, 145
//! (2018).
//! [64] G. S. Bisnovatyi-Kogan and O. Y. Tsupko, Plasma
//! Physics Reports 41, 562 (2015), arXiv:1507.08545 [grqc].
//! [65] H. Müller, A. Peters, and S. Chu, Nature 463, 926
//! (2010).
//! [66] S. Herrmann, F. Finke, M. Lülf, O. Kichakova, D. Puetzfeld, D. Knickmann, M. List, B. Rievers, G. Giorgi,
//! C. Günther, H. Dittus, R. Prieto-Cerdeira, F. Dilssner,
//! F. Gonzalez, E. Schönemann, J. Ventura-Traveset, and
//! C. Lämmerzahl, Phys. Rev. Lett. 121, 231102 (2018),
//! arXiv:1812.09161 [gr-qc].
//! [67] F. Di Pumpo, C. Ufrecht, A. Friedrich, E. Giese,
//! W. P. Schleich, and W. G. Unruh, arXiv e-prints ,
//! arXiv:2104.14391 (2021), arXiv:2104.14391 [quant-ph].
//! [68] A. Chattopadhyay and T. Chattopadhyay, Statistical
//! Methods for Astronomical Data Analysis, Springer series
//! in astrostatistics (Springer, 2014).
//!
//!
//! --- PAGE BREAK ---
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//!
