//! # Topological Rendering and the Event Horizon
//!
//! This section details the visual and geometrical representation of
//! non-associative spacetimes, implemented via the `gororoba_optix` and
//! OpenGL rendering pipelines.
//!
//! ## 1. The Logarithmic Wiregrid
//!
//! The Blackhole simulation utilizes a wiregrid with **logarithmic radial spacing**.
//! This is not merely an aesthetic choice; it mathematically ensures that the
//! topological density of the grid matches the exponential divergence of the
//! spatial metric near the event horizon ($r_s = 2GM/c^2$).
//!
//! Specifically, the minimum radial coordinate is algorithmically aligned to
//! $r_{min} \ge 1.01 \cdot r_s$, ensuring numerical stability while capturing
//! the extreme curvature gradients.
//!
//! ## 2. Rendering Non-Associativity
//!
//! How does one visualize the failure of associativity?
//!
//! The engine applies cinematic post-processing (chromatic aberration, vignette,
//! film grain) coupled with a high-contrast '16-bit Voxel' theme. In the
//! non-associative metric model, **chromatic aberration** acts as a visual surrogate
//! for the **Associativity Violation Tensor (AVT)**.
//!
//! As a ray-traced photon ($x^\mu$) approaches the horizon, the breakdown of the
//! associator $[x^\mu, x^\nu, x^\rho] \neq 0$ induces phase decoherence. This is
//! rendered visually as wavelength-dependent spatial splitting (aberration),
//! mimicking the topological friction and birefringence predicted by the 512D
//! resonant vacuum.
//!
//! ## 3. Interactive Topology
//!
//! The ImGui DockSpace architecture separates the 3D scene (the non-associative
//! bulk) from the UI (the associative boundary observer). Mouse interactivity
//! within the Viewport allows real-time perturbation of the algebraic defect
//! density ($\phi$), providing a direct visual feedback loop for exploring
//! the state space of the Sedenion vacuum.
