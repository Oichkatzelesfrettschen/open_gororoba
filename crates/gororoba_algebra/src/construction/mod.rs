#[cfg(feature = "physics")]
pub mod albert;
#[cfg(feature = "lie")]
pub mod category;
pub mod cayley_dickson;
#[cfg(all(feature = "analysis", feature = "physics"))]
pub mod cd_loop;
pub mod cd_tower;
#[cfg(feature = "analysis")]
pub mod chingon;
pub mod clifford;
pub mod composition_algebra_census;
pub mod deep_space;
pub mod e8_root_system;
#[cfg(feature = "lie")]
pub mod exceptional_bridge;
pub mod exotic_octonions;
pub mod functor;
pub mod g2_automorphisms;
#[cfg(all(feature = "analysis", feature = "physics"))]
pub mod hypercomplex;
pub mod jordan;
pub mod kronecker;
pub mod mult_table;
pub mod non_associative;
pub mod octonion;
#[cfg(feature = "physics")]
pub mod octonion_geometry;
pub mod padic;
pub mod real_part_projection;
pub mod signature_observables;
#[cfg(all(feature = "analysis", feature = "physics"))]
pub mod species_audit;
pub mod speculative;
pub mod split_octonion;
pub mod symmetric_composition;
pub mod tessarines;
pub mod twist;
pub mod wheels;

#[cfg(feature = "lie")]
pub mod golay_code;
#[cfg(feature = "lie")]
pub mod leech_lattice;

pub mod cd {
    #[cfg(all(feature = "analysis", feature = "physics"))]
    pub use super::hypercomplex::*;
    pub use super::{cayley_dickson::*, mult_table::*};
}

pub mod composition {
    pub use super::{
        composition_algebra_census::*, exotic_octonions::*, non_associative::*, octonion::*,
        signature_observables::*, split_octonion::*, symmetric_composition::*, tessarines::*,
    };
}

pub mod jordan_exceptional {
    #[cfg(feature = "physics")]
    pub use super::albert::*;
    #[cfg(feature = "lie")]
    pub use super::exceptional_bridge::*;
    pub use super::{g2_automorphisms::*, jordan::*};
}

pub mod auxiliary {
    #[cfg(feature = "analysis")]
    pub use super::chingon::*;
    #[cfg(feature = "physics")]
    pub use super::octonion_geometry::*;
    pub use super::{deep_space::*, e8_root_system::*, kronecker::*, padic::*, wheels::*};
}
#[cfg(feature = "physics")]
pub mod complex_octonion;
#[cfg(feature = "physics")]
pub mod icosians;
