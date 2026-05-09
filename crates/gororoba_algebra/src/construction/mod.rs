pub mod albert;
pub mod category;
pub mod cayley_dickson;
pub mod cd_loop;
pub mod cd_tower;
pub mod chingon;
pub mod clifford;
pub mod composition_algebra_census;
pub mod deep_space;
pub mod exceptional_bridge;
pub mod exotic_octonions;
pub mod functor;
pub mod g2_automorphisms;
pub mod hypercomplex;
pub mod jordan;
pub mod kronecker;
pub mod mult_table;
pub mod non_associative;
pub mod octonion;
pub mod octonion_geometry;
pub mod padic;
pub mod real_part_projection;
pub mod signature_observables;
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
    pub use super::{cayley_dickson::*, hypercomplex::*, mult_table::*};
}

pub mod composition {
    pub use super::{
        composition_algebra_census::*, exotic_octonions::*, non_associative::*, octonion::*,
        signature_observables::*, split_octonion::*, symmetric_composition::*, tessarines::*,
    };
}

pub mod jordan_exceptional {
    pub use super::{albert::*, exceptional_bridge::*, g2_automorphisms::*, jordan::*};
}

pub mod auxiliary {
    pub use super::{
        chingon::*, deep_space::*, kronecker::*, octonion_geometry::*, padic::*, wheels::*,
    };
}
pub mod complex_octonion;
pub mod icosians;
