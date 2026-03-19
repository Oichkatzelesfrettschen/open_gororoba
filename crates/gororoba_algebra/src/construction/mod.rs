pub mod albert;
pub mod category;
pub mod cayley_dickson;
pub mod chingon;
pub mod clifford;
pub mod composition_algebra_census;
pub mod deep_space;
pub mod e8_root_system;
pub mod exceptional_bridge;
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
pub mod speculative;
pub mod split_octonion;
pub mod tessarines;
pub mod wheels;

pub mod cd {
    pub use super::{cayley_dickson::*, hypercomplex::*, mult_table::*};
}

pub mod composition {
    pub use super::{
        composition_algebra_census::*, non_associative::*, octonion::*, split_octonion::*,
        tessarines::*,
    };
}

pub mod jordan_exceptional {
    pub use super::{albert::*, exceptional_bridge::*, g2_automorphisms::*, jordan::*};
}

pub mod auxiliary {
    pub use super::{
        chingon::*, deep_space::*, e8_root_system::*, kronecker::*, octonion_geometry::*, padic::*,
        wheels::*,
    };
}
pub mod complex_octonion;
