//! GRACE GGM05S static gravity field model provider.
//!
//! GGM05S is a satellite-only gravity field model derived from GRACE
//! (Gravity Recovery and Climate Experiment) data. Degree/order 180.
//! Available in ICGEM .gfc format (~350 KB).
//!
//! Source: ICGEM, <http://icgem.gfz-potsdam.de/>
//! Reference: Ries et al. (2016), GFZ Data Services

/// ICGEM .gfc download URLs for GGM05S.
const GGM05S_URLS: &[&str] = &[
    "https://icgem.gfz-potsdam.de/getmodel/gfc/06a6faa24892df587d29c8a345e09e7031428cf97d4fcc9435b31ae8e4ccc021/GGM05S.gfc",
];

simple_provider! {
    /// GRACE GGM05S gravity field model provider.
    pub struct GraceGgm05sProvider;
    name = "GRACE GGM05S Gravity Field";
    output = "GGM05S.gfc";
    urls = GGM05S_URLS;
}
