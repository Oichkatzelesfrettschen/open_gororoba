//! EGM2008 static Earth gravity model provider.
//!
//! Source: NGA Earth Gravity Model 2008 distribution.

const EGM2008_URLS: &[&str] = &[
    "https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/EGM2008_to2190_TideFree.gz",
    "http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/EGM2008_to2190_TideFree.gz",
];

simple_provider! {
    /// EGM2008 provider.
    pub struct Egm2008Provider;
    name = "EGM2008 Static Geoid";
    output = "EGM2008_to2190_TideFree.gz";
    urls = EGM2008_URLS;
}
