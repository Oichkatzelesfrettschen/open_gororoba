//! GRACE-FO gravity field model provider.
//!
//! GRACE Follow-On extends time-variable gravity monitoring after GRACE.
//! This provider fetches a representative monthly model file.
//!
//! Source: ICGEM GFZ model services

const GRACE_FO_URLS: &[&str] = &[
    "https://icgem.gfz-potsdam.de/getseries/01_GRACE/GFZ/GFZ%20Release%2006.3%20%28GFO%29/60x60/unfiltered/GSM-2_2018152-2018181_GRFO_GFZOP_BA01_0603.gfc",
    "https://icgem.gfz-potsdam.de/getseries/01_GRACE/JPL/JPL%20Release%2006.3%20%28GFO%29/60x60/unfiltered/GSM-2_2018152-2018181_GRFO_JPLEM_BA01_0603.gfc",
];

simple_provider! {
    /// GRACE-FO provider.
    pub struct GraceFoProvider;
    name = "GRACE-FO Gravity Field";
    output = "GRACEFO_monthly_sample.gfc";
    urls = GRACE_FO_URLS;
}
