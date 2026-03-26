//! Domain-specific query expansion.
//!
//! Ported in spirit from AutoResearchClaw's `domain_queries.py`.

/// Return high-value prebuilt queries for a topic and optional explicit domains.
pub fn get_domain_queries(topic: &str, domains: &[String]) -> Vec<String> {
    let mut active = domains
        .iter()
        .map(|domain| domain.trim().to_ascii_lowercase())
        .collect::<std::collections::BTreeSet<_>>();

    let topic_lower = topic.to_ascii_lowercase();
    for (domain, _) in domain_query_map() {
        let core_keyword = domain.split('-').next().unwrap_or(domain);
        if topic_lower.contains(core_keyword) {
            active.insert(domain.to_string());
        }
    }

    let mut queries = Vec::new();
    for domain in active {
        if let Some(candidates) = domain_query_map().get(domain.as_str()) {
            for candidate in *candidates {
                if !queries.iter().any(|existing| existing == candidate) {
                    queries.push((*candidate).to_string());
                }
            }
        }
    }
    queries
}

fn domain_query_map() -> std::collections::BTreeMap<&'static str, &'static [&'static str]> {
    std::collections::BTreeMap::from([
        (
            "manga",
            &[
                "MaNGA integral field spectroscopy rotation curve",
                "MaNGA IFU galaxy kinematics dark matter",
            ][..],
        ),
        (
            "ifu",
            &[
                "IFU spectroscopy galaxy kinematics",
                "integral field unit survey velocity dispersion",
            ][..],
        ),
        (
            "lotss",
            &[
                "LoTSS LOFAR low-frequency radio survey",
                "LoTSS DR2 radio continuum galaxy",
            ][..],
        ),
        (
            "rotation-curve",
            &[
                "rotation curve NFW profile dark matter constraint",
                "galaxy rotation curve systematic uncertainty",
            ][..],
        ),
        (
            "nfw",
            &[
                "NFW halo profile dark matter rotation curve",
                "Navarro-Frenk-White profile fitting",
            ][..],
        ),
        (
            "dark-matter",
            &[
                "dark matter detection null result upper limit",
                "dark matter halo galaxy kinematics constraint",
            ][..],
        ),
        (
            "null-result",
            &[
                "null result dark matter detection galaxy survey",
                "non-detection upper limit dark matter signal",
            ][..],
        ),
        (
            "cayley-dickson",
            &[
                "Cayley-Dickson algebra hypercomplex numbers physics",
                "octonion sedenion particle physics application",
            ][..],
        ),
        (
            "octonion",
            &[
                "octonion algebra exceptional Lie group physics",
                "G2 octonion gauge theory",
            ][..],
        ),
        (
            "sedenion",
            &[
                "sedenion algebra zero divisor physics",
                "16-dimensional hypercomplex number physics",
            ][..],
        ),
        (
            "formal-verification",
            &[
                "formal verification proof assistant scientific computing",
                "Coq Rocq theorem prover physics mathematics",
            ][..],
        ),
        (
            "rocq",
            &[
                "Rocq Coq proof assistant formal verification",
                "interactive theorem proving mathematics",
            ][..],
        ),
        (
            "cosmology",
            &[
                "dark energy equation of state observational constraint",
                "Pantheon supernova cosmological parameter",
            ][..],
        ),
        (
            "lbm",
            &[
                "lattice Boltzmann method fluid simulation turbulence",
                "LBM GPU acceleration computational fluid dynamics",
            ][..],
        ),
        (
            "neural",
            &[
                "neural network deep learning benchmark reproducibility",
                "machine learning scientific discovery",
            ][..],
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::get_domain_queries;

    #[test]
    fn explicit_domains_are_expanded() {
        let domains = vec!["cayley-dickson".to_string()];
        let queries = get_domain_queries("generic topic", &domains);
        assert!(queries.iter().any(|query| query.contains("Cayley-Dickson")));
    }

    #[test]
    fn topic_auto_detects_domain_keywords() {
        let queries = get_domain_queries("sedenion zero divisor structure", &[]);
        assert!(
            queries
                .iter()
                .any(|query| query.contains("sedenion algebra"))
        );
    }
}
