//! Parity-clique edge counting (Rocq-verified: C-882).
//!
//! |E(K_n)| = n*(n-1)/2.  At CD dim=16: 2*K_4 = 12 edges.
//! At CD dim=32: 2*K_8 = 56 edges.
//! Kernel-checked in `proofs/verified/C882_ParityCliqueExact.v`.

/// Number of edges in a complete graph K_n.
pub fn clique_edges(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(clique_edges(0), 0);
        assert_eq!(clique_edges(1), 0);
        assert_eq!(clique_edges(2), 1);
        assert_eq!(clique_edges(3), 3);
        assert_eq!(clique_edges(4), 6);
        assert_eq!(clique_edges(8), 28);
        assert_eq!(clique_edges(16), 120);
    }

    #[test]
    fn dim16_two_cliques() {
        assert_eq!(2 * clique_edges(4), 12);
    }

    #[test]
    fn dim32_two_cliques() {
        assert_eq!(2 * clique_edges(8), 56);
    }
}
