//! Hexagonal Graph Database (Zhilina Indexing)
//!
//! A database indexing schema based on Zhilina's oriented hexagons.
//! Nodes are connected by the rigid geometry of CD algebras. To query related data,
//! you multiply the current node by a specific sequence of Sedenion basis elements.

use cd_kernel::cayley_dickson::cd_multiply;

pub type ZhilinaNode = [f64; 16];

/// **Hexagonal Traversal Query**
/// Traverses a Zhilina hexagon to retrieve a related node.
/// Start -> edge_1 -> edge_2 -> yields the target node via multiplication.
pub fn traverse_hexagon(
    start_node: &ZhilinaNode,
    edge_1: &ZhilinaNode,
    edge_2: &ZhilinaNode,
) -> ZhilinaNode {
    let intermediate: [f64; 16] = cd_multiply(start_node, edge_1).try_into().unwrap();
    cd_multiply(&intermediate, edge_2).try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_graph_traversal() {
        let start = [1.0; 16];
        let e1 = [0.5; 16];
        let e2 = [0.2; 16];
        let target = traverse_hexagon(&start, &e1, &e2);
        assert_ne!(target[0], 0.0);
    }
}
