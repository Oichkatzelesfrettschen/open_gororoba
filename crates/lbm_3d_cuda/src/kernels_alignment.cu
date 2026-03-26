// GPU-accelerated Sedenion Box-Kite alignment scan.
//
// Calculates the alignment spectrum (projection weights) for a batch of
// 16D sedenions against the 7 box-kite subspaces, across all 168
// PSL(2,7) orientations.
//
// Grid: ceil(n_vectors / 256) x 1
// Block: 256 threads
//
// Each thread handles one vector and iterates over 168 orientations.
// Finds the orientation that maximizes the total captured energy.

extern "C" __global__ void box_kite_alignment_scan(
    const double* __restrict__ vectors,    // [n_vectors * 16] 16D f64
    const unsigned char* __restrict__ orientations, // [168 * 16] permutations of indices 0..15
    const unsigned char* __restrict__ bk_indices,   // [7 * 12] basis indices for each box-kite
    double* __restrict__ out_max_alignment, // [n_vectors] max alignment found
    unsigned int* __restrict__ out_best_orient, // [n_vectors] index of best orientation
    unsigned int n_vectors,
    unsigned int n_orientations // 168
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    const double* v = &vectors[idx * 16];
    
    // Compute norm squared of the vector
    double norm_sq = 0.0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        norm_sq += v[i] * v[i];
    }
    
    if (norm_sq < 1e-30) {
        out_max_alignment[idx] = 0.0;
        out_best_orient[idx] = 0;
        return;
    }

    double global_max_alignment = -1.0;
    unsigned int best_orient_idx = 0;

    // Scan all 168 PSL(2,7) orientations
    for (unsigned int o = 0; o < n_orientations; o++) {
        const unsigned char* perm = &orientations[o * 16];
        
        // Compute alignment for this orientation
        double current_total_captured = 0.0;
        
        // For each of the 7 box-kites
        for (int k = 0; k < 7; k++) {
            const unsigned char* indices = &bk_indices[k * 12];
            double proj_sq = 0.0;
            
            // Sum squared components at the permuted indices
            #pragma unroll
            for (int j = 0; j < 12; j++) {
                // Apply the permutation: v[perm[basis_index]]
                unsigned char p_idx = perm[indices[j]];
                proj_sq += v[p_idx] * v[p_idx];
            }
            
            double weight = proj_sq / norm_sq;
            // The box-kites may overlap, so we can't just sum weights
            // if we want a partition of unity, but here we just want
            // the max individual box-kite weight or total captured.
            // Let's find the max weight among the 7 box-kites for this orientation.
            if (weight > current_total_captured) {
                current_total_captured = weight;
            }
        }
        
        if (current_total_captured > global_max_alignment) {
            global_max_alignment = current_total_captured;
            best_orient_idx = o;
        }
    }

    out_max_alignment[idx] = global_max_alignment;
    out_best_orient[idx] = best_orient_idx;
}
