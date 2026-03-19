def cd_sign(p, q, dim):
    if dim == 1: return 1
    half = dim // 2
    if p < half and q < half: return cd_sign(p, q, half)
    if p < half and q >= half: return cd_sign(q - half, p, half)
    if p >= half and q < half:
        s = cd_sign(p - half, q, half)
        return s if q == 0 else -s
    qh, ph = q - half, p - half
    if qh == 0: return -1
    return cd_sign(qh, ph, half)

dim = 16
edges = []
for i in range(dim):
    for j in range(i + 1, dim):
        edges.append((i, j, cd_sign(i, j, dim)))

# Harary-Zaslavsky imbalance: distance to balanced state.
# A graph is balanced iff there exists a potential x_i in {-1, 1} s.t. sigma_{ij} = x_i * x_j.
# We want to maximize sum(sigma_{ij} * x_i * x_j).
# This is equivalent to MAX-CUT on the signed graph.

def imbalance(dim, edges):
    best_agreement = 0
    # Try all 2^n potentials
    for bits in range(1 << dim):
        x = [1 if (bits >> i) & 1 else -1 for i in range(dim)]
        agreement = 0
        for i, j, sign in edges:
            if sign == x[i] * x[j]:
                agreement += 1
        if agreement > best_agreement:
            best_agreement = agreement
    
    total = len(edges)
    min_flips = total - best_agreement
    return min_flips, total

min_flips, total = imbalance(dim, edges)
print(f"Dim: {dim}")
print(f"Total Edges: {total}")
print(f"Min Flips: {min_flips}")
print(f"Density: {min_flips/total}")
