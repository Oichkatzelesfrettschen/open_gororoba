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

def is_associative(i, j, k, dim):
    # (ei * ej) * ek == ei * (ej * ek)
    # s1 * e_{i^j} * ek == ei * s2 * e_{j^k}
    # s1 * s3 * e_{i^j^k} == s2 * s4 * e_{i^j^k}
    s1 = cd_sign(i, j, dim)
    s2 = cd_sign(j, k, dim)
    s3 = cd_sign(i ^ j, k, dim)
    s4 = cd_sign(i, j ^ k, dim)
    return s1 * s3 == s2 * s4

dim = 16
triples = []
for i in range(1, dim):
    for j in range(i + 1, dim):
        k = i ^ j
        if k > j:
            triples.append((i, j, k))

assoc = 0
non_assoc = 0
for i, j, k in triples:
    if is_associative(i, j, k, dim):
        assoc += 1
    else:
        non_assoc += 1

print(f"Total non-trivial triples: {len(triples)}")
print(f"Associative: {assoc}")
print(f"Non-associative: {non_assoc}")
print(f"Ratio: {non_assoc}/{len(triples)}")
