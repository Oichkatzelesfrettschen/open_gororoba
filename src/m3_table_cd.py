import csv
import os

# Generates the full m3 table for the CD(O) -> S contraction
# and saves to CSV.


class Octonion:
    # A lightweight integer-based Octonion multiplication table
    def __init__(self):
        # Basis 0..7
        # Table: (i, j) -> (sign, k)
        self.table = {}

        # Identity
        for i in range(8):
            self.table[(0, i)] = (1, i)
            self.table[(i, 0)] = (1, i)

        # Squares
        for i in range(1, 8):
            self.table[(i, i)] = (-1, 0)

        # Triples (Standard Fano)
        triples = [
            (1,2,3), (1,4,5), (1,7,6),
            (2,4,6), (2,5,7),
            (3,4,7), (3,6,5)
        ]

        for (i,j,k) in triples:
            self.table[(i,j)] = (1, k)
            self.table[(j,i)] = (-1, k)

            self.table[(j,k)] = (1, i)
            self.table[(k,j)] = (-1, i)

            self.table[(k,i)] = (1, j)
            self.table[(i,k)] = (-1, j)

    def mul(self, sign_a, idx_a, sign_b, idx_b):
        # Returns (sign_out, idx_out)
        if (idx_a, idx_b) not in self.table:
            # This should not happen if logic is correct for 0..7
            # But for safety, return 0 or error
            return (0, 0)
        s, k = self.table[(idx_a, idx_b)]
        return (sign_a * sign_b * s, k)

# CD Doubling Logic on (sign, idx) pairs
# S element: ( (s_a, i_a), (s_b, i_b) ) representing pair of scaled basis elements
# We compute m3 on PURE basis elements of O embedded via i(x) = (x,x)


def compute_m3(octonion, i, j, k):
    # m3(ei, ej, ek) = p( mu( h( mu(ii, ij) ), ik ) - mu( ii, h( mu(ij, ik) ) ) )

    # S vector: 16 ints.
    # i(e_k) -> vector with 1 at k and 1 at k+8.

    def to_vec(idx):
        v = [0]*16
        v[idx] = 1
        v[idx+8] = 1
        return v

    def vec_add(v1, v2):
        return [a + b for a, b in zip(v1, v2, strict=True)]

    def vec_sub(v1, v2):
        return [a - b for a, b in zip(v1, v2, strict=True)]

    # Multiplication of S-vectors
    def s_mul(v1, v2):
        # Unpack
        a = v1[:8]
        b = v1[8:]
        c = v2[:8]
        d = v2[8:]

        # O-multiplication helper (bilinear)
        def o_mul_vec(va, vb):
            res = [0]*8
            for ia, val_a in enumerate(va):
                if val_a == 0:
                    continue
                for ib, val_b in enumerate(vb):
                    if val_b == 0:
                        continue

                    s, k = octonion.mul(1, ia, 1, ib)
                    res[k] += val_a * val_b * s
            return res

        def o_conj_vec(v):
            res = [v[0]] + [-x for x in v[1:]]
            return res

        # ac - d*b
        ac = o_mul_vec(a, c)
        d_star = o_conj_vec(d)
        db = o_mul_vec(d_star, b)
        re = vec_sub(ac, db)

        # da + bc*
        da = o_mul_vec(d, a)
        c_star = o_conj_vec(c)
        bc = o_mul_vec(b, c_star)
        im = vec_add(da, bc)

        return re + im

    # Maps
    def p_map(v):
        # (a+b)/2
        a = v[:8]
        b = v[8:]
        return [(x + y) / 2.0 for x, y in zip(a, b, strict=True)]

    def i_map(o_vec):
        return o_vec + o_vec

    def h_map(v):
        # h(a,b) = ( (a-b)/2, -(a-b)/2 )
        a = v[:8]
        b = v[8:]
        half_diff = [(x - y) / 2.0 for x, y in zip(a, b, strict=True)]
        return half_diff + [-x for x in half_diff]

    # Build inputs
    vx = to_vec(i)
    vy = to_vec(j)
    vz = to_vec(k)

    # Left Tree: p mu( h mu(x, y), z )
    xy = s_mul(vx, vy)
    h_xy = h_map(xy)
    term1_s = s_mul(h_xy, vz)
    term1 = p_map(term1_s)

    # Right Tree: p mu( x, h mu(y, z) )
    yz = s_mul(vy, vz)
    h_yz = h_map(yz)
    term2_s = s_mul(vx, h_yz)
    term2 = p_map(term2_s)

    res = vec_sub(term1, term2)
    return res

def main():
    octonion = Octonion()

    print("Generating m3 table for octonion basis triples (1..7)...")

    # We will iterate all triples of imaginary units 1..7
    # 7*7*7 = 343 entries

    output_filename = "data/csv/m3_table.csv"
    os.makedirs("data/csv", exist_ok=True)

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "k", "m3_type", "value_index", "value_sign"])

        range_idxs = range(1, 8)

        counts = {"0": 0, "scalar": 0, "vector": 0}

        for i in range_idxs:
            for j in range_idxs:
                for k in range_idxs:

                    res = compute_m3(octonion, i, j, k)

                    # Analyze result
                    # Is it 0?
                    norm = sum([abs(x) for x in res])
                    if norm < 1e-9:
                        row = [i, j, k, "0", 0, 0]
                        counts["0"] += 1
                    else:
                        # Is it scalar (index 0 non-zero)?
                        if abs(res[0]) > 1e-9:
                            # Verify only index 0 is set
                            rest = sum([abs(x) for x in res[1:]])
                            if rest < 1e-9:
                                row = [i, j, k, "scalar", 0, res[0]]
                                counts["scalar"] += 1
                            else:
                                row = [i, j, k, "mixed", -1, 0] # Should not happen
                        else:
                            # Must be vector
                            # Find index
                            idx = -1
                            val = 0
                            for idx_check in range(1, 8):
                                if abs(res[idx_check]) > 1e-9:
                                    idx = idx_check
                                    val = res[idx_check]
                                    break
                            row = [i, j, k, "vector", idx, val]
                            counts["vector"] += 1

                    writer.writerow(row)

    print("Classification Summary:")
    print(counts)
    print(f"Done. Saved to {output_filename}")

if __name__ == "__main__":
    main()
