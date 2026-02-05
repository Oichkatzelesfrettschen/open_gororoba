import numpy as np

# --- 1. Cayley-Dickson Algebra Engine ---

class CDAlgebra:
    def __init__(self, n):
        self.n = n
        self.dim = 2**n
        # Multiplication table can be cached or computed on fly.
        # For n=3 (Oct) and n=4 (Sed), we compute on fly to ensure recursion is exact.

    def multiply(self, x, y):
        # x, y are arrays of size dim
        if self.n == 0: # Reals
            return x * y

        half = self.dim // 2
        a = x[:half]; b = x[half:]
        c = y[:half]; d = y[half:]

        # Recursive step to n-1
        sub_alg = CDAlgebra(self.n - 1)

        # (a,b)(c,d) = (ac - d*b, da + bc*)
        ac = sub_alg.multiply(a, c)
        db = sub_alg.multiply(sub_alg.conjugate(d), b)
        da = sub_alg.multiply(d, a)
        bc = sub_alg.multiply(b, sub_alg.conjugate(c))

        res_real = ac - db
        res_imag = da + bc

        return np.concatenate((res_real, res_imag))

    def conjugate(self, x):
        if self.n == 0:
            return x # Real conjugation is identity

        half = self.dim // 2
        a = x[:half]
        b = x[half:]

        # (a,b)* = (a*, -b)
        sub_alg = CDAlgebra(self.n - 1)
        a_star = sub_alg.conjugate(a)
        b_neg = -b

        return np.concatenate((a_star, b_neg))

    def basis(self, i):
        v = np.zeros(self.dim)
        v[i] = 1.0
        return v

# --- 2. Projection, Section, Homotopy ---

def p_map(s_vec):
    # s is 16D (Sedenion). p: S -> O
    # p(a,b) = (a+b)/2
    a = s_vec[:8]
    b = s_vec[8:]
    return (a + b) / 2.0

def i_map(o_vec):
    # i: O -> S
    # i(x) = (x,x)
    return np.concatenate((o_vec, o_vec))

def h_map(s_vec):
    # h: S -> S (mapping to kernel of p)
    # h(s) = s - i(p(s))
    # h(a,b) = ( (a-b)/2, -(a-b)/2 )
    a = s_vec[:8]
    b = s_vec[8:]
    diff = (a - b) / 2.0
    return np.concatenate((diff, -diff))

# --- 3. The Transfer Formula ---

def m3_transfer(x, y, z, alg_s):
    # m3(x,y,z) = p( mu( h( mu(ix, iy) ), iz ) - mu( ix, h( mu(iy, iz) ) ) )
    # inputs x,y,z are in O (8D)

    ix = i_map(x)
    iy = i_map(y)
    iz = i_map(z)

    # Left Branch: mu( h( mu(ix, iy) ), iz )
    ix_iy = alg_s.multiply(ix, iy)
    h_ix_iy = h_map(ix_iy)
    left_inner = alg_s.multiply(h_ix_iy, iz)

    # Right Branch: mu( ix, h( mu(iy, iz) ) )
    iy_iz = alg_s.multiply(iy, iz)
    h_iy_iz = h_map(iy_iz)
    right_inner = alg_s.multiply(ix, h_iy_iz)

    # Combine
    diff = left_inner - right_inner
    result = p_map(diff)

    return result

# --- 4. Execution ---

if __name__ == "__main__":
    O_alg = CDAlgebra(3)
    S_alg = CDAlgebra(4)

    e1 = O_alg.basis(1)
    e2 = O_alg.basis(2)
    e3 = O_alg.basis(3)
    e4 = O_alg.basis(4)

    print("--- Computing m3 Associator for CD(O)->S ---")

    # Case A: (e1, e2, e3)
    res_A = m3_transfer(e1, e2, e3, S_alg)
    print("\nCase A: m3(e1, e2, e3)")
    print("Expected: -2.0 scalar (on real component)")
    print(f"Result: {res_A}")

    # Case B: (e1, e2, e4)
    res_B = m3_transfer(e1, e2, e4, S_alg)
    print("\nCase B: m3(e1, e2, e4)")
    print("Scanning for non-zero components...")
    # Identify non-zero components
    found = False
    for k in range(8):
        if abs(res_B[k]) > 1e-9:
            print(f"Component e{k}: {res_B[k]}")
            found = True
    if not found:
        print("Result is effectively zero.")
