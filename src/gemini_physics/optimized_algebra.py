"""
Optimized Cayley-Dickson algebra operations.

This module provides a unified interface for CD multiplication,
preferring the Rust implementation (gororoba_py) when available,
falling back to the Numba JIT version otherwise.
"""
import time

import numpy as np

# Try to import Rust bindings first
_USE_RUST = False
try:
    import gororoba_py as _gp
    _USE_RUST = True
except ImportError:
    pass

if _USE_RUST:
    def cd_multiply_jit(a, b, dim):
        """
        Cayley-Dickson multiplication using Rust backend.
        Input arrays a and b are flattened representations of the algebra elements.
        """
        # gororoba_py expects lists/arrays without explicit dim parameter
        result = _gp.py_cd_multiply(list(a), list(b))
        return np.asarray(result)

    def cd_conjugate(a):
        """Cayley-Dickson conjugate using Rust backend."""
        result = _gp.py_cd_conjugate(list(a))
        return np.asarray(result)

    def cd_norm(a):
        """Cayley-Dickson norm using Rust backend."""
        return _gp.py_cd_norm(list(a))

    def cd_associator_norm(a, b, c):
        """Compute ||(ab)c - a(bc)|| using Rust backend."""
        return _gp.py_cd_associator_norm(list(a), list(b), list(c))

else:
    from numba import njit

    @njit
    def cd_multiply_jit(a, b, dim):
        """
        Highly optimized recursive Cayley-Dickson multiplication using Numba.
        Input arrays a and b are flattened representations of the algebra elements.
        """
        if dim == 1:
            return np.array([a[0] * b[0]])

        half = dim // 2
        aL = a[:half]
        aR = a[half:]
        cL = b[:half]
        cR = b[half:]

        # (a,b)(c,d) = (ac - d*b, da + bc*)

        # Helper: conjugate
        def conjugate(x):
            res = x.copy()
            res[1:] = -res[1:]
            return res

        # L = ac - d*b
        # term1 = ac
        term1 = cd_multiply_jit(aL, cL, half)
        # term2 = d*b
        term2 = cd_multiply_jit(conjugate(cR), aR, half)
        L = term1 - term2

        # R = da + bc*
        # term3 = da
        term3 = cd_multiply_jit(cR, aL, half)
        # term4 = bc*
        term4 = cd_multiply_jit(aR, conjugate(cL), half)
        R = term3 + term4

        # Combine
        res = np.zeros(dim)
        res[:half] = L
        res[half:] = R
        return res

    def cd_conjugate(a):
        """Cayley-Dickson conjugate (pure Python fallback)."""
        res = a.copy()
        res[1:] = -res[1:]
        return res

    def cd_norm(a):
        """Cayley-Dickson norm (pure Python fallback)."""
        return np.sqrt(np.sum(a * a))

    def cd_associator_norm(a, b, c):
        """Compute ||(ab)c - a(bc)|| (pure Python fallback)."""
        dim = len(a)
        ab = cd_multiply_jit(a, b, dim)
        abc1 = cd_multiply_jit(ab, c, dim)
        bc = cd_multiply_jit(b, c, dim)
        abc2 = cd_multiply_jit(a, bc, dim)
        return np.linalg.norm(abc1 - abc2)


def measure_associator_density(dim, trials=5000):
    """
    Quantifies non-associativity: % of triples (a,b,c) where (ab)c != a(bc).
    This is Step 2 of the Gemini Roadmap.
    """
    print(f"Measuring Associator Density for Dimension {dim}...")
    failures = 0
    start_time = time.time()

    for _ in range(trials):
        a = np.random.uniform(-1, 1, dim)
        b = np.random.uniform(-1, 1, dim)
        c = np.random.uniform(-1, 1, dim)

        # (ab)c
        ab = cd_multiply_jit(a, b, dim)
        abc_1 = cd_multiply_jit(ab, c, dim)

        # a(bc)
        bc = cd_multiply_jit(b, c, dim)
        abc_2 = cd_multiply_jit(a, bc, dim)

        if not np.allclose(abc_1, abc_2, atol=1e-8):
            failures += 1

    end_time = time.time()
    density = (failures / trials) * 100
    print(f"Density: {density:.2f}% ({failures}/{trials} failures)")
    print(f"Time taken: {end_time - start_time:.2f}s")
    backend = "Rust (gororoba_py)" if _USE_RUST else "Python (Numba)"
    print(f"Backend: {backend}")
    return density


if __name__ == "__main__":
    # Test 8D (Octonions) and 16D (Sedenions)
    measure_associator_density(8)
    measure_associator_density(16)
