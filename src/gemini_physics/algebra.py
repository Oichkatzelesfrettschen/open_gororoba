import numpy as np


class CayleyDickson:
    """
    Implements the Cayley-Dickson construction for high-dimensional algebras.

    Terminology:
    - 1D: Reals (Field, Ordered, Commutative, Associative)
    - 2D: Complex (Field, Commutative, Associative)
    - 4D: Quaternions (Division Ring, Non-Commutative, Associative)
    - 8D: Octonions (Non-Associative Division Algebra, Alternative)
    - 16D: Sedenions (Non-Associative, Non-Alternative, Has Zero Divisors)
    """
    def __init__(self, dimension):
        self.dimension = dimension
        if dimension & (dimension - 1) != 0:
            raise ValueError("Dimension must be a power of 2")

    @staticmethod
    def conjugate(a):
        """Recursively conjugate an element: (L, R)* = (L*, -R)"""
        if isinstance(a, (int, float)):
            return a
        return (CayleyDickson.conjugate(a[0]), CayleyDickson.negate(a[1]))

    @staticmethod
    def negate(a):
        if isinstance(a, (int, float)):
            return -a
        return (CayleyDickson.negate(a[0]), CayleyDickson.negate(a[1]))

    @staticmethod
    def add(a, b):
        if isinstance(a, (int, float)):
            return a + b
        return (CayleyDickson.add(a[0], b[0]), CayleyDickson.add(a[1], b[1]))

    @staticmethod
    def multiply(a, b):
        """
        Cayley-Dickson Multiplication: (a,b)(c,d) = (ac - d*b, da + bc*)
        """
        if isinstance(a, (int, float)):
            return a * b

        aL, aR = a
        c, d = b

        d_star = CayleyDickson.conjugate(d)
        c_star = CayleyDickson.conjugate(c)

        # L = ac - d*b
        term1 = CayleyDickson.multiply(aL, c)
        term2 = CayleyDickson.multiply(d_star, aR)
        L = CayleyDickson.add(term1, CayleyDickson.negate(term2))

        # R = da + bc*
        term3 = CayleyDickson.multiply(d, aL)
        term4 = CayleyDickson.multiply(aR, c_star)
        R = CayleyDickson.add(term3, term4)

        return (L, R)

    @staticmethod
    def magnitude_sq(a):
        """Returns the squared Euclidean norm."""
        if isinstance(a, (int, float)):
            return a * a
        return CayleyDickson.magnitude_sq(a[0]) + CayleyDickson.magnitude_sq(a[1])

    @staticmethod
    def random_element(dim, scale=1.0):
        raise NotImplementedError(
            "Random element generation was removed to avoid non-reproducible "
            "'validation' in this repo. "
            "Use the vector-based implementation in `gemini_physics.optimized_algebra` "
            "and seeded RNGs in tests."
        )
        return (
            CayleyDickson.random_element(dim // 2, scale),
            CayleyDickson.random_element(dim // 2, scale),
        )

    @staticmethod
    def random_unit_element(dim):
        raise NotImplementedError(
            "Random unit generation was removed to avoid non-reproducible "
            "'validation' in this repo. "
            "Use the vector-based implementation in `gemini_physics.optimized_algebra` "
            "and seeded RNGs in tests."
        )

def sedenion_zero_divisor_example() -> tuple[np.ndarray, np.ndarray]:
    """
    Return an explicit sedenion zero-divisor pair (a,b) in R^16 (vector basis) such that a*b = 0.

    Convention: basis e0..e15 with e0 as the real unit and e1..e15 imaginary units, using the
    standard Cayley-Dickson multiplication convention where e1*e2=e3 and e1*e4=e5.
    """
    a = np.zeros(16)
    b = np.zeros(16)
    a[1] = 1.0
    a[10] = 1.0
    b[4] = 1.0
    b[15] = -1.0
    return a, b

if __name__ == "__main__":
    from gemini_physics.optimized_algebra import cd_multiply_jit

    a, b = sedenion_zero_divisor_example()
    prod = cd_multiply_jit(a, b, 16)
    print("||a*b|| =", float(np.linalg.norm(prod)))
