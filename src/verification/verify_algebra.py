import random


def multiply(a, b):
    """
    Recursive Cayley-Dickson multiplication.
    Elements are tuples (left, right).
    Base case: floats/ints.
    (a,b)(c,d) = (ac - d*b, da + bc*)
    where * is conjugate.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b

    # Unpack tuples
    a_L, a_R = a
    c, d = b # matching (c,d) from formula

    # Conjugate helper
    def conjugate(x):
        if isinstance(x, (int, float)):
            return x
        l, r = x
        return (conjugate(l), negate(r))

    def negate(x):
        if isinstance(x, (int, float)):
            return -x
        l, r = x
        return (negate(l), negate(r))

    # Recursive steps
    # L = ac - d*b
    # R = da + bc*

    # We need to handle the recursive structure carefully.
    # Note: d* is conjugate(d)

    term1 = multiply(a_L, c)
    term2 = multiply(conjugate(d), a_R)
    L = add(term1, negate(term2)) # ac - d*b

    term3 = multiply(d, a_L)
    term4 = multiply(a_R, conjugate(c))
    R = add(term3, term4) # da + bc*

    return (L, R)

def add(a, b):
    if isinstance(a, (int, float)):
        return a + b
    return (add(a[0], b[0]), add(a[1], b[1]))

def generate_random_element(depth):
    if depth == 0:
        return random.randint(-10, 10)
    return (generate_random_element(depth-1), generate_random_element(depth-1))

def check_associativity(depth, trials=100):
    """Checks (ab)c == a(bc)"""
    failures = 0
    for _ in range(trials):
        a = generate_random_element(depth)
        b = generate_random_element(depth)
        c = generate_random_element(depth)

        ab_c = multiply(multiply(a, b), c)
        a_bc = multiply(a, multiply(b, c))

        if ab_c != a_bc:
            failures += 1

    return failures

if __name__ == "__main__":
    print("Testing Cayley-Dickson Associativity:")
    # 0 = Re (1D), 1 = Complex (2D), 2 = Quaternion (4D), 3 = Octonion (8D), 4 = Sedenion (16D)
    dims = {0: "Real (1D)", 1: "Complex (2D)", 2: "Quaternion (4D)", 3: "Octonion (8D)", 4: "Sedenion (16D)"}

    for d in range(5):
        fails = check_associativity(d, trials=50)
        status = "Associative" if fails == 0 else f"Non-Associative ({fails} failures)"
        print(f"Dimension {2**d} ({dims[d]}): {status}")

    print("\nHypothesis for 256D:")
    if check_associativity(3, 1) > 0: # If Octonions fail
        print("Since 8D is non-associative, 256D (which contains 8D) MUST be non-associative.")
