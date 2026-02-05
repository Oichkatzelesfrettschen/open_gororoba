from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gemini_physics.optimized_algebra import cd_multiply_jit


@dataclass(frozen=True)
class AnnihilatorInfo:
    left_nullity: int
    right_nullity: int


def left_multiplication_matrix(a: np.ndarray) -> np.ndarray:
    """
    Build the 16x16 real matrix L_a such that L_a @ b == a*b (in the chosen basis).
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (16,):
        raise ValueError("Expected a 16D vector for a sedenion element.")

    cols = []
    for i in range(16):
        e = np.zeros(16)
        e[i] = 1.0
        cols.append(cd_multiply_jit(a, e, 16))
    return np.stack(cols, axis=1)


def right_multiplication_matrix(a: np.ndarray) -> np.ndarray:
    """
    Build the 16x16 real matrix R_a such that R_a @ b == b*a (in the chosen basis).
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (16,):
        raise ValueError("Expected a 16D vector for a sedenion element.")

    cols = []
    for i in range(16):
        e = np.zeros(16)
        e[i] = 1.0
        cols.append(cd_multiply_jit(e, a, 16))
    return np.stack(cols, axis=1)


def nullspace_basis(mat: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """
    Return an orthonormal basis for the (right) nullspace of mat using SVD.

    The returned matrix has shape (n, k) where each column is a basis vector.
    """
    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    rank = int(np.sum(s > atol))
    n = mat.shape[1]
    if rank == n:
        return np.zeros((n, 0))
    # For a matrix m x n, vh has shape (n, n); nullspace basis are rows rank..n-1 of vh.
    ns = vh[rank:].T
    return ns


def annihilator_info(a: np.ndarray, *, atol: float = 1e-12) -> AnnihilatorInfo:
    """
    Return dimensions of left/right annihilators of a (as nullities of L_a and R_a).
    """
    la = left_multiplication_matrix(a)
    ra = right_multiplication_matrix(a)
    left_nullity = nullspace_basis(la, atol=atol).shape[1]
    right_nullity = nullspace_basis(ra, atol=atol).shape[1]
    return AnnihilatorInfo(left_nullity=left_nullity, right_nullity=right_nullity)


def is_zero_divisor(a: np.ndarray, *, atol: float = 1e-12) -> bool:
    info = annihilator_info(a, atol=atol)
    return info.left_nullity > 0 or info.right_nullity > 0


def is_reggiani_zd(a: np.ndarray, *, atol: float = 1e-12) -> bool:
    """
    Check membership in Reggiani's ZD(S) normalization convention.

    In Reggiani (2024), ZD(S) is described as the submanifold of normalized sedenions with
    non-trivial annihilators. The paper states a "norm 2" convention; in this repo we implement
    this as squared Euclidean norm 2 (i.e. ||u|| = sqrt2), matching the diagonal-form zero divisors
    (e_i +/- e_j).
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (16,):
        raise ValueError("Expected a 16D vector for a sedenion element.")
    if not np.isclose(float(np.dot(a, a)), 2.0, atol=atol, rtol=0.0):
        return False
    return is_zero_divisor(a, atol=atol)


def find_left_annihilator_vector(a: np.ndarray, *, atol: float = 1e-12) -> np.ndarray | None:
    """
    Return a non-zero b such that a*b = 0, if one is found (via nullspace of L_a).
    """
    la = left_multiplication_matrix(a)
    ns = nullspace_basis(la, atol=atol)
    if ns.shape[1] == 0:
        return None
    return ns[:, 0]
