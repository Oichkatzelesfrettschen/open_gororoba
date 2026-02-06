"""
gemini_physics

Small, testable kernels backing the repository's experiments.
"""

# Core algebra operations now use Rust backend via gororoba_py when available
from .optimized_algebra import cd_multiply_jit, cd_conjugate, cd_norm  # noqa: F401
