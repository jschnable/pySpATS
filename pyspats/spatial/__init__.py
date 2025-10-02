"""
Spatial modeling utilities for pySpATS.

This subpackage contains efficient implementations for spatial P-spline modeling:
- Kronecker-structured tensor product bases
- Lazy linear operators for large interaction terms
- Whitened reparameterization for separable P-splines
- Polynomial projection for PS-ANOVA orthogonality
- Gram-matrix-based Z'Z and X'Z computation
"""

from .kron_utils import (
    kron_matvec,
    kron_rmatvec,
    kron_linear_operator,
)

from .projection import (
    LeftProjectedLO,
)

from .gram_kron import (
    compute_GrGc,
    kron_whitened_ZtZ,
    projected_ZtZ,
    XtZ_poly_zero,
    XtZ_other,
    build_interaction_gram_terms,
)

__all__ = [
    "kron_matvec",
    "kron_rmatvec",
    "kron_linear_operator",
    "LeftProjectedLO",
    "compute_GrGc",
    "kron_whitened_ZtZ",
    "projected_ZtZ",
    "XtZ_poly_zero",
    "XtZ_other",
    "build_interaction_gram_terms",
]
