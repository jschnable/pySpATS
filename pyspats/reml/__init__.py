"""
REML (Restricted Maximum Likelihood) estimation with ED-based variance updates.

This module provides efficient REML estimation that:
- Reuses CHOLMOD factorization per iteration
- Updates variance components using exact effective dimensions (ED)
- Leverages Takahashi selected inverse for ED computation

Key components:
- REMLOptions: Configuration for REML optimizer
- REMLResult: Result container with variance components and EDs
- fit_reml(): Main REML optimization routine
"""

from .optimizer import fit_reml, REMLOptions, REMLResult

__all__ = ['fit_reml', 'REMLOptions', 'REMLResult']
