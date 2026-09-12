"""Validated iteration controls shared by Gaussian SAP and working GLMM fits."""

from dataclasses import dataclass
import math


@dataclass
class SpATSControl:
    """SAP uses absolute change in the R restricted objective for convergence.

    Non-Gaussian fits additionally test relative squared change in the linear
    predictor. ``max_iter`` bounds both loops. Reaching a bound emits a warning
    and returns ``converged=False``. Dispersion is estimated for Gaussian and
    fixed to one for Poisson/binomial unless explicitly enabled.
    """

    tolerance: float = 1e-6
    max_iter: int = 200
    monitoring: bool = False
    update_psi: bool = False
    update_psi_gauss: bool = True

    def __post_init__(self):
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, int)
            or self.max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
