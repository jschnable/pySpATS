"""
Control parameters for SpATS algorithm.
"""

class SpATSControl:
    """
    Control parameters for SpATS model fitting algorithm.
    
    Parameters
    ----------
    tolerance : float, default=1e-4
        Convergence tolerance for the algorithm
    max_iter : int, default=200
        Maximum number of iterations
    monitoring : bool, default=False
        Whether to print iteration progress
    update_psi : bool, default=False
        Whether to update dispersion parameter for non-Gaussian families
    update_psi_gauss : bool, default=True
        Whether to update dispersion parameter for Gaussian family
    """
    
    def __init__(
        self,
        tolerance: float = 1e-4,
        max_iter: int = 200, 
        monitoring: bool = False,
        update_psi: bool = False,
        update_psi_gauss: bool = True
    ):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.monitoring = monitoring
        self.update_psi = update_psi
        self.update_psi_gauss = update_psi_gauss
        
    def __repr__(self):
        return (f"SpATSControl(tolerance={self.tolerance}, max_iter={self.max_iter}, "
                f"monitoring={self.monitoring})")