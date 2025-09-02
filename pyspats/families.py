"""
Exponential family distributions and link functions for SpATS.
"""

import numpy as np
from abc import ABC, abstractmethod


class Family(ABC):
    """Base class for exponential family distributions."""
    
    @property
    @abstractmethod
    def family(self):
        """Family name."""
        pass
    
    @property
    @abstractmethod 
    def link(self):
        """Link function name."""
        pass
    
    @abstractmethod
    def inverse_link(self, eta):
        """Inverse link function: g^-1(eta) = mu."""
        pass
    
    @abstractmethod
    def d_inverse_link(self, eta):
        """Derivative of inverse link: d(mu)/d(eta)."""
        pass
    
    @abstractmethod
    def variance(self, mu):
        """Variance function: V(mu)."""
        pass
    
    @abstractmethod
    def deviance(self, y, mu, weights=None):
        """Deviance function."""
        pass
    
    @abstractmethod
    def initialize(self, y, weights=None):
        """Initialize mu and eta for IRLS."""
        pass


class GaussianFamily(Family):
    """Gaussian family with identity link."""
    
    def __init__(self, link='identity'):
        if link != 'identity':
            raise NotImplementedError(f"Link '{link}' not implemented for Gaussian")
        self._link = link
    
    @property
    def family(self):
        return 'gaussian'
    
    @property 
    def link(self):
        return self._link
    
    def inverse_link(self, eta):
        """Identity link: mu = eta."""
        return eta
    
    def d_inverse_link(self, eta):
        """Derivative of identity link: 1."""
        return np.ones_like(eta)
    
    def variance(self, mu):
        """Constant variance: 1."""
        return np.ones_like(mu)
    
    def deviance(self, y, mu, weights=None):
        """Gaussian deviance: sum of squared residuals."""
        if weights is None:
            weights = np.ones_like(y)
        return np.sum(weights * (y - mu)**2)
    
    def initialize(self, y, weights=None):
        """Initialize with observed values."""
        if weights is None:
            weights = np.ones_like(y)
        
        # Handle zeros in weights
        valid = weights > 0
        mu = np.where(valid, y, np.mean(y[valid]))
        eta = mu.copy()  # Identity link
        
        return mu, eta


class BinomialFamily(Family):
    """Binomial family with logit link."""
    
    def __init__(self, link='logit'):
        if link != 'logit':
            raise NotImplementedError(f"Link '{link}' not implemented for Binomial")
        self._link = link
    
    @property
    def family(self):
        return 'binomial'
    
    @property
    def link(self):
        return self._link
    
    def inverse_link(self, eta):
        """Logit inverse link: mu = exp(eta)/(1 + exp(eta))."""
        # Stable computation
        exp_eta = np.exp(np.clip(eta, -700, 700))
        return exp_eta / (1 + exp_eta)
    
    def d_inverse_link(self, eta):
        """Derivative of logit inverse link."""
        mu = self.inverse_link(eta)
        return mu * (1 - mu)
    
    def variance(self, mu):
        """Binomial variance: mu * (1 - mu)."""
        return mu * (1 - mu)
    
    def deviance(self, y, mu, weights=None):
        """Binomial deviance."""
        if weights is None:
            weights = np.ones_like(y)
        
        # Avoid log(0)
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        
        dev = np.zeros_like(y)
        nonzero_y = y > 0
        nonone_y = y < 1
        
        dev[nonzero_y] += y[nonzero_y] * np.log(y[nonzero_y] / mu[nonzero_y])
        dev[nonone_y] += (1 - y[nonone_y]) * np.log((1 - y[nonone_y]) / (1 - mu[nonone_y]))
        
        return 2 * np.sum(weights * dev)
    
    def initialize(self, y, weights=None):
        """Initialize mu and eta for binomial."""
        if weights is None:
            weights = np.ones_like(y)
        
        # Adjust y to avoid 0 and 1
        mu = np.clip(y, 1e-3, 1 - 1e-3)
        eta = np.log(mu / (1 - mu))  # logit link
        
        return mu, eta


class PoissonFamily(Family):
    """Poisson family with log link."""
    
    def __init__(self, link='log'):
        if link != 'log':
            raise NotImplementedError(f"Link '{link}' not implemented for Poisson")
        self._link = link
    
    @property
    def family(self):
        return 'poisson'
    
    @property
    def link(self):
        return self._link
    
    def inverse_link(self, eta):
        """Log inverse link: mu = exp(eta)."""
        return np.exp(np.clip(eta, -700, 700))
    
    def d_inverse_link(self, eta):
        """Derivative of log inverse link: mu."""
        return self.inverse_link(eta)
    
    def variance(self, mu):
        """Poisson variance: mu."""
        return mu
    
    def deviance(self, y, mu, weights=None):
        """Poisson deviance."""
        if weights is None:
            weights = np.ones_like(y)
        
        # Avoid log(0)
        mu = np.maximum(mu, 1e-15)
        
        dev = np.zeros_like(y)
        nonzero_y = y > 0
        dev[nonzero_y] = y[nonzero_y] * np.log(y[nonzero_y] / mu[nonzero_y])
        dev -= y - mu
        
        return 2 * np.sum(weights * dev)
    
    def initialize(self, y, weights=None):
        """Initialize mu and eta for Poisson."""
        if weights is None:
            weights = np.ones_like(y)
        
        # Ensure positive mu
        mu = np.maximum(y, 0.1)
        eta = np.log(mu)  # log link
        
        return mu, eta


# Convenience functions
def gaussian(link='identity'):
    """Gaussian family with identity link."""
    return GaussianFamily(link)

def binomial(link='logit'):
    """Binomial family with logit link.""" 
    return BinomialFamily(link)

def poisson(link='log'):
    """Poisson family with log link."""
    return PoissonFamily(link)