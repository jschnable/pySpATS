"""
Polynomial projection utilities for PS-ANOVA orthogonality.

This module provides LinearOperator wrappers that enforce orthogonality
to polynomial fixed effects by applying left projection P_⊥ = I - Qx Qx^T.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import LinearOperator


class LeftProjectedLO(LinearOperator):
    """
    LinearOperator wrapper that projects out polynomial space on the left.

    For a base operator Z and orthonormal basis Qx for the fixed polynomial space,
    this operator computes:

        matvec(v):  P_⊥ @ (Z @ v) = (I - Qx Qx^T) @ (Z @ v)
        rmatvec(w): Z^T @ (P_⊥ @ w) = Z^T @ ((I - Qx Qx^T) @ w)

    This ensures the random effect Z is orthogonal to the polynomial space,
    satisfying the PS-ANOVA decomposition requirement X^T Z = 0.

    Parameters
    ----------
    base : LinearOperator
        Base operator to wrap (e.g., Kronecker interaction operator)
    Qx : np.ndarray
        Orthonormal basis for polynomial space (n_obs × q)
        Typically from QR decomposition of X_poly = [1, r, c]

    Attributes
    ----------
    base : LinearOperator
        The wrapped base operator
    Qx : np.ndarray
        Orthonormal polynomial basis

    Notes
    -----
    The projection operator P_⊥ = I - Qx Qx^T is idempotent and symmetric:
    - P_⊥^2 = P_⊥
    - P_⊥^T = P_⊥

    For matvec: y = Z @ v, then return P_⊥ @ y = y - Qx (Qx^T y)
    For rmatvec: w_perp = P_⊥ @ w = w - Qx (Qx^T w), then return Z^T @ w_perp

    Memory efficiency: Only stores Qx (typically n_obs × 3), never forms I or P_⊥.

    Examples
    --------
    >>> X_poly = np.column_stack([np.ones(100), np.arange(100), np.arange(100)])
    >>> Qx, _ = np.linalg.qr(X_poly, mode='reduced')
    >>> Z_raw = LinearOperator((100, 50), matvec=..., rmatvec=...)
    >>> Z_projected = LeftProjectedLO(Z_raw, Qx)
    >>> # Now Z_projected @ v is orthogonal to X_poly columns
    """

    def __init__(self, base: LinearOperator, Qx: np.ndarray):
        """
        Initialize left-projected LinearOperator.

        Parameters
        ----------
        base : LinearOperator
            Base operator to wrap
        Qx : np.ndarray
            Orthonormal basis for polynomial space (n_obs × q)
        """
        if base.dtype != np.float64:
            raise ValueError(f"Base operator must have dtype float64, got {base.dtype}")

        if not isinstance(Qx, np.ndarray):
            raise TypeError("Qx must be a numpy array")

        if Qx.shape[0] != base.shape[0]:
            raise ValueError(
                f"Qx rows ({Qx.shape[0]}) must match base operator rows ({base.shape[0]})"
            )

        self.base = base
        self.Qx = Qx

        # Initialize parent LinearOperator
        super().__init__(dtype=np.float64, shape=base.shape)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Compute P_⊥ @ (Z @ v) = (I - Qx Qx^T) @ (Z @ v).

        Parameters
        ----------
        v : np.ndarray
            Input vector (shape: base.shape[1])

        Returns
        -------
        np.ndarray
            Projected result: (I - Qx Qx^T) @ (base @ v)
        """
        # Compute y = Z @ v
        y = self.base @ v

        # Apply left projection: P_⊥ @ y = y - Qx (Qx^T @ y)
        # This removes any component of y in the polynomial space
        y_perp = y - self.Qx @ (self.Qx.T @ y)

        return y_perp

    def _rmatvec(self, w: np.ndarray) -> np.ndarray:
        """
        Compute Z^T @ (P_⊥ @ w) = Z^T @ ((I - Qx Qx^T) @ w).

        Parameters
        ----------
        w : np.ndarray
            Input vector (shape: base.shape[0])

        Returns
        -------
        np.ndarray
            Adjoint projected result: base^T @ ((I - Qx Qx^T) @ w)
        """
        # Apply left projection to input: w_perp = P_⊥ @ w = w - Qx (Qx^T @ w)
        w_perp = w - self.Qx @ (self.Qx.T @ w)

        # Compute Z^T @ w_perp
        result = self.base.rmatvec(w_perp)

        return result

    def _matmat(self, V: np.ndarray) -> np.ndarray:
        """
        Compute P_⊥ @ (Z @ V) for matrix V.

        Parameters
        ----------
        V : np.ndarray
            Input matrix (shape: base.shape[1] × k)

        Returns
        -------
        np.ndarray
            Projected result: (I - Qx Qx^T) @ (base @ V)
        """
        # Compute Y = Z @ V
        Y = self.base @ V

        # Apply left projection to each column
        Y_perp = Y - self.Qx @ (self.Qx.T @ Y)

        return Y_perp

    def _rmatmat(self, W: np.ndarray) -> np.ndarray:
        """
        Compute Z^T @ (P_⊥ @ W) for matrix W.

        Parameters
        ----------
        W : np.ndarray
            Input matrix (shape: base.shape[0] × k)

        Returns
        -------
        np.ndarray
            Adjoint projected result: base^T @ ((I - Qx Qx^T) @ W)
        """
        # Apply left projection to each column
        W_perp = W - self.Qx @ (self.Qx.T @ W)

        # Compute Z^T @ W_perp
        result = self.base.rmatvec(W_perp) if W_perp.shape[1] == 1 else self.base.T @ W_perp

        return result
