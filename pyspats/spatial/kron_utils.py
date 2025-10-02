"""
Kronecker product utilities for efficient tensor product operations.

This module provides lazy evaluation of Kronecker products (B_r ⊗ B_c) without
materializing the full tensor product matrix. This is crucial for memory efficiency
when dealing with large spatial grids.

For P-spline interaction terms, we use the identity:
    (B_r ⊗ B_c) @ vec(X) = vec(B_r @ X @ B_c^T)

where X is reshaped from the coefficient vector. This allows us to work with
the much smaller B_r and B_c matrices directly.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from typing import Tuple


def kron_matvec(
    B_r: sp.spmatrix,
    B_c: sp.spmatrix,
    x: np.ndarray,
    shape_rc: Tuple[int, int]
) -> np.ndarray:
    """
    Compute (B_r ⊗ B_c) @ x without explicit Kronecker product.

    Uses the identity: (B_r ⊗ B_c) @ vec(X) = vec(B_r @ X @ B_c^T)
    where X = reshape(x, (p_r, p_c), order="C") and vec uses C-order (row-major).

    Parameters
    ----------
    B_r : scipy.sparse matrix
        Row basis matrix (n_r × p_r)
    B_c : scipy.sparse matrix
        Column basis matrix (n_c × p_c)
    x : np.ndarray
        Coefficient vector (p_r * p_c,)
    shape_rc : tuple of int
        Output shape (n_r, n_c) for the evaluated field

    Returns
    -------
    np.ndarray
        Result vector (n_r * n_c,), flattened in C-order (row-major)

    Notes
    -----
    Uses C-order (row-major) vec operation to match scipy.sparse.kron convention.
    The identity (A ⊗ B) @ vec(X) = vec(A @ X @ B.T) holds when vec uses
    row-major ordering.

    Examples
    --------
    >>> B_r = sp.eye(3, format='csc')
    >>> B_c = sp.eye(2, format='csc')
    >>> x = np.arange(6.0)
    >>> y = kron_matvec(B_r, B_c, x, (3, 2))
    >>> y_dense = sp.kron(B_r, B_c).toarray() @ x
    >>> np.allclose(y, y_dense)
    True
    """
    p_r, p_c = B_r.shape[1], B_c.shape[1]
    n_r, n_c = shape_rc

    # Reshape coefficient vector to matrix (C order for kron consistency)
    # For kron(B_r, B_c), we use: vec(B_r @ X @ B_c.T) with C-order vec
    X = x.reshape(p_r, p_c, order="C")

    # Compute B_r @ X @ B_c^T efficiently
    # Result is (n_r × n_c) matrix representing the evaluated field
    Y = B_r @ X @ (B_c.T)

    # Convert to dense if sparse (result of sparse matrix multiplication can be sparse or dense)
    if sp.issparse(Y):
        Y = Y.toarray()

    # Flatten result in C order
    return Y.ravel(order="C")


def kron_rmatvec(
    B_r: sp.spmatrix,
    B_c: sp.spmatrix,
    y: np.ndarray,
    shape_rc: Tuple[int, int]
) -> np.ndarray:
    """
    Compute (B_r ⊗ B_c)^T @ y without explicit Kronecker product.

    Uses the transpose identity: (B_r ⊗ B_c)^T @ vec(Y) = vec(B_r^T @ Y @ B_c)
    where Y = reshape(y, (n_r, n_c), order="C") and vec uses C-order (row-major).

    Parameters
    ----------
    B_r : scipy.sparse matrix
        Row basis matrix (n_r × p_r)
    B_c : scipy.sparse matrix
        Column basis matrix (n_c × p_c)
    y : np.ndarray
        Input vector (n_r * n_c,)
    shape_rc : tuple of int
        Shape (n_r, n_c) to reshape y into

    Returns
    -------
    np.ndarray
        Result vector (p_r * p_c,), flattened in C-order (row-major)

    Notes
    -----
    This implements the adjoint operation for kron_matvec, crucial for
    computing gradients and forming normal equations in least squares.
    Uses C-order vec to match scipy.sparse.kron convention.

    Examples
    --------
    >>> B_r = sp.eye(3, format='csc')
    >>> B_c = sp.eye(2, format='csc')
    >>> y = np.arange(6.0)
    >>> x = kron_rmatvec(B_r, B_c, y, (3, 2))
    >>> x_dense = sp.kron(B_r, B_c).T.toarray() @ y
    >>> np.allclose(x, x_dense)
    True
    """
    p_r, p_c = B_r.shape[1], B_c.shape[1]
    n_r, n_c = shape_rc

    # Reshape input vector to field matrix (C order for kron consistency)
    Y = y.reshape(n_r, n_c, order="C")

    # Compute B_r^T @ Y @ B_c
    # Result is (p_r × p_c) matrix of coefficients
    X = (B_r.T @ Y) @ B_c

    # Convert to dense if sparse
    if sp.issparse(X):
        X = X.toarray()

    # Flatten result in C order
    return X.ravel(order="C")


def kron_linear_operator(
    B_r: sp.spmatrix,
    B_c: sp.spmatrix,
    n_r: int,
    n_c: int
) -> LinearOperator:
    """
    Create a LinearOperator for (B_r ⊗ B_c) without materialization.

    Parameters
    ----------
    B_r : scipy.sparse matrix
        Row basis matrix (n_r × p_r)
    B_c : scipy.sparse matrix
        Column basis matrix (n_c × p_c)
    n_r : int
        Number of rows in the output field
    n_c : int
        Number of columns in the output field

    Returns
    -------
    LinearOperator
        Lazy Kronecker product operator with shape (n_r*n_c, p_r*p_c)

    Notes
    -----
    The LinearOperator supports matrix-vector and matrix-transpose-vector
    products without ever forming the full Kronecker product matrix.

    For a field with n_r=100 rows and n_c=100 columns, and bases with
    p_r=50, p_c=50 coefficients, this saves:
        Memory: (100*100) * (50*50) * 8 bytes = 2 GB (dense)
        vs.     (100*50 + 100*50) * nnz * 8 bytes ≈ 8 MB (sparse bases)
        Reduction: 250x

    Examples
    --------
    >>> B_r = sp.random(100, 50, density=0.1, format='csc')
    >>> B_c = sp.random(100, 50, density=0.1, format='csc')
    >>> Z_op = kron_linear_operator(B_r, B_c, 100, 100)
    >>> Z_op.shape
    (10000, 2500)
    >>> x = np.random.randn(2500)
    >>> y = Z_op @ x  # Lazy evaluation via matvec
    >>> y.shape
    (10000,)
    """
    p_r, p_c = B_r.shape[1], B_c.shape[1]
    m = n_r * n_c  # Output dimension
    n = p_r * p_c  # Input dimension

    # Define lazy matrix-vector operations
    def mv(x):
        return kron_matvec(B_r, B_c, x, (n_r, n_c))

    def rmv(y):
        return kron_rmatvec(B_r, B_c, y, (n_r, n_c))

    return LinearOperator((m, n), matvec=mv, rmatvec=rmv)
