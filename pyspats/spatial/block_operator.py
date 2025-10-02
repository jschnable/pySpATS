"""
Block linear operator utilities for mixed sparse/LinearOperator blocks.

This module provides a unified interface for working with blocks that may be
either sparse matrices or LinearOperators, crucial for the Kronecker-structured
interaction smooth.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from typing import Union, List
from ..ed_selected_inverse import BlockInfo


class BlockLinearOperator:
    """
    Wrapper for a single block that can be either sparse or LinearOperator.

    Provides a common API (matvec, rmatvec, shape) for both types.

    Parameters
    ----------
    block : scipy.sparse matrix or LinearOperator
        The block to wrap
    name : str
        Name of this block (e.g., "row_smooth", "interaction_smooth")

    Attributes
    ----------
    block : sparse matrix or LinearOperator
        The underlying block
    name : str
        Block identifier
    shape : tuple of int
        Shape (n_rows, n_cols) of the block
    is_sparse : bool
        True if block is a sparse matrix, False if LinearOperator
    """

    def __init__(self, block: Union[sp.spmatrix, LinearOperator], name: str):
        self.block = block
        self.name = name
        self.shape = block.shape
        self.is_sparse = sp.issparse(block)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Compute block @ x.

        Parameters
        ----------
        x : np.ndarray
            Input vector (n_cols,)

        Returns
        -------
        np.ndarray
            Output vector (n_rows,)
        """
        if self.is_sparse:
            result = self.block @ x
            # Ensure 1D array
            if sp.issparse(result):
                result = result.toarray().ravel()
            return np.asarray(result).ravel()
        else:
            # LinearOperator
            result = self.block @ x
            return np.asarray(result).ravel()

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Compute block^T @ y.

        Parameters
        ----------
        y : np.ndarray
            Input vector (n_rows,)

        Returns
        -------
        np.ndarray
            Output vector (n_cols,)
        """
        if self.is_sparse:
            result = self.block.T @ y
            # Ensure 1D array
            if sp.issparse(result):
                result = result.toarray().ravel()
            return np.asarray(result).ravel()
        else:
            # LinearOperator has rmatvec method
            result = self.block.rmatvec(y)
            return np.asarray(result).ravel()

    @property
    def n_rows(self) -> int:
        """Number of rows in the block."""
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        """Number of columns in the block."""
        return self.shape[1]

    def __repr__(self) -> str:
        block_type = "sparse" if self.is_sparse else "LinearOperator"
        return f"BlockLinearOperator('{self.name}', {block_type}, shape={self.shape})"


class ConcatenatedBlockOperator:
    """
    Container for multiple block operators concatenated column-wise.

    This represents Z = [Z_1, Z_2, ..., Z_k] where each Z_i can be either
    a sparse matrix or a LinearOperator. Provides efficient matvec and rmatvec
    without materializing the full concatenated matrix.

    Parameters
    ----------
    blocks : list of BlockLinearOperator
        Individual blocks in order
    block_info : list of BlockInfo
        Metadata for each block (start, stop indices in coefficient space)

    Attributes
    ----------
    blocks : list of BlockLinearOperator
        The wrapped blocks
    block_info : list of BlockInfo
        Block metadata with contiguous indices
    n_rows : int
        Number of rows (common to all blocks)
    n_cols : int
        Total number of columns (sum of block sizes)
    shape : tuple of int
        Shape (n_rows, n_cols)
    """

    def __init__(
        self,
        blocks: List[BlockLinearOperator],
        block_info: List[BlockInfo]
    ):
        if len(blocks) != len(block_info):
            raise ValueError("Number of blocks must match block_info length")

        self.blocks = blocks
        self.block_info = block_info

        # Verify all blocks have same number of rows
        if blocks:
            self.n_rows = blocks[0].n_rows
            if not all(b.n_rows == self.n_rows for b in blocks):
                raise ValueError("All blocks must have the same number of rows")
        else:
            self.n_rows = 0

        # Total columns is sum of block sizes
        self.n_cols = sum(b.n_cols for b in blocks)

        self.shape = (self.n_rows, self.n_cols)

    def matvec(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Z @ u = sum_k Z_k @ u_k.

        Parameters
        ----------
        u : np.ndarray
            Concatenated coefficient vector (n_cols,)

        Returns
        -------
        np.ndarray
            Result vector (n_rows,)
        """
        if len(u) != self.n_cols:
            raise ValueError(f"Input size {len(u)} != n_cols {self.n_cols}")

        # Initialize result
        result = np.zeros(self.n_rows)

        # Accumulate contribution from each block
        for block, info in zip(self.blocks, self.block_info):
            u_k = u[info.start:info.stop]
            result += block.matvec(u_k)

        return result

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Compute Z^T @ y = [Z_1^T @ y; Z_2^T @ y; ...; Z_k^T @ y].

        Parameters
        ----------
        y : np.ndarray
            Input vector (n_rows,)

        Returns
        -------
        np.ndarray
            Concatenated result vector (n_cols,)
        """
        if len(y) != self.n_rows:
            raise ValueError(f"Input size {len(y)} != n_rows {self.n_rows}")

        # Initialize result
        result = np.zeros(self.n_cols)

        # Accumulate each block's contribution
        for block, info in zip(self.blocks, self.block_info):
            result[info.start:info.stop] = block.rmatvec(y)

        return result

    def __matmul__(self, other):
        """Support @ operator for matrix-vector multiplication."""
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                return self.matvec(other)
            else:
                raise ValueError("Only 1D arrays supported for @ operator")
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """Support @ operator for vector-matrix multiplication (v @ Z)."""
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                return self.rmatvec(other)
            else:
                raise ValueError("Only 1D arrays supported for @ operator")
        else:
            return NotImplemented

    def as_linear_operator(self) -> LinearOperator:
        """
        Create a LinearOperator wrapper for this concatenated block operator.

        Returns
        -------
        LinearOperator
            Scipy LinearOperator with matvec and rmatvec methods
        """
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec
        )

    def get_block(self, name: str) -> BlockLinearOperator:
        """
        Get a specific block by name.

        Parameters
        ----------
        name : str
            Block name

        Returns
        -------
        BlockLinearOperator
            The requested block

        Raises
        ------
        KeyError
            If block name not found
        """
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"Block '{name}' not found")

    def __repr__(self) -> str:
        block_names = [b.name for b in self.blocks]
        return f"ConcatenatedBlockOperator({block_names}, shape={self.shape})"
