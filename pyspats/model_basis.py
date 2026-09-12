"""Tensor P-splines translated from SpATS 1.0-20 (GPL).

The null-space rotation and penalty scaling follow MM.basis.R and
construct.2d.pspline.R. Evaluation retains the training knots and rotation.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import svd


def pair(value, name):
    a = np.asarray(value)
    if a.ndim == 0:
        a = np.repeat(a, 2)
    if (
        a.shape != (2,)
        or not np.all(np.isfinite(a))
        or np.any(a != np.round(a))
        or np.any(a < 1)
    ):
        raise ValueError(
            f"{name} must be a positive integer or a pair of positive integers"
        )
    return tuple(a.astype(int))


@dataclass(frozen=True)
class SpatialSpec:
    """Spatial coordinates and spline resolution; segment counts are not knot counts.

    ``nest_div`` reduces only the smooth-by-smooth interaction basis, exactly
    as in R. ``kind`` is PSANOVA (five variances), SAP (two overlapping
    penalties), or SAP.ANOVA (four overlapping penalties).
    """

    x: str
    y: str
    nseg: tuple = (10, 10)
    degree: tuple = (3, 3)
    penalty_order: tuple = (2, 2)
    nest_div: tuple = (1, 1)
    center: bool = False
    kind: str = "PSANOVA"

    def __post_init__(self):
        for key in ("nseg", "degree", "penalty_order", "nest_div"):
            object.__setattr__(self, key, pair(getattr(self, key), key))
        if self.x == self.y:
            raise ValueError("Spatial coordinates must be different columns")
        if self.kind not in ("PSANOVA", "SAP", "SAP.ANOVA"):
            raise ValueError("Unknown spatial model kind")
        if self.kind == "PSANOVA" and self.penalty_order != (2, 2):
            raise ValueError("PSANOVA requires second-order penalties, as in R")
        if any(n % d for n, d in zip(self.nseg, self.nest_div)):
            raise ValueError("nseg must be divisible by nest_div")
        if any(
            p >= n // div + d
            for p, d, n, div in zip(
                self.penalty_order, self.degree, self.nseg, self.nest_div
            )
        ):
            raise ValueError("penalty_order must not exhaust the nested basis")


def spatial_spec(value):
    if isinstance(value, SpatialSpec):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return SpatialSpec(*value)
    if isinstance(value, dict):
        v = dict(value)
        return SpatialSpec(
            v.pop("x_coord"), v.pop("y_coord"), kind=v.pop("type", "PSANOVA"), **v
        )
    raise TypeError(
        "spatial must be (x_column, y_column), SpatialSpec, PSANOVA(), or SAP()"
    )


class MarginalBasis:
    def __init__(self, x, nseg, degree, order):
        lo, hi = np.min(x), np.max(x)
        if hi <= lo:
            raise ValueError(
                "Each spatial coordinate must have at least two distinct values"
            )
        self.bounds = lo, hi
        step = (hi - lo) / nseg
        self.knots = lo + np.arange(-degree, nseg + degree + 1) * step
        self.degree = degree
        m = nseg + degree
        self.spline = BSpline(self.knots, np.eye(m), degree, extrapolate=False)
        b = self.spline(x)
        diff = np.diff(np.eye(m), n=order, axis=0)
        u, d, _ = svd(diff.T @ diff)
        self.uz, self.d = u[:, : m - order], d[: m - order]
        ux = u[:, m - order :]
        null = b @ ux
        centered = null - null.mean(axis=0)
        rotation, _, _ = svd(centered.T @ centered)
        self.ux = ux @ rotation[:, ::-1]

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if np.any((x < self.bounds[0]) | (x > self.bounds[1])):
            raise ValueError(
                "Prediction coordinates are outside the fitted field; extrapolation is disabled"
            )
        b = self.spline(x)
        return b @ self.ux, b @ self.uz


def tensor(a, b):
    """Row tensor product, with b's columns varying fastest (Rten2)."""
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


class SpatialBasis:
    def __init__(self, x, y, spec, observed):
        self.spec = spec
        self.marginals = [
            MarginalBasis(v, n, d, p)
            for v, n, d, p in zip((x, y), spec.nseg, spec.degree, spec.penalty_order)
        ]
        self.nested = [
            m if div == 1 else MarginalBasis(v, n // div, d, p)
            for m, v, n, d, p, div in zip(
                self.marginals,
                (x, y),
                spec.nseg,
                spec.degree,
                spec.penalty_order,
                spec.nest_div,
            )
        ]
        X, Z, blocks = self._raw(x, y)
        self.xmean = X[observed].mean(0) if spec.center else np.zeros(X.shape[1])
        self.zmean = Z[observed].mean(0) if spec.center else np.zeros(Z.shape[1])
        self.blocks = blocks
        dx, dy = (m.d for m in self.marginals)
        nx, ny = (m.d for m in self.nested)
        px, py = spec.penalty_order
        ds = [
            dx,
            dy,
            np.tile(dx, py - 1),
            np.repeat(dy, px - 1),
            np.tile(nx, len(ny)) + np.repeat(ny, len(nx)),
        ]
        if spec.kind == "SAP":
            sizes = [len(dx) * py, len(dy) * px, len(nx) * len(ny)]
            self.penalties = np.array(
                [
                    np.r_[np.tile(dx, py), np.zeros(sizes[1]), np.tile(nx, len(ny))],
                    np.r_[
                        np.zeros(sizes[0]), np.repeat(dy, px), np.repeat(ny, len(nx))
                    ],
                ]
            )
            self.names = ["spatial_x", "spatial_y"]
        else:
            self.names = [
                "x_smooth",
                "y_smooth",
                "x_smooth:y_linear",
                "x_linear:y_smooth",
                "x_smooth:y_smooth",
            ]
            self.penalties = np.zeros((5, Z.shape[1]))
            for i, (sl, dd) in enumerate(zip(blocks, ds)):
                self.penalties[i, sl] = dd
            if spec.kind == "SAP.ANOVA":
                self.penalties[2, blocks[-1]] = np.tile(nx, len(ny))
                self.penalties[3, blocks[-1]] = np.repeat(ny, len(nx))
                self.penalties = self.penalties[:4]
                self.names = ["x_smooth", "y_smooth", "spatial_x", "spatial_y"]

    def _raw(self, x, y):
        X1, Z1 = self.marginals[0].evaluate(x)
        X2, Z2 = self.marginals[1].evaluate(y)
        _, N1 = self.nested[0].evaluate(x)
        _, N2 = self.nested[1].evaluate(y)
        X = tensor(X2, X1)[:, 1:]
        if self.spec.kind == "SAP":
            parts = [tensor(X2, Z1), tensor(Z2, X1), tensor(N2, N1)]
        else:
            parts = [
                tensor(X2[:, :1], Z1),
                tensor(Z2, X1[:, :1]),
                tensor(X2[:, 1:], Z1),
                tensor(Z2, X1[:, 1:]),
                tensor(N2, N1),
            ]
        ends = np.cumsum([0] + [a.shape[1] for a in parts])
        return X, np.hstack(parts), [slice(a, b) for a, b in zip(ends[:-1], ends[1:])]

    def evaluate(self, x, y):
        X, Z, _ = self._raw(x, y)
        return X - self.xmean, Z - self.zmean
