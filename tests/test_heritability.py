"""Heritability is a fitted random-genotype estimability calculation."""

import pytest
from pyspats import get_heritability
from .test_r_parity import fit_case


def test_requires_random_genotypes():
    with pytest.raises(ValueError, match="random"):
        get_heritability(fit_case("PSANOVA_fixed"))


def test_matches_fitted_estimability():
    m = fit_case("PSANOVA_random")
    assert get_heritability(m) == pytest.approx(m.effective_dim["geno"] / 23)


def test_population_components():
    m = fit_case("populations")
    h = m.heritability
    assert set(h) == {"geno:P1", "geno:P2"}
    assert all(0 <= v <= 1 for v in h.values())


def test_arithmetic_shortcut_rejected():
    with pytest.raises(TypeError, match="fitted"):
        get_heritability(10)
