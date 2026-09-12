"""Public spatial specifications and fitted-model heritability."""


def SAP(
    x_coord,
    y_coord,
    nseg=(10, 10),
    degree=3,
    penalty_order=2,
    nest_div=(1, 1),
    center=False,
    ANOVA=False,
):
    """Specify R's two-penalty SAP model (four penalties with ANOVA=True)."""
    from .model_basis import SpatialSpec

    return SpatialSpec(
        x_coord,
        y_coord,
        nseg,
        degree,
        penalty_order,
        nest_div,
        center,
        "SAP.ANOVA" if ANOVA else "SAP",
    )


def PSANOVA(
    x_coord,
    y_coord,
    nseg=(10, 10),
    degree=3,
    penalty_order=2,
    nest_div=(1, 1),
    center=False,
):
    """Specify R's five-component P-spline ANOVA model, with nested interaction bases."""
    from .model_basis import SpatialSpec

    return SpatialSpec(x_coord, y_coord, nseg, degree, penalty_order, nest_div, center)


def get_heritability(model):
    """Generalized heritability from a fitted random-genotype SpATS model.

    ED/count arithmetic was removed: the denominator must account for
    confounding with every fixed effect, which requires the fitted design.
    """
    if not hasattr(model, "get_heritability"):
        raise TypeError("Pass a fitted SpATS model with random genotypes")
    return model.get_heritability()
