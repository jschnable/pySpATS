"""Plots of fitted quantities; no reconstructed or synthetic spatial surfaces."""


def plot_spats(model, which="all", figsize=(12, 8), show=True, **kwargs):
    """Plot a field map or all four diagnostics; returns a Matplotlib figure."""
    import matplotlib.pyplot as plt

    if which == "all":
        return model.plot(show=show, figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    if which == "spatial":
        obj = ax.scatter(
            model.data[model.spec.x],
            model.data[model.spec.y],
            c=model.spatial_trend,
            marker="s",
        )
        ax.set(xlabel=model.spec.x, ylabel=model.spec.y, title="Fitted spatial trend")
        fig.colorbar(obj, ax=ax)
    elif which == "residuals":
        ax.scatter(model.fitted_values, model.residuals)
        ax.axhline(0, color="gray")
        ax.set(xlabel="Fitted", ylabel="Response residual")
    elif which == "fitted":
        ax.scatter(model.data[model.response], model.fitted_values)
        ax.set(xlabel="Observed", ylabel="Fitted")
    else:
        raise ValueError("which must be all, spatial, residuals, or fitted")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_variogram(variogram_obj, figsize=(8, 6), show=True, **kwargs):
    """Plot empirical semivariances and an optional fitted variogram curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(variogram_obj.distances, variogram_obj.gamma)
    if hasattr(variogram_obj, "fitted_gamma"):
        ax.plot(variogram_obj.distances, variogram_obj.fitted_gamma)
    ax.set(xlabel="Distance", ylabel="Semivariance")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
