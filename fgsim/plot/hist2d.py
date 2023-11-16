from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fgsim.plot.binborders import binborders_wo_outliers, chip_to_binborders

np.set_printoptions(formatter={"float_kind": "{:.3g}".format})


def hist2d(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
    v1bins: Optional[np.ndarray] = None,
    v2bins: Optional[np.ndarray] = None,
    simw: Optional[np.ndarray] = None,
    genw: Optional[np.ndarray] = None,
    step: Optional[int] = None,
) -> Figure:
    plt.close("all")
    plt.cla()
    plt.clf()
    fig: Figure
    sim_axes: Axes
    gen_axes: Axes
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(4.5, 3.7))
    (sim_axes, gen_axes) = axes
    if v1bins is None:
        xedges = binborders_wo_outliers(sim[:, 0])
        yedges = binborders_wo_outliers(sim[:, 1])
    else:
        xedges = v1bins
        yedges = v2bins

    # flip x and y axis if the x val has more bins
    if len(xedges) > len(yedges) + 5:
        xedges, yedges = yedges, xedges
        sim = sim[:, (1, 0)]
        gen = gen[:, (1, 0)]
        v1bins, v2bins = v2bins, v1bins
        v1name, v2name = v2name, v1name

    norm = None
    for iax, (ax, arr, weights) in enumerate(
        zip([sim_axes, gen_axes], [sim, gen], [simw, genw])
    ):
        x = arr[:, 0]
        y = arr[:, 1]
        if len(x) > 500_000:
            sel = np.random.choice(x.shape[0], 500_000)
            x = x[sel]
            y = y[sel]
            if weights is not None:
                weights = weights[sel]
        mesh, norm = _2dhist_with_autonorm(
            ax=ax,
            x=chip_to_binborders(x, xedges),
            y=chip_to_binborders(y, yedges),
            bins=[xedges, yedges],
            cmap=plt.cm.hot,
            norm=norm,
            linewidths=2,
            weights=weights,
        )
        for spline in ax.spines.values():
            spline.set_linewidth(1.2)
            spline.set_color("black")

        if iax == 0:
            ax.set_ylabel(v2name)
        ax.set_xlabel(v1name)

    sim_axes.set_title("Simulation")
    gen_axes.set_title("Model")

    if step is not None:
        title += f"\nStep {step}"
    fig.suptitle(title)

    cax = make_axes_locatable(plt.gca()).append_axes("right", "5%", pad="3%")

    fig.colorbar(mesh, cax)
    return fig


def edge_to_labels(edges):
    if len(edges) < 10:
        ticks = np.arange(len(edges))
    else:
        ticks = np.linspace(0, len(edges) - 1, 10, dtype="int")
    labels = [str(int(e)) for e in edges[ticks]]
    return dict(ticks=ticks, labels=labels)


def _2dhist_with_autonorm(
    ax,
    x,
    y,
    bins,
    norm=None,
    range=None,
    density=False,
    weights=None,
    **kwargs,
):
    xedges, yedges = bins
    h, _, _ = np.histogram2d(
        x, y, bins=bins, range=range, density=density, weights=weights
    )

    if norm is None:
        if (h > (h.max() / 10)).mean() < 0.05:
            # if h.max() / max(np.median(h), 1) > 6:
            norm = LogNorm(1, h.max())
        else:
            norm = Normalize(0, h.max())

    pc = ax.pcolormesh(xedges, yedges, h.T, norm=norm, **kwargs)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    return pc, norm
