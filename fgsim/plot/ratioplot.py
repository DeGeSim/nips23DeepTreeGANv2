from typing import Optional

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from .binborders import binborders_wo_outliers, bincenters


def ratioplot(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    bins: Optional[np.ndarray] = None,
    simw: Optional[np.ndarray] = None,
    genw: Optional[np.ndarray] = None,
) -> Figure:
    if bins is None:
        bins = binborders_wo_outliers(sim)
    n_bins = len(bins) - 1

    sim_hist, _ = np.histogram(sim, bins=bins, weights=simw)
    gen_hist, _ = np.histogram(gen, bins=bins, weights=genw)
    sim_error = np.sqrt(sim_hist)
    gen_error = np.sqrt(gen_hist)

    scale_factor = 1  # int(np.floor(np.log10(max(sim_hist.max(), gen_hist.max()))))

    sim_hist = sim_hist * (10**-scale_factor)
    gen_hist = gen_hist * (10**-scale_factor)
    sim_error = sim_error * (10**-scale_factor)
    gen_error = gen_error * (10**-scale_factor)

    plt.close("all")
    ax: Axes
    axrat: Axes
    fig, (ax, axrat) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [2, 0.7]}, sharex="col"
    )

    mplhep.histplot(
        [sim_hist, gen_hist],
        bins=bins,
        label=["Simulation", "Model"],
        yerr=[sim_error, gen_error],
        ax=ax,
    )
    # overflow bins
    delta = (bins[1] - bins[0]) / 2
    simcolor = ax.containers[1][0]._color
    gencolor = ax.containers[2][0]._color
    kwstyle = dict(linestyle=(0, (0.5, 0.3)), lw=3)
    for arr, color, factor in zip([gen, sim], [gencolor, simcolor], [1, 2]):
        ax.vlines(
            x=bins[0] - factor * delta,
            ymin=0,
            ymax=(arr < bins[0]).sum() * (10**-scale_factor),
            color=color,
            **kwstyle,
        )
        ax.vlines(
            x=bins[-1] + factor * delta,
            ymin=0,
            ymax=(arr > bins[-1]).sum() * (10**-scale_factor),
            color=color,
            **kwstyle,
        )

    if (sim_hist > (sim_hist.max() / 10)).mean() < 0.1:
        ax.set_yscale("log")
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(13)
    ax.set_ylabel("Counts/Bin", fontsize=17)

    ax.legend(fontsize=16, loc="best")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=13)
    axrat.tick_params(axis="both", which="major", labelsize=15)
    axrat.tick_params(axis="both", which="minor", labelsize=13)

    # ratioplot
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = gen_hist / sim_hist
        frac_error_y = np.abs(frac) * np.sqrt(
            (sim_error / sim_hist) ** 2 + (gen_error / gen_hist) ** 2
        )
        frac_error_x = np.array([(bins[1] - bins[0]) / 2.0] * n_bins)
        frac_mask = (frac != 0) & np.invert(np.isnan(frac_error_y))
    axrat.axhline(1, color="grey")
    axrat.set_ylabel(r"$\frac{\text{Model}}{\text{Simulation}}$", fontsize=14)

    axrat.errorbar(
        x=bincenters(bins)[frac_mask],
        y=frac[frac_mask],
        yerr=frac_error_y[frac_mask],
        xerr=frac_error_x[frac_mask],
        barsabove=True,
        linestyle="",
        marker=None,
        ecolor="black",
        markersize=2,
    )

    axrat.set_ylim(0.48, 1.52)
    axrat.xaxis.tick_top()

    for iax in [ax, axrat]:
        for spline in iax.spines.values():
            spline.set_linewidth(1)
            spline.set_color("black")
    if simw is not None:
        title += " weighted"
    fig.suptitle(title, fontsize=26)
    plt.tight_layout()
    return fig
