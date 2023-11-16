from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from fgsim.monitoring.metrics_aggr import GradHistAggregator
from fgsim.plot.binborders import bounds_wo_outliers


def get_grad_dict(model):
    named_parameters = model.named_parameters()
    ave_grads = []
    weights = []
    # max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(
                n.replace(".parametrizations.weight.orig", ".snorm")
                .rstrip("_orig")
                .rstrip("_weight")
                .replace(".seq.", ".")
                .replace(".nn.", ".")
                .replace("_nn.", ".")
                .replace(".reduction.", ".red.")
                .rstrip(".linear")
            )
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
            weights.append(p.abs().mean().cpu().detach().numpy())
            # max_grads.append(p.grad.abs().max().cpu())
    return {
        "weights": {k: weights for k, weights in zip(layers, weights)},
        "grads": {k: weights for k, weights in zip(layers, ave_grads)},
    }


def fig_grads(grad_aggr: GradHistAggregator, partname: str) -> Figure:
    graddict: OrderedDict = grad_aggr.grad_history
    weightdict: OrderedDict = grad_aggr.weigth_history

    steps = np.array(grad_aggr.steps)
    layers = list(graddict.keys())
    ave_grads = np.array([list(e) for e in graddict.values()])
    ave_weigths = np.array([list(e) for e in weightdict.values()])

    layers_formated = format_layer_labels(layers)

    max_bins = 25

    ave_grads = pad_to_multiple(ave_grads, max_bins)
    ave_weigths = pad_to_multiple(ave_weigths, max_bins)
    steps = pad_to_multiple(steps.reshape(1, -1), max_bins).reshape(-1)
    nparts, ntimesteps = ave_grads.shape

    plt.clf()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2,
        2,
        sharey="row",
        figsize=(20, 22),
        height_ratios=(4, 1),
        width_ratios=(1, 1),
    )

    # vmin, vmax = bounds_wo_outliers(ave_weigths.reshape(-1))
    vmin, vmax = 0.02, 0.2
    im1 = ax1.imshow(
        ave_weigths, cmap=plt.cm.coolwarm, norm=LogNorm()  # (vmin=vmin, vmax=vmax)
    )
    # annotate(ax1, ave_weigths, vmin, vmax)
    ax1.set_yticks(
        ticks=np.arange(nparts), labels=layers_formated, family="monospace"
    )
    ax1.set_xticks(ticks=np.arange(ntimesteps), labels=steps, rotation=45)
    ax1.set_ylabel("Layers")
    ax1.set_xlabel("Step")
    ax1.set_title("Weigths")
    cax = ax1.inset_axes([1.04, 0.2, 0.05, 0.6])
    fig.colorbar(im1, cax)

    vmin, vmax = bounds_wo_outliers(ave_weigths.reshape(-1))
    vmin, vmax = 1e-3, 1e0
    im2 = ax2.imshow(
        ave_grads, cmap=plt.cm.coolwarm, norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    annotate(ax2, ave_grads, vmin, vmax)
    ax2.set_xticks(ticks=np.arange(ntimesteps), labels=steps, rotation=45)
    ax2.set_xlabel("Step")
    ax2.set_title("Grads")
    cax = ax2.inset_axes([1.04, 0.2, 0.05, 0.6])
    fig.colorbar(im2, cax)

    ax3.hist(x=ave_weigths.reshape(-1), bins=max_bins)
    ax3.set_ylabel("Frequency")
    ax4.hist(x=ave_grads.reshape(-1), bins=max_bins)

    fig.suptitle(f"{partname} Model Means")
    fig.tight_layout()

    return fig


def annotate(ax, arr, vmin, vmax):
    nparts, ntimesteps = arr.shape
    for i in range(nparts):
        for j in range(ntimesteps):
            if arr[i][j] < vmin:
                ax.text(j, i, "V", ha="center", va="center")
            if arr[i][j] > vmax:
                ax.text(j, i, "O", ha="center", va="center")


def format_layer_labels(layers):
    layers_split = [e.split(".") for e in layers]
    max_levels = max([len(e) for e in layers_split])
    # expand to  same size
    layers_split = [e + [""] * (max_levels - len(e)) for e in layers_split]
    max_chars_per_lvl = [
        max([len(e[ilevel]) for e in layers_split]) for ilevel in range(max_levels)
    ]
    # padd the characters
    layers_split = [
        [
            substr + " " * (max_chars_per_lvl[ilvl] - len(substr))
            for ilvl, substr in enumerate(e)
        ]
        for e in layers_split
    ]
    layers_formated = ["/".join(e) for e in layers_split]
    return layers_formated


def pad_to_multiple(arr: np.ndarray, max_bins: int):
    _, nentries = arr.shape
    if nentries <= max_bins:
        return arr
    choice = np.arange(max_bins) * (nentries // max_bins)
    return arr[:, choice]
