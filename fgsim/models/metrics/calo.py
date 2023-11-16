import numpy as np
import torch
from caloutils.distances import calc_ecdf_dist, calc_hist_dist, calc_sw1_dist
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.plot import binborders_wo_outliers, var_to_bins


def run_dists(sim_batch, gen_batch, k, bins=None) -> dict[str, np.float32]:
    if k is not None:
        ftxnames = [e for e in sim_batch.hlv.keys() if e.startswith(k)]
        sim = sim_batch.hlv
    else:
        ftxnames = conf.loader.x_features
        sim = sim_batch.x

    if bins is None:
        bins = []
        if len(ftxnames) == 0:
            arr = sim.cpu().numpy()
            bins.append(torch.tensor(binborders_wo_outliers(arr, bins=300)).float())
        else:
            for fn in ftxnames:
                arr = sim[fn].cpu().numpy()
                bins.append(
                    torch.tensor(binborders_wo_outliers(arr, bins=300)).float()
                )
    if k is None:
        real = sim_batch.x
        fake = gen_batch.x
    else:
        real = torch.stack([sim_batch.hlv[ftxn] for ftxn in ftxnames], -1)
        fake = torch.stack([gen_batch.hlv[ftxn] for ftxn in ftxnames], -1)

    assert real.shape[-1] == fake.shape[-1]
    assert real.shape[0] >= fake.shape[0]
    # if necessairy, sample r
    if real.shape[0] > fake.shape[0]:
        idx = torch.randperm(real.shape[0])[: fake.shape[0]]
        real = real[idx]

    dists_d = {}
    for distname, fct in zip(
        ["cdf", "sw1", "histd"],
        [calc_ecdf_dist, calc_sw1_dist, calc_hist_dist],
    ):
        if distname == "histd":
            dists_d[distname] = fct(r=real, f=fake, bins=bins)
        else:
            dists_d[distname] = fct(r=real, f=fake)

    res_d = {}
    for dname, darr in dists_d.items():
        if len(ftxnames) == 0:
            res_d[f"{dname}"] = darr[0]
            continue
        for iftx, ftxname in enumerate(ftxnames):
            res_d[f"{ftxname}/{dname}"] = darr[iftx]

    return res_d


def marginal(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(
        sim_batch,
        gen_batch,
        k=None,
        bins=[torch.tensor(var_to_bins(i)).float() for i in range(4)],
    )


def marginalEw(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    # Eralphz
    res_d = {}

    real = sim_batch.x[..., 1:]
    fake = gen_batch.x[..., 1:]
    rw = sim_batch.x[..., 0]
    fw = gen_batch.x[..., 0]

    assert real.shape[-1] == fake.shape[-1]
    assert real.shape[0] >= fake.shape[0]
    # if necessairy, sample r
    if real.shape[0] > fake.shape[0]:
        idx = torch.randperm(real.shape[0])[: fake.shape[0]]
        real = real[idx]
        rw = rw[idx]

    kw = {
        "r": real,
        "f": fake,
        "rw": rw,
        "fw": fw,
    }

    cdfdist = calc_ecdf_dist(**kw)
    sw1dist = calc_sw1_dist(**kw)
    histdist = calc_hist_dist(
        **kw, bins=[torch.tensor(var_to_bins(i)).float() for i in range(1, 4)]
    )
    distnames = ["cdf", "sw1", "histd"]
    distarrays = [cdfdist, sw1dist, histdist]
    for iftx, k in enumerate(conf.loader.x_features[1:]):
        for distname, arr in zip(distnames, distarrays):
            res_d[f"{k}/{distname}"] = arr[iftx]

    return res_d


def sphereratio(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="sphereratio")


def cyratio(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="cyratio")


def response(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="response")


def fpc(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="fpc")


def showershape(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="showershape")


def nhits(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="nhits")
