import numpy as np
import torch
from jetnet.evaluation import gen_metrics
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.datasets.jetnet.utils import to_stacked_mask

num_batches = 1 if conf.command == "train" else 5
num_eval_samples = 50_000 if conf.command == "train" else 25_000
jnkw = {"num_batches": num_batches, "num_eval_samples": num_eval_samples}


def w1m(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jf1 = get_jf_mask(gen_batch)[0]
    jf2 = get_jf_mask(sim_batch)[0]
    score = gen_metrics.w1m(jets1=jf1, jets2=jf2, **jnkw)
    return bound_res(score)


def w1p(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jf1, mask1 = get_jf_mask(gen_batch)
    jf2, mask2 = get_jf_mask(sim_batch)
    res = gen_metrics.w1p(
        jets1=jf1,
        jets2=jf2,
        mask1=mask1,
        mask2=mask2,
        exclude_zeros=True,
        **(jnkw | {"num_batches": 5}),
    )
    score, error = std_weighted_mean(res)
    return bound_res([score, error])


def w1efp(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jf1, _ = get_jf_mask(gen_batch)
    jf2, _ = get_jf_mask(sim_batch)
    res = gen_metrics.w1efp(jets1=jf1, jets2=jf2, efp_jobs=10, **jnkw)
    score, error = std_weighted_mean(res)
    return bound_res([score, error], 1e5)


def fpnd(gen_batch: Batch, **kwargs) -> tuple[float, float]:
    jets = to_stacked_mask(gen_batch)[:, :, :3]
    if conf.loader.n_points != 30:
        pts = jets[..., conf.loader.x_ftx_energy_pos]
        topidxs = pts.topk(k=30, dim=1, largest=True).indices

        highptjets = torch.stack([jet[idx] for jet, idx in zip(jets, topidxs)])

    else:
        highptjets = jets

    try:
        score = gen_metrics.fpnd(
            jets=highptjets,
            jet_type=conf.loader.jettype,
            use_tqdm=False,
        )
        return (min(float(score), 1e5), np.nan)
    except ValueError:
        return (1e5, np.nan)


def cov_mmd(gen_batch: Batch, sim_batch: Batch, **kwargs):
    try:
        real_jets = to_stacked_mask(sim_batch)
        gen_jets = to_stacked_mask(gen_batch)
        if gen_jets.isnan().any():
            raise ValueError
        score_cov, score_mmd = gen_metrics.cov_mmd(
            real_jets=real_jets,
            gen_jets=gen_jets,
            use_tqdm=False,
            num_batches=num_batches,
        )
        return bound_res([score_cov, score_mmd], 1e5)
    except ValueError:
        return (1e5, 1e5)


def kpd(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    res = gen_metrics.kpd(
        real_features=sim_batch.efps[:num_eval_samples],
        gen_features=gen_batch.efps[:num_eval_samples],
        num_threads=10,
        num_batches=num_batches,
    )
    return bound_res(res, 1e3)


def fpd(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    res = gen_metrics.fpd(
        real_features=sim_batch.efps[:num_eval_samples],
        gen_features=gen_batch.efps[:num_eval_samples],
        num_batches=num_batches,
    )
    return bound_res(res, 1e3)


def get_jf_mask(batch):
    jets = to_stacked_mask(batch).detach().cpu().numpy()
    jf = jets[:num_eval_samples, ..., :3]
    mask = jets[:num_eval_samples, ..., -1]
    return jf, mask


def bound_res(res, scale=1e3):
    assert len(res) == 2
    return tuple(min(float(e), 1) * scale for e in res)


def std_weighted_mean(res):
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    x, stds = res[0], res[1]
    ws = stds**-2
    score = (x * ws).sum() / ws.sum()
    error = np.sqrt((1 / ws.sum()))
    return score, error
