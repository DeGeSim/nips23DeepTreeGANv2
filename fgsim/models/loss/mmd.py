from typing import List

import torch
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, kernel, bandwidth: List[float]) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth

    def __call__(self, holder: Holder, batch: Batch, *args, **kwargs):
        shape = (
            conf.loader.batch_size,
            conf.loader.n_points * conf.loader.n_features,
        )
        sim_sample = batch.x.reshape(*shape)
        gen_sample = holder.gen_points_w_grad.x.reshape(*shape)
        assert sim_sample.shape == gen_sample.shape

        loss = MMD(
            sim_sample,
            gen_sample,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        )
        return loss


def MMD(x, y, bandwidth, kernel):
    # https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
    """Emprical maximum mean discrepancy. The lower the result
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    dxx = torch.pow(torch.cdist(x, x), 2)
    dyy = torch.pow(torch.cdist(y, y), 2)
    dxy = torch.pow(torch.cdist(x, y), 2)

    XX, YY, XY = (
        torch.zeros(dxx.shape).to(device),
        torch.zeros(dxx.shape).to(device),
        torch.zeros(dxx.shape).to(device),
    )

    for a in bandwidth:
        if kernel == "multiscale":
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1
        elif kernel == "rbf":
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
        else:
            raise Exception

    return torch.mean(XX + YY - 2.0 * XY)
