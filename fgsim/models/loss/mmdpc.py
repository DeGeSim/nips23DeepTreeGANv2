from typing import List

import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.utils.jetnetutils import to_stacked_mask

from .mmd import MMD


class LossGen:
    def __init__(
        self, kernel, bandwidth: List[float], batch_wise: bool = False
    ) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.batch_wise = batch_wise

    def __call__(self, sim_batch: Batch, gen_batch: Batch, **kwargs):
        n_features = sim_batch.x.shape[1]
        batch_size = int(sim_batch.batch[-1] + 1)
        shape = (
            (1, -1, n_features) if self.batch_wise else (batch_size, -1, n_features)
        )
        sim_sample = to_stacked_mask(sim_batch)[..., : conf.loader.n_features]
        gen_sample = gen_batch.x.reshape(*shape)

        losses: List[torch.Tensor] = []
        for ifeature in range(conf.loader.n_features):
            losses.append(
                MMD(
                    sort_by_feature(sim_sample, ifeature),
                    sort_by_feature(gen_sample, ifeature),
                    bandwidth=self.bandwidth,
                    kernel=self.kernel,
                )
            )
        loss = sum(losses)
        if loss < 0:
            raise Exception
        return loss


def sort_by_feature(batch: torch.Tensor, ifeature: int) -> torch.Tensor:
    assert 0 <= ifeature <= batch.shape[-1]
    sorted_ftx_idxs = torch.argsort(batch[..., ifeature]).reshape(-1)
    batch_idxs = (
        torch.arange(batch.shape[0]).repeat_interleave(batch.shape[1]).reshape(-1)
    )
    batch_sorted = batch[batch_idxs, sorted_ftx_idxs, :].reshape(*batch.shape)
    return batch_sorted
