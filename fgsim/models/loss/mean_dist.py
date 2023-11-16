"""
We create a loss class. This class has a method `__call__`
that takes a `Holder` and a `Batch` as arguments.
It then computes the mean of the points in the batch
and the mean of the points in the generated batch.
It then uses the `torch.nn.MSELoss` to compute the loss
between these two means. It then backpropagates the loss
and returns the loss.
"""
import torch

# from fgsim.config import conf
from torch_geometric.data import Batch

from fgsim.ml.holder import Holder

# from torch_scatter import scatter_mean


class LossGen:
    def __init__(self):
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, batch: Batch):
        gen_batch = holder.gen_points_w_grad
        # Aggregate the means of over the points in a event
        # sim_means = scatter_mean(batch.x, batch.batch, dim=0).sort(dim=0).values
        # gen_means = (
        #     scatter_mean(gen_batch.x, gen_batch.batch, dim=0).sort(dim=0).values
        # )
        # assert (
        #     list(sim_means.shape)
        #     == list(gen_means.shape)
        #     == [conf.loader.batch_size, conf.loader.n_features]
        # )
        # loss = self.lossf(gen_means, sim_means)
        mean_sim, mean_gen = (
            torch.mean(sample.x, dim=0) for sample in (batch, gen_batch)
        )
        cov_sim, cov_gen = (torch.cov(sample.x.T) for sample in (batch, gen_batch))
        loss = self.lossf(mean_sim, mean_gen) + self.lossf(cov_sim, cov_gen)

        return loss
