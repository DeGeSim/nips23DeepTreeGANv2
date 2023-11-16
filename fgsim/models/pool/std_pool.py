from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import pool


def global_mad_pool(x: Tensor, batchidx: Optional[Tensor] = None) -> tuple[Tensor]:
    if x.dim() not in (2, 3):
        raise Exception
    if x.dim() == 2:
        if batchidx is None:
            raise Exception
        counts = batchidx.unique(sorted=True, return_counts=True)[1].reshape(-1, 1)
        means = pool.global_mean_pool(x, batchidx)
        deltas = (means[batchidx] - x).abs()
        widths = pool.global_mean_pool(deltas, batchidx)
    else:
        means = x.mean(1)
        counts = torch.ones_like(means[..., [0]]) * len(means)
        widths = (x - means.unsqueeze(1).repeat(1, x.shape[1], 1)).abs().mean(1)

    return counts, means, widths


def global_mad_pool2(x: Tensor, batchidx: Optional[Tensor] = None) -> tuple[Tensor]:
    if x.dim() not in (2, 3):
        raise Exception
    if x.dim() == 2:
        if batchidx is None:
            raise Exception

        sums = pool.global_add_pool(x, batchidx)
        maxs = pool.global_max_pool(x, batchidx)
        counts = batchidx.unique(sorted=True, return_counts=True)[1].reshape(-1, 1)

        # MAE
        means = sums / counts
        deltas = (means[batchidx] - x).abs()
        widths = pool.global_mean_pool(deltas, batchidx)
    else:
        sums = x.sum(1)
        maxs = x.max(1).values
        counts = torch.ones_like(sums[..., [0]]) * len(sums)

        # MAE
        means = sums / counts
        widths = (x - means.unsqueeze(1).repeat(1, x.shape[1], 1)).abs().mean(1)
    return counts, sums, widths, maxs


def global_var_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    means = pool.global_mean_pool(x, batchidx)
    deltas = torch.pow(means[batchidx] - x, 2)
    return pool.global_mean_pool(deltas, batchidx)


def global_std_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    return torch.sqrt(global_var_pool(x, batchidx))


# from torch_scatter import scatter_sum
# import torch

# x = torch.normal(0, 1, (200,), requires_grad=False)
# width = torch.tensor(2.0, requires_grad=True)
# shift = torch.tensor(4.0, requires_grad=True)

# xprime = x * width + shift
# aggr = scatter_std(
#     xprime.reshape(-1, 1),
#     torch.zeros(200, dtype=torch.long),
#     dim=-2,
# ).sum()
# aggr.backward()
# print(width.grad, shift.grad)
