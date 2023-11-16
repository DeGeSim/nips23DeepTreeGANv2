from statistics import harmonic_mean

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from .ffn import FFN


class DynHLVsLayer(nn.Module):
    def __init__(
        self, n_features: int, n_global: int, n_cond: int, batch_size: int
    ):
        super().__init__()
        self.n_features = n_features
        self.n_global = n_global
        self.n_cond = n_cond
        self.batch_size = batch_size
        if self.n_global == 0:
            return

        n_intermediate = int(harmonic_mean((self.n_features, self.n_global)))
        self.pre_nn: nn.Module = FFN(self.n_features, n_intermediate)
        self.post_nn: nn.Module = FFN(n_intermediate + n_cond, self.n_global)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        if self.n_global == 0:
            return torch.empty(
                self.batch_size,
                self.n_global,
                dtype=torch.float,
                device=x.device,
            )

        ftx_mtx = self.pre_nn(x)
        gsum = global_add_pool(ftx_mtx, batch)
        global_ftx = self.post_nn(torch.hstack((gsum, cond)))
        return global_ftx.reshape(self.batch_size, self.n_global)
