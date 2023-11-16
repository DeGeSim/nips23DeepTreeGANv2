import torch
from torch import nn


class FtxScaleLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.pars = nn.Parameter(torch.ones(n_features), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return x @ torch.diag(self.pars)
