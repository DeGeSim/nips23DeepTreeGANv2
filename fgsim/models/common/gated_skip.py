import torch
from torch import Tensor, nn


class GatedSkip(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.mat = nn.Parameter(
            torch.normal(0, 1, size=(2 * n_features, n_features))
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        res = torch.mm(torch.hstack([x, y]), self.mat)
        assert res.shape == x.shape == y.shape
        return res
