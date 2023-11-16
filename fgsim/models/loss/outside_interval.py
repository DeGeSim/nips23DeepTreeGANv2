import torch

from fgsim.config import conf
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def __call__(self, holder: Holder, *args, **kwargs):
        shape = (
            conf.loader.batch_size,
            conf.loader.n_points * conf.loader.n_features,
        )
        gen_sample = holder.gen_points_w_grad.x.reshape(*shape)
        higher = gen_sample[gen_sample > self.high]
        d_higher = torch.abs(higher - self.high)
        lower = gen_sample[gen_sample < self.low]
        d_lower = torch.abs(lower - self.low)
        dists = torch.hstack([d_higher, d_lower])

        loss: torch.Tensor = torch.sum(torch.pow(dists, 2) + dists)
        return loss
