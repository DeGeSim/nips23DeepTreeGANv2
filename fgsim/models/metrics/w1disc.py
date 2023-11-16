import torch
from scipy.stats import wasserstein_distance


class Metric:
    def __init__(
        self,
    ):
        pass

    def __call__(
        self, gen_crit: torch.Tensor, sim_crit: torch.Tensor, **kwargs
    ) -> float:
        assert sim_crit.shape == gen_crit.shape
        assert sim_crit.dim() == 2 and sim_crit.shape[1] == 1
        return wasserstein_distance(
            gen_crit.detach().cpu().numpy().reshape(-1)[:2000],
            sim_crit.detach().cpu().numpy().reshape(-1)[:2000],
        )


w1disc = Metric()
