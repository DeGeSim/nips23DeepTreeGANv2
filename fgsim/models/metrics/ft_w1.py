from typing import Dict

from scipy.stats import wasserstein_distance
from torch_geometric.data import Batch

from fgsim.config import conf


class Metric:
    def __init__(
        self,
    ):
        pass

    def lossf(self, a, b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()

        return wasserstein_distance(a, b)

    def __call__(
        self, gen_batch: Batch, sim_batch: Batch, *args, **kwargs
    ) -> Dict[str, float]:
        out_dict: Dict[str, float] = {}
        for ivar, var in enumerate(conf.loader.x_features):
            out_dict[f"w1{var}"] = self.lossf(
                sim_batch.x[:1000, ..., ivar], gen_batch.x[:1000, ..., ivar]
            )

        return out_dict


ft_w1 = Metric()
