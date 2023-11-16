from typing import Dict

from scipy.stats import wasserstein_distance
from torch_geometric.data import Batch

from fgsim.ml.holder import Holder


class Metric:
    def __init__(
        self,
        foreach_hlv: bool,
    ):
        self.foreach_hlv: bool = foreach_hlv

    def lossf(self, a, b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()

        return wasserstein_distance(a, b)

    def __call__(self, holder: Holder, batch: Batch) -> Dict[str, float]:
        out_dict: Dict[str, float] = {}
        for var in holder.gen_points.hlvs:
            out_dict[var] = self.lossf(batch.hlvs[var], holder.gen_points.hlvs[var])

        return out_dict
