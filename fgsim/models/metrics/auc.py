import torch
from sklearn.metrics import roc_auc_score


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
        sim_mean_disc = sim_crit.detach().cpu().reshape(-1)[:2000]
        gen_mean_disc = gen_crit.detach().cpu().reshape(-1)[:2000]

        y_pred = torch.sigmoid(torch.hstack([sim_mean_disc, gen_mean_disc])).numpy()
        y_true = torch.hstack(
            [torch.ones_like(sim_mean_disc), torch.zeros_like(gen_mean_disc)]
        ).numpy()
        return float(roc_auc_score(y_true, y_pred))


auc = Metric()
