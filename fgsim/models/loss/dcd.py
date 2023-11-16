from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.models.metrics.dcd import dcd
from fgsim.utils.jetnetutils import to_stacked_mask


class LossGen:
    def __init__(
        self, alpha, lpnorm: float = 2.0, batch_wise: bool = False, pow: float = 1
    ) -> None:
        self.alpha: float = alpha
        self.lpnorm: float = lpnorm
        self.batch_wise: bool = batch_wise
        self.pow: float = pow

    def __call__(
        self,
        sim_batch: Batch,
        gen_batch: Batch,
        **kwargs,
    ):
        assert gen_batch.x.requires_grad
        n_features = sim_batch.x.shape[1]
        batch_size = int(sim_batch.batch[-1] + 1)
        shape = (
            (1, -1, n_features) if self.batch_wise else (batch_size, -1, n_features)
        )
        sim = to_stacked_mask(sim_batch)[..., : conf.loader.n_features]
        loss = dcd(
            gen_batch.x.reshape(*shape),
            sim.reshape(*shape),
            alpha=self.alpha,
            lpnorm=self.lpnorm,
            pow=self.pow,
        ).mean()
        # if holder.state.epoch >= 150:
        #     loss *= 0
        if loss < 0:
            raise Exception
        return loss
