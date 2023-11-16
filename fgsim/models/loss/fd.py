import torch
from torch_geometric.data import Batch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        sim_batch: Batch,
        gen_batch: Batch,
        **kwargs,
    ):
        assert gen_batch.x.requires_grad
        n_features = sim_batch.x.shape[1]
        batch_size = int(sim_batch.batch[-1] + 1)
        shape = (batch_size, -1, n_features)
        loss = torch.cdist(
            gen_batch.x.reshape(*shape),
            sim_batch.x.reshape(*shape),
            p=2,
        ).mean()
        # if holder.state.epoch >= 150:
        #     loss *= 0
        if loss < 0:
            raise Exception
        return loss
