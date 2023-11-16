import torch
import torch_scatter
from torch_geometric.data import Batch

from fgsim.config import conf


# Make sure the unchosen particles are close to 0
class LossGen:
    def __init__(self) -> None:
        pass

    def get_unselected_batchidx(self, gen_batch):
        batch_size = gen_batch.batch[-1] + 1
        device = gen_batch.x.device
        max_points = (gen_batch.x.shape[0] + gen_batch.xnot.shape[0]) // batch_size

        assert (gen_batch.x.shape[0] + gen_batch.xnot.shape[0]) % batch_size == 0
        # get the batch information for the unselected elements
        ptr_delta = (
            torch.arange(batch_size + 1, device=device) * max_points - gen_batch.ptr
        )

        # The shift in the ptr add up, so we subtract a tensor shifted by 1
        removed_points_per_batch = (
            ptr_delta
            - torch.hstack([torch.tensor(0, device=device), ptr_delta])[:-1]
        )
        # repeat a batch vector for the number of missing points
        # shift by 1
        batch_unsel = (
            torch.arange(batch_size + 1, device=device).repeat_interleave(
                removed_points_per_batch
            )
            - 1
        )
        if len(batch_unsel):
            assert (
                max_points - (gen_batch.batch == batch_unsel[0]).sum()
                == (batch_unsel == batch_unsel[0]).sum()
            )
        assert len(gen_batch.xnot) == len(batch_unsel)
        return batch_unsel

    def __call__(
        self,
        gen_batch: Batch,
        **kwargs,
    ) -> torch.Tensor:
        assert gen_batch.x.requires_grad

        n_ftx = gen_batch.xnot.shape[-1]

        E_sel = gen_batch.x[..., conf.loader.x_ftx_energy_pos]
        E_unsel = gen_batch.xnot[..., conf.loader.x_ftx_energy_pos]
        # everything but the energy
        pos_unsel = gen_batch.xnot[
            ..., torch.arange(n_ftx) != conf.loader.x_ftx_energy_pos
        ]

        pos_sel = gen_batch.x[
            ..., torch.arange(n_ftx) != conf.loader.x_ftx_energy_pos
        ]
        batch_unsel = gen_batch.batchnot

        # Penalize for being far away from the center
        # Get std and mean for the selected points
        # and index them for the not selected points
        mean = torch_scatter.scatter_mean(pos_sel, gen_batch.batch, dim=0)[
            batch_unsel
        ]
        std = torch_scatter.scatter_std(pos_sel, gen_batch.batch, dim=0)[
            batch_unsel
        ]

        dist = (pos_unsel - mean).abs()
        dist_scaled = dist / (std * 2)

        loss = (dist_scaled * (dist_scaled > 1).float()).sum()

        E_min = torch_scatter.scatter_mean(E_sel, gen_batch.batch, dim=0)[
            batch_unsel
        ]
        loss += ((E_unsel - E_min) * (E_unsel > E_min).float()).sum()

        return loss
