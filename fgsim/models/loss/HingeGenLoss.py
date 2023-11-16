import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_crit.requires_grad
        batch_size = kwargs["gen_batch"].num_graphs
        assert len(gen_crit) % batch_size == 0
        n_discs = len(gen_crit) // batch_size

        loss = -1 * gen_crit
        loss_d = {
            str(icrit): loss[icrit * batch_size : (icrit + 1) * batch_size].mean()
            for icrit in range(n_discs)
        }
        return loss_d
