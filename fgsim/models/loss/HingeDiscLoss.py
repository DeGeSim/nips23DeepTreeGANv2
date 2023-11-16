import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def hinge_act(self, x: torch.Tensor):
        return (x < 0).float() * x

    def __call__(
        self,
        sim_crit: torch.Tensor,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert not kwargs["gen_batch"].x.requires_grad
        assert sim_crit.requires_grad and sim_crit.requires_grad
        batch_size = kwargs["gen_batch"].num_graphs
        assert len(gen_crit) % batch_size == 0
        n_discs = len(gen_crit) // batch_size

        loss = -self.hinge_act(sim_crit - 1)
        loss += -self.hinge_act(-gen_crit - 1)
        loss_d = {
            str(icrit): loss[icrit * batch_size : (icrit + 1) * batch_size].mean()
            for icrit in range(n_discs)
        }

        return loss_d
