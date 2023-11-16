import torch


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self) -> None:
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(
        self,
        sim_crit: torch.Tensor,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert sim_crit.requires_grad and sim_crit.requires_grad
        sample_disc_loss = self.criterion(sim_crit, torch.ones_like(sim_crit))
        gen_disc_loss = self.criterion(gen_crit, torch.zeros_like(gen_crit))

        return gen_disc_loss + sample_disc_loss
