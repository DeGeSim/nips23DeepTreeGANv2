import torch


class LossGen:
    def __init__(self) -> None:
        self.lossf = torch.nn.MSELoss()

    def __call__(
        self,
        sim_crit: torch.Tensor,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert sim_crit.requires_grad and sim_crit.requires_grad
        return self.lossf(torch.ones_like(sim_crit), sim_crit) + self.lossf(
            torch.zeros_like(gen_crit), gen_crit
        )
