import torch


class LossGen:
    def __init__(self) -> None:
        self.lossf = torch.nn.MSELoss()

    def __call__(
        self,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_crit.requires_grad
        return self.lossf(gen_crit, torch.ones_like(gen_crit))
