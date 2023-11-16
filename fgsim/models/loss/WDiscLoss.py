import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        sim_crit: torch.Tensor,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert sim_crit.requires_grad and sim_crit.requires_grad
        return gen_crit.mean() - sim_crit.mean()
