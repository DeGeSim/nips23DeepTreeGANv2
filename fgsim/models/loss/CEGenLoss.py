import torch

from fgsim.config import conf, device


class LossGen:
    def __init__(
        self,
    ) -> None:
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )

    def __call__(
        self,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_crit.requires_grad
        errG = self.criterion(gen_crit, torch.ones_like(gen_crit))
        return errG
