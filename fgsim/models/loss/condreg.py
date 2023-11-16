import torch


class LossGen:
    def __call__(
        self,
        gen_condreg: torch.Tensor,
        sim_batch: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_condreg.requires_grad
        y = sim_batch.y.detach()
        mean, std = y.mean(0), y.std(0)
        std[std == 0] = 1
        yp = (y - mean) / (std + 1e-3)

        yhat = gen_condreg
        yphat = (yhat - mean) / (std + 1e-3)
        loss = (yp - yphat).abs().mean()
        return loss
