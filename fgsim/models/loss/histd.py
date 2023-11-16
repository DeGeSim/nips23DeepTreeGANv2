import torch


class LossGen:
    def __init__(self):
        self.timesteps = 0
        self.sum_parameters = []

    def __call__(self, holder, **kwargs):
        disc: torch.nn.Module = holder.models.disc
        if self.timesteps == 0:
            for p in disc.parameters():
                param = p.data.clone()
                self.sum_parameters.append(param)
            self.timesteps += 1
            return None
        else:
            loss = 0.0
            for i, p in enumerate(disc.parameters()):
                loss += torch.sum(
                    (p - (self.sum_parameters[i].data / self.timesteps)) ** 2
                )
                self.sum_parameters[i] += p.data.clone()
            self.timesteps += 1
            return loss
