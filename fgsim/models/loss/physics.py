import torch
from torch_geometric.data import Batch

from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self):
        pass

    def lossf(self, pc):
        energy = pc[..., 0]
        x = pc[..., 1]
        y = pc[..., 2]
        z = pc[..., 3]
        r = torch.sqrt(x**2 + y**2)
        # Penalty for negative Energies
        e_lt_zero = torch.sum(-energy * (energy < 0))
        # Penalty for super high values > 100keV
        e_high = torch.sum(energy * (energy > 0.1))
        # Only forward direction for now
        z_lt_zero = torch.sum(-z * (z < 0))

        # Limit r  to 270cm
        # and z to 550cm
        z_high = torch.sum(z * (z > 550))
        r_high = torch.sum(r * (r > 270))
        return sum([e_lt_zero, e_high, z_lt_zero, z_high, r_high])

    def __call__(self, holder: Holder, batch: Batch):
        with torch.no_grad():
            real_loss = self.lossf(batch.x)
            if real_loss != 0:
                raise RuntimeError
        loss = self.lossf(holder.gen_points_w_grad.x)
        loss.backward(retain_graph=True)
        return float(loss)
