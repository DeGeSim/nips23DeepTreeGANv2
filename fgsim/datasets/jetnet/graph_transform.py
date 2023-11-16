import torch
from caloutils.utils.batch import add_graph_attr, add_node_attr, init_batch
from torch_geometric.data import Batch

from .readin import read_chunks


def events_to_batch_unscaled(chks: tuple[torch.Tensor, torch.Tensor]) -> Batch:
    y, x = [torch.stack(e) for e in zip(*read_chunks(chks))]
    nonzero = x[..., 3].reshape(-1).bool()
    n_pointsv = x[..., 3].sum(1).long()
    from fgsim.config import conf

    assert (
        n_pointsv == y[..., conf.loader.y_features.index("num_particles")]
    ).all()

    # create batch_idx array
    batch_idx = torch.arange(x.shape[0]).repeat_interleave(x.shape[1]).long()

    batch_idx = batch_idx[nonzero]
    xn = x.reshape(x.shape[0] * x.shape[1], x.shape[2])[nonzero][..., :3]

    batch = init_batch(batch_idx)
    add_node_attr(batch, "x", xn)
    add_graph_attr(batch, "y", y)
    add_graph_attr(batch, "n_pointsv", n_pointsv)
    return batch
