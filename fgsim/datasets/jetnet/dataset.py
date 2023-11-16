from torch_geometric.data import Batch

from ..base_dataset import BaseDS
from .graph_transform import events_to_batch_unscaled
from .readin import file_manager
from .scaler import scaler


class Dataset(BaseDS):
    def __init__(self):
        super().__init__(file_manager)

    def _chunk_to_batch(self, chunks):
        batch = scale_batch(events_to_batch_unscaled(chunks))
        return batch


def scale_batch(graph: Batch) -> Batch:
    graph.x = scaler.transform(graph.x, "x")
    graph.y = scaler.transform(graph.y, "y")
    return graph
