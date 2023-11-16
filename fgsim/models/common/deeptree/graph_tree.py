from typing import Tuple

import torch
from torch_geometric.data import Batch

from fgsim.io.batch_tools import batch_from_pcs_list

from .tree import Tree


class TreeGraph:
    def __init__(
        self,
        tftx: torch.Tensor,
        tree: Tree,
        global_features: torch.Tensor,
        cur_level: int = 0,
    ):
        self.tftx = tftx
        self.tree = tree
        self.cur_level = cur_level
        self.global_features = global_features

    def get_batch_skeleton(self) -> Tuple[Batch, torch.Tensor]:
        """
        It takes a batch of graphs and returns the skeleton batch of graphs

        Returns:
        A batch object with the edge_index

        """
        last_level_batch_idx = self.tree.tbatch_by_level[-1][
            self.tree.idxs_by_level[-1]
        ]
        res: Batch = batch_from_pcs_list(
            self.tftx_by_level(-1),
            last_level_batch_idx,
        )
        self.__presaved_batch: Batch = res.detach().clone()
        self.__presaved_batch.x = None
        self.__presaved_batch_indexing: torch.Tensor = torch.argsort(
            last_level_batch_idx
        )
        return self.__presaved_batch, self.__presaved_batch_indexing

    def tftx_by_level(self, slice):
        return self.tftx[self.tree.idxs_by_level[slice]]

    def batch_by_level(self, slice):
        return self.tree.tbatch_by_level[self.cur_level][
            self.tree.idxs_by_level[slice]
        ]
