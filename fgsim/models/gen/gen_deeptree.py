from math import prod
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.models.common import DynHLVsLayer, FtxScaleLayer, MPLSeq
from fgsim.models.common.deeptree import BranchingLayer, Tree, TreeGraph
from fgsim.monitoring import logger
from fgsim.utils import check_tensor

# from fgsim.plot.model_plotter import model_plotter


class ModelClass(nn.Module):
    def __init__(
        self,
        n_global: int,
        ancestor_mpl: Dict,
        child_mpl: Dict,
        branching_param: Dict,
        final_layer_scaler: bool,
        connect_all_ancestors: bool,
        dim_red_in_branching: bool,
        pruning: str,
        **kwargs,
    ):
        super().__init__()
        self.n_global = n_global
        self.n_cond = sum(conf.loader.cond_gen_features)
        self.batch_size = conf.loader.batch_size
        self.final_layer_scaler = final_layer_scaler
        self.ancestor_mpl = ancestor_mpl
        self.child_mpl = child_mpl
        self.dim_red_in_branching = dim_red_in_branching
        self.pruning = pruning

        self.features: list[int] = list(OmegaConf.to_container(conf.tree.features))
        self.branches: list[int] = list(OmegaConf.to_container(conf.tree.branches))
        n_levels = len(self.features)

        # Shape of the random vector
        self.z_shape = conf.loader.batch_size, 1, self.features[0]

        # Calculate the output points
        self.output_points = prod(self.branches)
        assert self.output_points == conf.loader.n_points
        assert self.features[-1] == conf.loader.n_features

        logger.debug(f"Generator output will be {self.output_points}")
        if conf.loader.n_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.n_points} < {self.output_points}"
            )

        self.tree = Tree(
            batch_size=conf.loader.batch_size,
            connect_all_ancestors=connect_all_ancestors,
            branches=self.branches,
            features=self.features,
        )

        self.branchings: nn.ModuleList[BranchingLayer] = nn.ModuleList(
            [
                BranchingLayer(
                    tree=self.tree,
                    level=level,
                    n_global=n_global,
                    n_cond=self.n_cond,
                    dim_red=self.dim_red_in_branching,
                    **branching_param,
                )
                for level in range(n_levels - 1)
            ]
        )

        self.hlvs_nn = nn.ModuleList(
            [
                DynHLVsLayer(
                    n_features=nfeatures,
                    n_cond=self.n_cond,
                    n_global=self.n_global,
                    batch_size=self.batch_size,
                )
                for nfeatures in self.features[:-1]
            ]
        )
        # self.postbr_hlvs: nn.ModuleList[DynHLVsLayer] = nn.ModuleList(
        #     [
        #         DynHLVsLayer(
        #             n_features=nfeatures,
        #             n_cond=self.n_cond,
        #             n_global=self.n_global,
        #             batch_size=self.batch_size,
        #         )
        #         for nfeatures in self.features[1:]
        #     ]
        # )

        if self.ancestor_mpl["n_mpl"] > 0:
            self.ac_mpl = nn.ModuleList(
                [
                    self.wrap_layer_init(ilevel, type="ac")
                    for ilevel in range(n_levels - 1)
                ]
            )

        if self.child_mpl["n_mpl"] > 0:
            self.child_conv_layers = nn.ModuleList(
                [
                    self.wrap_layer_init(ilevel, type="child")
                    for ilevel in range(n_levels - 1)
                ]
            )

        if self.final_layer_scaler:
            self.ftx_scaling = FtxScaleLayer(self.features[-1])

        # Allocate the Tensors used later to construct the batch
        self.presaved_batch: Optional[Batch] = None
        self.presaved_batch_indexing: Optional[torch.Tensor] = None

        # self.ac_mpl[-1].mpls[-1].nn.seq[-1].register_backward_hook(
        #     lambda m, go, gi: logger.debug(
        #         f"grad: ac abs mean {gi[0].abs().mean():.0e} std {gi[0].std():.0e}"
        #     )
        # )
        # self.branchings[0].proj_nn.seq[-1].register_backward_hook(
        #     lambda m, go, gi: logger.debug(
        #         f"grad: br abs mean {gi[0].abs().mean():.0e} std {gi[0].std():.0e}"
        #     )
        # )

    def wrap_layer_init(self, ilevel, type: str):
        if type == "ac":
            conv_param = self.ancestor_mpl
        elif type == "child":
            conv_param = self.child_mpl
        else:
            raise Exception

        return MPLSeq(
            in_features=(
                self.features[ilevel + int(self.dim_red_in_branching)]
                if type == "ac"
                else self.features[ilevel + 1]
            ),
            out_features=self.features[ilevel + 1],
            n_cond=self.n_cond,
            n_global=self.n_global,
            # batch_size=self.batch_size,
            **conv_param,
        )

    def forward(
        self, random_vector: torch.Tensor, cond: torch.Tensor, n_pointsv
    ) -> Batch:
        batch_size = self.batch_size
        features = self.features
        device = random_vector.device
        n_levels = len(self.features)

        # Init the graph object
        graph_tree: TreeGraph = TreeGraph(
            tftx=random_vector.reshape(batch_size, features[0]),
            global_features=torch.empty(
                batch_size, self.n_global, dtype=torch.float, device=device
            ),
            tree=self.tree,
        )

        # Assign the index vectors for the root node tree
        batchidx = self.tree.tbatch_by_level[0]
        idxs_level = self.tree.idxs_by_level[0]
        batch_level = batchidx[idxs_level]
        edge_index = self.tree.ancestor_ei(0)
        edge_attr = self.tree.ancestor_ea(0)

        # model_plotter.save_tensor("input noise", graph_tree.tftx)
        print_dist("initial", graph_tree.tftx_by_level(0))
        # Do the branching
        for ilevel in range(n_levels - 1):
            # assert graph_tree.tftx.shape[1] == self.tree.features[ilevel]
            # assert graph_tree.tftx.shape[0] == (
            #     self.tree.tree_lists[ilevel][-1].idxs[-1] + 1
            # )

            # Assign the global features
            assert (
                batch_level
                == self.tree.tbatch_by_level[ilevel][
                    self.tree.idxs_by_level[ilevel]
                ]
            ).all()
            ftx_level = graph_tree.tftx_by_level(ilevel)
            graph_tree.global_features = self.hlvs_nn[ilevel](
                x=ftx_level, cond=cond, batch=batch_level
            )

            # Branch the leaves
            graph_tree = self.branchings[ilevel](graph_tree, cond)
            check_tensor(graph_tree.tftx)

            # Assign the new indices for the updated tree
            batchidx = self.tree.tbatch_by_level[ilevel + 1]
            idxs_level = self.tree.idxs_by_level[ilevel + 1]
            batch_level = batchidx[idxs_level]
            edge_index = self.tree.ancestor_ei(ilevel + 1)
            edge_attr = self.tree.ancestor_ea(ilevel + 1)

            # # Assign the global features
            # ftx_level = graph_tree.tftx_by_level(ilevel + 1)
            # graph_tree.global_features = self.postbr_hlvs[ilevel](
            #     x=ftx_level, cond=cond, batch=batch_level
            # )

            print_dist("branching", graph_tree.tftx_by_level(ilevel + 1))
            assert (
                graph_tree.tftx.shape[1]
                == self.tree.features[ilevel + int(self.dim_red_in_branching)]
            )
            assert graph_tree.tftx.shape[0] == (
                self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
            )
            # model_plotter.save_tensor(
            #     f"branching output level{ilevel+1}",
            #     graph_tree.tftx_by_level(ilevel + 1),
            # )

            if self.ancestor_mpl["n_mpl"] > 0:
                graph_tree.tftx = self.ac_mpl[ilevel](
                    x=graph_tree.tftx,
                    cond=cond,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batchidx,
                    global_features=graph_tree.global_features,
                )
                print_dist("ac", graph_tree.tftx_by_level(ilevel + 1))
                assert graph_tree.tftx.shape[1] == self.tree.features[ilevel + 1]
                assert graph_tree.tftx.shape[0] == (
                    self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
                )
                # model_plotter.save_tensor(
                #     f"ancestor conv output level{ilevel+1}",
                #     graph_tree.tftx_by_level(ilevel + 1),
                # )

            if self.child_mpl["n_mpl"] > 0:
                graph_tree.tftx = self.child_conv_layers[ilevel](
                    x=graph_tree.tftx,
                    cond=cond,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batchidx,
                )
                assert graph_tree.tftx.shape[1] == self.tree.features[ilevel + 1]
                assert graph_tree.tftx.shape[0] == (
                    self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
                )

                # model_plotter.save_tensor(
                #     f"child conv output level{ilevel+1}",
                #     graph_tree.tftx_by_level(ilevel + 1),
                # )
            check_tensor(graph_tree.tftx)

        batch = self.construct_batch(graph_tree, n_pointsv)

        if self.final_layer_scaler:
            batch.x = self.ftx_scaling(batch.x)

        return batch

    def construct_batch(self, graph_tree: TreeGraph, n_pointsv: torch.Tensor):
        device = graph_tree.tftx.device
        batch_size = self.batch_size
        n_features = self.features[-1]
        if self.presaved_batch is None:
            (
                self.presaved_batch,
                self.presaved_batch_indexing,
            ) = graph_tree.get_batch_skeleton()

        batch = self.presaved_batch.clone()
        batch.x = graph_tree.tftx_by_level(-1)[self.presaved_batch_indexing]

        x = batch.x.reshape(self.output_points, batch_size, n_features).transpose(
            0, 1
        )

        sel_point_idx = self.get_sel_idxs(x, n_pointsv)

        batch.x, batch.xnot = batch.x[sel_point_idx], batch.x[~sel_point_idx]
        batch.batch, batch.batchnot = (
            batch.batch[sel_point_idx],
            batch.batch[~sel_point_idx],
        )

        # Set the slice_dict to allow splitting the batch again
        batch._slice_dict["x"] = torch.concat(
            [torch.zeros(1, device=device), n_pointsv.cumsum(0)]
        )
        batch._slice_dict["xnot"] = torch.concat(
            [
                torch.zeros(1, device=device),
                (self.output_points - n_pointsv).cumsum(0),
            ]
        )
        batch._inc_dict["x"] = torch.zeros(self.batch_size, device=device)
        batch._inc_dict["xnot"] = torch.zeros(self.batch_size, device=device)
        batch.ptr = torch.hstack(
            [torch.zeros(1, device=device, dtype=torch.long), n_pointsv.cumsum(0)]
        )
        batch.num_nodes = len(batch.x)

        assert (
            len(batch.x) + len(batch.xnot) == self.output_points * self.batch_size
        )
        assert batch.x.shape[-1] == self.features[-1]
        assert batch.num_graphs == self.batch_size
        check_tensor(batch.x)

        return batch

    def get_sel_idxs(self, x: torch.Tensor, n_pointsv: torch.Tensor):
        device = x.device
        gidx = torch.zeros(
            self.output_points * self.batch_size, device=device
        ).bool()

        if self.pruning == "cut":
            return get_cut_idxs(gidx, n_pointsv, self.output_points)
        elif self.pruning == "topk":
            shift = 0
            for xe, ne in zip(x, n_pointsv):
                idxs = (
                    xe[..., conf.loader.x_ftx_energy_pos]
                    .topk(k=int(ne), dim=0, largest=True, sorted=False)
                    .indices
                ) + shift
                gidx[idxs] = True
                shift += self.output_points
            return gidx
        else:
            raise Exception

    def to(self, device):
        super().to(device)
        self.tree.to(device)
        return self


@torch.jit.script
def get_cut_idxs(gidx: torch.Tensor, n_pointsv: torch.Tensor, output_points: int):
    shift = 0
    for ne in n_pointsv:
        gidx[shift : shift + ne] = True
        shift += output_points
    return gidx


def print_dist(name, x):
    return
    if not conf.debug:
        return
    x = x.detach().cpu().numpy()
    print(
        f"{name}:\n\tmean\n\t{x.mean(0)} global {x.mean():.2f}\n"
        f"\tstd\n\t{x.std(0)} global {x.std():.2f}"
    )
