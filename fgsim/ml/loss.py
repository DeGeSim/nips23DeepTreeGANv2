"""Dynamically import the losses"""
import importlib
from typing import Dict, Optional, Protocol

import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.monitoring.metrics_aggr import GradHistAggregator, MetricAggregator
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.modelgrads import get_grad_dict
from fgsim.utils import check_tensor


class LossFunction(Protocol):
    def __call__(self, holder, batch: Batch) -> torch.Tensor:
        ...


class SubNetworkLoss:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step
    Saves all losses in a MetricAggregator
    """

    def __init__(
        self, subnetworkname: str, pconf: DictConfig, train_log: TrainLog
    ) -> None:
        self.name = subnetworkname
        self.pconf = pconf
        self.train_log = train_log
        self.parts: Dict[str, LossFunction] = {}
        self.metric_aggr = MetricAggregator()
        self.grad_aggr = GradHistAggregator()

        for lossname, lossconf in pconf.items():
            assert lossname != "parts"
            params = lossconf if lossconf is not None else {}
            if isinstance(params, DictConfig):
                params = OmegaConf.to_container(params)
            del params["factor"]
            try:
                loss_module = importlib.import_module(
                    f"fgsim.models.loss.{lossname}"
                )
                loss = loss_module.LossGen(**params)
            except ImportError:
                loss = getattr(torch.nn, lossname)().to(device)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __getitem__(self, lossname: str) -> LossFunction:
        return self.parts[lossname]

    def __call__(self, holder, **res):
        losses_dict: Dict[str, Optional[torch.Tensor]] = {}
        # Iterate over the lossed for the model
        for lossname, loss in self.parts.items():
            loss_value = loss(holder=holder, **res)
            if loss_value is None:
                pass
            elif isinstance(loss_value, torch.Tensor):
                losses_dict[lossname] = loss_value * self.pconf[lossname]["factor"]
            elif isinstance(loss_value, dict):
                for sublossname, sublossval in loss_value.items():
                    losses_dict[f"{lossname}/{sublossname}"] = (
                        sublossval * self.pconf[lossname]["factor"]
                    )
            else:
                raise Exception
        for v in losses_dict.values():
            check_tensor(v)

        partloss: torch.Tensor = sum(losses_dict.values())
        if conf.models[self.name].retain_graph_on_backprop:
            partloss.backward(retain_graph=True)
        else:
            partloss.backward()

        self.grad_aggr.append_dict(get_grad_dict(holder.models[self.name]))
        self.metric_aggr.append_dict(
            {k: v.detach().cpu().numpy() for k, v in losses_dict.items()}
        )


class LossesCol:
    """Holds all losses for all subnetworks as attributes or as a dict."""

    def __init__(self, train_log: TrainLog) -> None:
        self.parts: Dict[str, SubNetworkLoss] = {}

        for subnetworkname, subnetworkconf in conf.models.items():
            snl = SubNetworkLoss(subnetworkname, subnetworkconf.losses, train_log)
            self.parts[subnetworkname] = snl
            setattr(self, subnetworkname, snl)

    def __getitem__(self, subnetworkname: str) -> SubNetworkLoss:
        return self.parts[subnetworkname]

    def __iter__(self):
        return iter(self.parts.values())
