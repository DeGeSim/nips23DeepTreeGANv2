"""Manages the networks for the holder class"""
import importlib
from typing import Dict, Union

import torch
from omegaconf import DictConfig, OmegaConf


class SubNetworkCollector(torch.nn.Module):
    """Collect all parts of the model in one to allow
    things like holder.model.to(device), float() ect."""

    def __init__(self, pconf: DictConfig) -> None:
        super().__init__()
        self.pconf = pconf
        self.parts: Dict[str, torch.nn.Module] = {}

        for name, submodelconf in pconf.items():
            modelparams = (
                submodelconf.params if submodelconf.params is not None else {}
            )
            submodel: torch.nn.Module = import_nn(
                name, submodelconf.name, modelparams
            )

            self.parts[name] = submodel
            setattr(self, name, submodel)

    def forward(self, in_X):
        for model in self.parts.values():
            in_X = model(in_X)
        return in_X

    def get_par_dict(self) -> Dict:
        return {
            name: submodel.parameters() for name, submodel in self.parts.items()
        }

    def __getitem__(self, subnetworkname: str) -> torch.nn.Module:
        return self.parts[subnetworkname]

    def to(self, *args, **kwargs):
        self.parts = {k: v.to(*args, **kwargs) for k, v in self.parts.items()}
        return self


# Import the python file containing the models with dynamically
def import_nn(
    partname: str, nn_name: str, modelparams: Union[DictConfig, dict]
) -> torch.nn.Module:
    model_module = None
    for import_path in [
        f"fgsim.models.{partname}.{nn_name}",
        f"fgsim.models.{partname}.{nn_name}.model",
        f"fgsim.models.{partname}.{nn_name}.{nn_name}",
    ]:
        try:
            model_module = importlib.import_module(import_path)
            if not hasattr(model_module, "ModelClass"):
                raise ModuleNotFoundError
            break
        except ModuleNotFoundError:
            model_module = None

    if model_module is None:
        raise ImportError
    if isinstance(modelparams, DictConfig):
        modelparams = OmegaConf.to_container(modelparams)

    submodel = model_module.ModelClass(**modelparams)

    return submodel
