"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""


from pathlib import Path
from typing import Union

import torch

from fgsim.config import conf
from fgsim.ml.holder import Holder

holder = Holder()

cppath = Path(conf.path.run_path) / "checkpoint.torch"
checkpoint = torch.load(cppath)


def recur_update(cp_dict, model_dict):
    valid_keys = model_dict.keys()

    for k in list(cp_dict.keys()):
        if k not in valid_keys:
            del cp_dict[k]
    for k in valid_keys:
        if k not in cp_dict:
            cp_dict[k] = model_dict[k]
        elif isinstance(model_dict[k], dict):
            assert isinstance(cp_dict[k], dict)
            recur_update(cp_dict[k], model_dict[k])
        elif isinstance(model_dict[k], list):
            assert isinstance(cp_dict[k], list)
            assert len(cp_dict[k]) == len(model_dict[k])
        elif isinstance(model_dict[k], torch.Tensor):
            if model_dict[k].shape != cp_dict[k].shape:
                cp_dict[k] = model_dict[k]
        elif isinstance(model_dict[k], Union[int, float]):
            pass
        elif cp_dict[k] == model_dict[k]:
            pass
        else:
            raise Exception()
    assert set(cp_dict.keys()) == set(valid_keys)


recur_update(checkpoint["models"], holder.models.state_dict())
holder.models.load_state_dict(checkpoint["models"])
checkpoint["best_model"] = checkpoint["models"]

if len(holder.swa_models):
    for pname, part in holder.swa_models.items():
        recur_update(checkpoint["swa_model"][pname], part.state_dict())
        part.load_state_dict(checkpoint["swa_model"][pname])
    checkpoint["best_swa_model"] = checkpoint["swa_model"]

for part in ["gen", "disc"]:
    recur_update(
        checkpoint["optims"]["optimizers"][part],
        holder.optims._optimizers[part].state_dict(),
    )
    if part in holder.optims._schedulers:
        recur_update(
            checkpoint["optims"]["schedulers"][part],
            holder.optims._schedulers[part].state_dict(),
        )
holder.optims.load_state_dict(checkpoint["optims"])
torch.save(checkpoint, cppath)
