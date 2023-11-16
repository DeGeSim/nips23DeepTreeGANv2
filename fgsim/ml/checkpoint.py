import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fgsim.config import conf
from fgsim.monitoring import logger
from fgsim.utils.check_model import contains_nans


class CheckPointManager:
    def __init__(self, holder) -> None:
        self.holder = holder
        self._last_checkpoint_time = datetime.now()
        self._training_start_time = datetime.now()
        self.saved_first_checkpoint = False
        self.state_path = Path(conf.path.run_path) / "state.yaml"
        self.state_path_old = Path(conf.path.run_path) / "state_old.yaml"
        self.checkpoint_path = Path(conf.path.run_path) / "checkpoint.torch"
        self.checkpoint_path_old = Path(conf.path.run_path) / "checkpoint_old.torch"

    def load_checkpoint(self):
        if not (self.state_path.is_file() and (self.checkpoint_path).is_file()):
            if conf.command != "train":
                raise FileNotFoundError("Could not find checkpoint")
            logger.warning("Proceeding without loading checkpoint.")
            return
        self._load_checkpoint_path(self.state_path, self.checkpoint_path)

    def load_ray_checkpoint(self, ray_tmp_checkpoint_path: str):
        checkpoint_path = Path(ray_tmp_checkpoint_path) / "cp.pth"
        state_path = Path(ray_tmp_checkpoint_path) / "state.pth"
        self._load_checkpoint_path(state_path, checkpoint_path)

    def _load_checkpoint_path(self, state_path, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.holder.device)
        strict = not conf.training.implant_checkpoint

        assert not contains_nans(checkpoint["models"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.holder.models.load_state_dict(checkpoint["models"], strict)
        if conf.command == "train":
            self.holder.optims.load_state_dict(checkpoint["optims"])
        self.best_model_state = checkpoint["best_model"]

        if "swa_model" in checkpoint:
            for pname, part in self.holder.swa_models.items():
                part.load_state_dict(checkpoint["swa_model"][pname], strict)
            self.best_swa_model_state = checkpoint["best_swa_model"]

        self.holder.history.update(checkpoint["history"])
        self.holder.state.update(OmegaConf.load(state_path))
        self.checkpoint_loaded = True

        logger.warning(
            "Loaded model from checkpoint at"
            + f" epoch {self.holder.state['epoch']}"
            + f" grad_step {self.holder.state['grad_step']}."
        )

    def select_best_model(self):
        if conf.ray and conf.command == "test":
            return
        strict = not conf.training.implant_checkpoint
        self.holder.models.load_state_dict(self.best_model_state, strict)
        self.holder.models = self.holder.models.float().to(self.holder.device)
        if len(self.holder.swa_models.keys()):
            for n, p in self.holder.swa_models.items():
                p.load_state_dict(self.best_swa_model_state[n], strict)
                self.holder.swa_models[n] = (
                    self.holder.swa_models[n].float().to(self.holder.device)
                )

    def save_checkpoint(
        self,
    ):
        if conf.debug:
            return
        push_to_old(self.checkpoint_path, self.checkpoint_path_old)
        safed = {
            "models": self.holder.models.state_dict(),
            "optims": cylerlr_workaround(self.holder.optims.state_dict()),
            "best_model": self.best_model_state,
            "history": self.holder.history,
        }

        if len(self.holder.swa_models.keys()):
            safed["swa_model"] = {}
            for pname, part in self.holder.swa_models.items():
                safed["swa_model"][pname] = part.state_dict()
            safed["best_swa_model"] = self.best_swa_model_state

        torch.save(safed, self.checkpoint_path)

        push_to_old(self.state_path, self.state_path_old)
        OmegaConf.save(config=self.holder.state, f=self.state_path)
        self._last_checkpoint_time = datetime.now()
        logger.warning(
            f"{self._last_checkpoint_time.strftime('%d/%m/%Y, %H:%M:%S')}"
            f"Checkpoint saved to {self.checkpoint_path}"
        )

    def save_ray_checkpoint(self, ray_tmp_checkpoint_path: str):
        checkpoint_path = Path(ray_tmp_checkpoint_path) / "cp.pth"
        state_path = Path(ray_tmp_checkpoint_path) / "state.pth"
        torch.save(
            {
                "models": self.holder.models.state_dict(),
                "optims": cylerlr_workaround(self.holder.optims.state_dict()),
                "best_model": self.best_model_state,
                "history": self.holder.history,
            },
            checkpoint_path,
        )
        OmegaConf.save(config=self.holder.state, f=state_path)
        self._last_checkpoint_time = datetime.now()
        logger.warning(
            f"{self._last_checkpoint_time.strftime('%d/%m/%Y, %H:%M:%S')}"
            f"Checkpoint saved to {ray_tmp_checkpoint_path}"
        )
        return ray_tmp_checkpoint_path

    def checkpoint_after_time(self):
        now = datetime.now()
        time_since_last_checkpoint = (
            now - self._last_checkpoint_time
        ).seconds // 60
        interval = conf.training.checkpoint_minutes

        if time_since_last_checkpoint > interval:
            self.save_checkpoint()

        time_since_training_start = (now - self._training_start_time).seconds // 60
        if time_since_training_start > 5 and not self.saved_first_checkpoint:
            self.saved_first_checkpoint = True
            self.save_checkpoint()


def cylerlr_workaround(sd):
    for pname in sd["schedulers"]:
        if "_scale_fn_ref" in sd["schedulers"][pname]:
            del sd["schedulers"][pname]["_scale_fn_ref"]
    return sd


def push_to_old(path_new, path_old):
    if os.path.isfile(path_new):
        if os.path.isfile(path_old):
            os.remove(path_old)
        os.rename(path_new, path_old)
