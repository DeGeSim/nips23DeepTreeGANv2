from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

import wandb
from fgsim.config import conf
from fgsim.monitoring import logger
from fgsim.monitoring.experiment_organizer import exp_orga_wandb


class TrainLog:
    """Initialized with the `holder`,
    provides the logging with wandb/tensorboard."""

    def __init__(self, state):
        # This code block is formatting the hyperparameters
        # for the experiment and creating a list of tags.
        self.state: DictConfig = state
        default_log = conf.command == "test" or (
            conf.command == "train" and not conf.debug
        )
        self.use_tb = False
        self.use_wandb = default_log

        if self.use_tb:
            self.writer: SummaryWriter = SummaryWriter(
                Path(conf.path.run_path) / "tb"
            )
            self.writer.add_scalar(
                "epoch",
                self.state["epoch"],
                self.state["grad_step"],
                new_style=True,
            )

        if self.use_wandb:
            if not conf.ray:
                wandb_name = f"{conf['hash']}_{conf.command}"
                if conf.command == "train":
                    wandb_id = exp_orga_wandb[conf["hash"]]
                elif conf.command == "test":
                    wandb_id = exp_orga_wandb[wandb_name]
                else:
                    raise NotImplementedError

                self.wandb_run = wandb.init(
                    id=wandb_id,
                    resume="must",
                    name=wandb_name,
                    group=conf["hash"],
                    dir=conf.path.run_path,
                    project=conf.project_name,
                    job_type=conf.command,
                    allow_val_change=True,
                    settings={"quiet": True},
                )
            wandb.define_metric("grad_step")
            wandb.define_metric("*", step_metric="grad_step")
            wandb.define_metric("val/*", goal="minimize")
            self._wandb_tmp: dict[str, float] = {}
            self._wandb_step = None
            self._wandb_epoch = None

        self.log_cmd_time()

    def log_cmd_time(self):
        if self.use_wandb:
            self.wandb_run.summary[f"time/{conf.command}"] = (
                datetime.now().strftime("%y-%m-%d-%H:%M")
            )

    def log_model_graph(self, model):
        if self.use_wandb:
            wandb.watch(model)

    def log_metrics(
        self,
        metrics_dict: dict[str, Union[tuple, float, torch.Tensor]],
        step=None,
        epoch=None,
        prefix=None,
    ):
        if conf.debug and conf.command != "test":
            return
        if step is None:
            step = self.state["grad_step"]
            epoch = self.state["epoch"]
        if epoch is None:
            raise Exception

        if prefix is not None:
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}

        for k in list(metrics_dict.keys()):
            v = metrics_dict[k]
            if isinstance(v, tuple) and len(v) == 2:
                metrics_dict[k] = v[0]
                metrics_dict[k + "δ"] = v[1]

        if self.use_tb:
            for name, value in metrics_dict.items():
                self.writer.add_scalar(name, value, step, new_style=True)

        if self.use_wandb:
            self._set_wandb_state(step, epoch)
            self._wandb_tmp.update(metrics_dict)

    def _set_wandb_state(self, step: int, epoch: int):
        if self._wandb_step is None and self._wandb_epoch is None:
            self._wandb_step = step
            self._wandb_epoch = epoch
        elif self._wandb_step != step or self._wandb_epoch != epoch:
            raise Exception("Step of metrics doesnt match")
        else:
            return

    def flush(self):
        if self.use_wandb:
            if len(self._wandb_tmp):
                logger.debug("Wandb flush")
                if conf.command == "test":
                    self._flush_test_wandb()
                else:
                    wandb.log(
                        self._wandb_tmp
                        | {
                            "epoch": self._wandb_epoch,
                            "grad_step": self._wandb_step,
                        }
                    )
                self._wandb_tmp = {}
                self._wandb_step = None
                self._wandb_epoch = None

    def _flush_test_wandb(self):
        wandb.log(
            {k: v for k, v in self._wandb_tmp.items() if "/best/" in k}
            | {"epoch": self._wandb_epoch},
        )
        for k, v in self._wandb_tmp.items():
            karr = k.split("/")
            match karr[0]:
                case "m":
                    self.wandb_run.summary["/".join(karr[1:])] = v
                case "p":
                    pass
                case _:
                    pass

    def log_figure(
        self,
        figure_name: str,
        figure: Figure,
        step: int = None,
    ):
        if step is None:
            step = self.state["grad_step"]
        if self.use_tb:
            self.writer.add_figure(tag=figure_name, figure=figure, global_step=step)
        if self.use_wandb:
            self._wandb_tmp.update({f"p/{figure_name}": wandb.Image(figure)})
        plt.close(figure)

    def write_trainstep_logs(self, interval) -> None:
        if not all(
            [
                hasattr(self.state, time)
                for time in [
                    "time_train_step_end",
                    "time_train_step_start",
                    "time_io_end",
                ]
            ]
        ):
            return
        traintime = (
            self.state.time_train_step_end - self.state.time_train_step_start
        )
        iotime = self.state.time_io_end - self.state.time_train_step_start
        utilisation = 1 - iotime / traintime

        self.log_metrics(
            {
                "batchtime": traintime / interval,
                "utilisation": utilisation,
                "processed_events": self.state.processed_events,
            },
            self.state["grad_step"],
            self.state["epoch"],
            prefix="speed",
        )

    def next_epoch(self) -> None:
        self.state["epoch"] += 1

        if self.use_tb:
            self.writer.add_scalar(
                "epoch",
                self.state["epoch"],
                self.state["grad_step"],
                new_style=True,
            )
        if self.use_wandb:
            wandb.log(
                data={"epoch": self.state["epoch"]},
                step=self.state["grad_step"],
            )

    def __del__(self):
        if self.use_tb:
            self.writer.flush()
            self.writer.close()

    def end(self) -> None:
        pass
