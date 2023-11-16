"""Dynamically import the losses"""
import importlib
from datetime import datetime

# from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from fgsim.config import conf
from fgsim.monitoring import MetricAggregator, TrainLog
from fgsim.monitoring.logger import logger


class EvaluationMetrics:
    """
    Calling this object will save the output of the validation batches to
    self.metric_aggr . In the log_metrics they are aggregated and logged
    and a score is calculated.
    """

    def __init__(self, train_log: TrainLog, history) -> None:
        self.train_log = train_log
        self.parts: Dict[str, Callable] = {}
        self._lastlosses: Dict[str, List[float]] = {}
        self.metric_aggr_val = MetricAggregator()
        self.history = history

        match conf.command:
            case "train":
                metrics = conf.metrics.debug if conf.debug else conf.metrics.val
            case "test":
                metrics = conf.metrics.test
            case _:
                metrics = []

        for metric_name in metrics:
            assert metric_name != "parts"
            # params = metric_conf if metric_conf is not None else DictConfig({})
            params = DictConfig({})
            loss = import_metric(metric_name, params)

            self.parts[metric_name] = loss
            setattr(self, metric_name, loss)

    def __call__(self, **kwargs) -> None:
        # During the validation, this function is called once per batch.
        # All losses are save in a dict for later evaluation log_lossses
        mval = {}
        with torch.no_grad():
            for metric_name, metric in self.parts.items():
                logger.debug(f"Running metric {metric_name}")
                start = datetime.now()
                comp_metrics = metric(**kwargs)
                delta = datetime.now() - start
                if delta.seconds > 10:
                    logger.info(
                        f"Metric {metric_name} took {delta.total_seconds()} sec"
                    )
                if isinstance(comp_metrics, dict):
                    # If the loss is processed for each hlv
                    # the return type is Dict[str,float]
                    for var, lossval in comp_metrics.items():
                        mval[f"{metric_name}/{var}"] = float(lossval)
                else:
                    mval[metric_name] = comp_metrics

        if conf.command == "train":
            self.metric_aggr_val.append_dict(
                {
                    k: v[0] if isinstance(v, tuple) and len(v) == 2 else v
                    for k, v in mval.items()
                }
            )
        else:
            self.test_md = mval

    def get_metrics(self) -> tuple[dict, Optional[list]]:
        """
        The function takes the validation metrics and computes the fraction
        of times that the value of this metric is smaller then the other runs
        """
        # Call metric_aggr to aggregate the collected metrics over the
        # validation batches.
        if conf.command == "test":
            up_metrics_d = self.test_md
            logger.info(up_metrics_d)
            return up_metrics_d, None

        # score calculatation if during training
        up_metrics_d = self.__aggr_dists(self.metric_aggr_val.aggregate())

        logstr = ""
        for metric_name, metric_val in up_metrics_d.items():
            val_metric_hist = self.history["val"][metric_name]
            val_metric_hist.append(metric_val)

            logstr += f"{metric_name} {val_metric_hist[-1]:.2f}"
            if len(val_metric_hist) > 1:
                logstr += (
                    f"(Δ{(val_metric_hist[-1]/val_metric_hist[-2]-1)*100:+.0f}%)"
                )
            logstr += "  "
        logger.info(logstr)
        # if conf.debug:
        #     return dict(), list()

        score = self.__compute_score_per_val(up_metrics_d)
        self.history["score"] = score
        return up_metrics_d, score

    def __aggr_dists(self, md):
        for dname in ["cdf", "sw1", "histd"]:
            if any([k.endswith(dname) for k in md.keys()]):
                md[f"dmean_{dname}"] = float(
                    np.nanmean([v for k, v in md.items() if k.endswith(dname)])
                )
        return md

    def __compute_score_per_val(self, up_metrics_d):
        # compute the stop_metric
        val_metrics = self.history["val"]
        # check which of the metrics should be used for the early stopping
        # If a metric returns a dict, use all
        val_metrics_names = [
            k
            for k in up_metrics_d.keys()
            if any([k.startswith(mn) for mn in conf.metrics.stopping])
        ]

        # for the following, all recordings need to have the same
        # lenght, so we count the most frequent one
        histlen = max([len(val_metrics[metric]) for metric in val_metrics_names])
        # collect all metrics for all validation runs in a 2d array
        loss_history = np.stack(
            [
                val_metrics[metric]
                for metric in val_metrics_names
                if len(val_metrics[metric]) == histlen
            ]
        )
        # for a given metric and validation run,
        # count the fraction of times that the value of this metric
        # is smaller then the other runs
        score = np.apply_along_axis(
            lambda row: np.array([np.mean(row >= e) for e in row]), 1, loss_history
        ).mean(0)
        score = [float(val) for val in score]
        return score

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]


def import_metric(metric_name: str, params: DictConfig) -> Callable:
    try:
        metrics = importlib.import_module("fgsim.models.metrics")
        fct = getattr(metrics, metric_name)
        return fct
    except AttributeError:
        pass
    MetricClass: Optional = None
    for import_path in [
        f"torch.nn.{metric_name}",
        f"fgsim.models.metrics.{metric_name}",
    ]:
        try:
            model_module = importlib.import_module(import_path)
            # Check if it is a class
            if not isinstance(model_module, type):
                if not hasattr(model_module, "Metric"):
                    raise ModuleNotFoundError
                else:
                    MetricClass = model_module.Metric
            else:
                MetricClass = model_module

            break
        except ModuleNotFoundError:
            MetricClass = None

    if MetricClass is None:
        raise ImportError

    metric = MetricClass(**params)

    return metric
