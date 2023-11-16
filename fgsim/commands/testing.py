"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from fgsim.config import conf, device
from fgsim.datasets import Dataset
from fgsim.ml.eval import eval_res_d, gen_res_from_sim_batches
from fgsim.ml.holder import Holder
from fgsim.monitoring import logger

batch_size = conf.loader.batch_size


@dataclass
class TestDataset:
    res_d: dict
    hlvs_dict: Optional[Dict[str, Dict[str, np.ndarray]]]
    grad_step: int
    loader_hash: str
    hash: str


def test_procedure() -> None:
    holder: Holder = Holder(device)
    ds_dict = {
        best_or_last: get_testing_datasets(holder, best_or_last)
        for best_or_last in ["best"]  # "last",
    }

    for best_or_last in ds_dict.keys():
        test_data: TestDataset = ds_dict[best_or_last]
        plot_path = Path(f"{conf.path.run_path}")
        plot_path.mkdir(exist_ok=True)

        if best_or_last == "best":
            step = holder.state.best_step
            epoch = holder.state.best_epoch

        else:
            step = holder.state.grad_step
            epoch = holder.state.epoch

        logger.info(f"Evalutating {best_or_last} dataset")
        eval_res_d(
            test_data.res_d,
            holder,
            step,
            epoch,
            [f"test_{best_or_last}"],
            plot_path,
        )
        holder.train_log.flush()

    exit(0)


def get_testing_datasets(holder: Holder, best_or_last) -> TestDataset:
    ds_path = Path(conf.path.run_path) / f"test_{best_or_last}" / "testdata.pt"
    ds_path.parent.mkdir(exist_ok=True)
    test_data: TestDataset
    if best_or_last == "best":
        step = holder.state.best_step
    else:
        step = holder.state.grad_step

    if ds_path.is_file():
        logger.info(f"Loading test dataset from {ds_path}")
        test_data = TestDataset(
            **torch.load(ds_path, map_location=torch.device("cpu"))
        )
        reprocess = False

        if test_data.loader_hash != conf.loader_hash:
            logger.warning(
                f"Loader hash changed, reprocessing {best_or_last} Dataset"
            )
            reprocess = True
        if test_data.hash != conf.hash:
            logger.warning(f"Hash changed, reprocessing {best_or_last} Dataset")
            reprocess = True

        if test_data.grad_step != step:
            logger.warning(
                f"New step available, reprocessing {best_or_last} Dataset"
            )
            reprocess = True
    else:
        reprocess = True

    if reprocess:
        # Make sure the batches are loaded
        loader = Dataset()

        # Check if we need to rerun the model
        # if yes, pickle it
        if best_or_last == "best":
            holder.checkpoint_manager.select_best_model()

        res_d = gen_res_from_sim_batches(loader.testing_batches, holder)

        test_data = TestDataset(
            res_d=res_d,
            grad_step=step,
            hlvs_dict=None,
            loader_hash=conf.loader_hash,
            hash=conf.hash,
        )
        logger.info(f"Saving test dataset to {ds_path}")
        torch.save(
            test_data.__dict__,
            ds_path,
        )

    return test_data
