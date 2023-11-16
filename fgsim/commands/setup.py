import os
from pathlib import Path
from shutil import copytree

from omegaconf import OmegaConf

import wandb
from fgsim.config import conf, hyperparameters
from fgsim.monitoring.experiment_organizer import exp_orga_wandb
from fgsim.utils.oc_utils import dict_to_kv


def setup_procedure() -> str:
    rpath = Path(conf.path.run_path)

    # If the experiment has been setup, exit directly
    if rpath.is_dir():
        print(f"Experiment already setup with hash {conf.hash}.")
        return

    os.makedirs(conf.path.run_path, exist_ok=True)

    OmegaConf.save(conf, rpath / "conf.yaml")
    OmegaConf.save(hyperparameters, rpath / "hyperparameters.yaml")

    # Backup the python files
    copytree(
        "fgsim",
        rpath / "fgsim",
        ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
        dirs_exist_ok=True,
    )
    copytree(
        "fgsim",
        rpath / "fgbackup",
        ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
        dirs_exist_ok=True,
    )
    setup_experiment()
    print(f"Experiment setup with hash {conf.hash}.")
    return


def filter_paths(d, f):
    if f in ["old", "__pycache__"] or f.startswith("."):
        return True
    if (Path(d) / Path(f)).is_dir():
        return False
    if f.endswith(".py") or f.endswith(".yaml"):
        return False
    return True


def setup_experiment() -> None:
    from fgsim.config import conf

    """Generates a new experiment."""
    if conf.hash in exp_orga_wandb.keys():
        if conf.ray:
            return
        raise Exception("Experiment exists")

    # Format the hyperparameter
    from fgsim.config import hyperparameters

    assert len(hyperparameters) > 0
    hyperparameters_keyval_list = dict(dict_to_kv(hyperparameters))
    hyperparameters_keyval_list["hash"] = conf["hash"]
    hyperparameters_keyval_list["loader_hash"] = conf["loader_hash"]
    tags_list = list(set(conf.tag.split("_")))

    # wandb
    run_train = wandb.init(
        project=conf.project_name,
        name=f"{conf['hash']}_train",
        group=conf["hash"],
        tags=tags_list,
        config=hyperparameters_keyval_list,
        dir=conf.path.run_path,
        job_type="train",
        resume=False,
        reinit=True,
        allow_val_change=True,
        settings={
            "quiet": True,
            "disable_job_creation": True,
        },
    )
    codepath = (Path(conf.path.run_path) / "fgsim/").absolute()
    assert codepath.is_dir()
    run_train.log_code(str(codepath), conf.hash)

    exp_orga_wandb[conf["hash"]] = run_train.id

    run_test = wandb.init(
        project=conf.project_name,
        name=f"{conf['hash']}_test",
        group=conf["hash"],
        tags=tags_list,
        config=hyperparameters_keyval_list,
        dir=conf.path.run_path,
        job_type="test",
        reinit=True,
        resume=False,
        allow_val_change=True,
        settings={"quiet": True, "disable_job_creation": True},
    )
    exp_orga_wandb[f"{conf['hash']}_test"] = run_test.id
