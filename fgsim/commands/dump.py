import os
import shutil
from glob import glob

import wandb
from fgsim.cli import get_args
from fgsim.monitoring.experiment_organizer import exp_orga_wandb


def dump_procedure():
    args = get_args()

    # wandb
    from fgsim.config import conf

    api = wandb.Api()
    for s in [conf.hash, f"{conf.hash}_train", f"{conf.hash}_test"]:
        if s in exp_orga_wandb:
            try:
                run = api.from_path(f"{conf.project_name}/runs/{exp_orga_wandb[s]}")
                run.delete()
            except wandb.errors.CommError:
                pass

            del exp_orga_wandb[s]

    # local
    paths = glob(f"{args.work_dir}/*/{args.hash}")
    if len(paths) == 1:
        assert os.path.isdir(paths[0])
        shutil.rmtree(paths[0])
    elif len(paths) == 0:
        print("No directory found!")
    else:
        print("No directory found!")
        raise Exception
