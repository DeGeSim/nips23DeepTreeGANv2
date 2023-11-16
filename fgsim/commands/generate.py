"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from pathlib import Path

import h5py
import torch
from caloutils.processing import pc_to_voxel
from tqdm import tqdm

from fgsim.cli import get_args
from fgsim.config import conf, device
from fgsim.datasets import Dataset
from fgsim.ml.eval import postprocess
from fgsim.ml.holder import Holder


def generate_procedure() -> None:
    ds = Dataset()
    args = get_args()

    batch_size = (
        args.batch_size if args.batch_size is not None else conf.loader.batch_size
    )
    conf.loader.batch_size = batch_size

    # init holder once batch_size is overwritten
    holder: Holder = Holder(device)

    outdir = args.output_dir if args.output_dir is not None else conf.path.run_path
    outdir = Path(outdir).expanduser().absolute()
    outdir.mkdir(exist_ok=True, parents=True)

    for best_or_last in ["best"]:
        if best_or_last == "best":
            holder.checkpoint_manager.select_best_model()

        dspath = Path(outdir).absolute() / f"out_{best_or_last}.hdf5"
        if not dspath.exists():
            __write_dataset(holder, ds, dspath)
        else:
            with h5py.File(dspath, "r") as ds:
                if "hash" not in ds.attrs or "grad_step" not in ds.attrs:
                    __write_dataset(holder, ds, dspath)
                assert ds.attrs["hash"] == conf.hash
                if ds.attrs["grad_step"] != holder.state["grad_step"]:
                    __write_dataset(holder, ds, dspath)

        ## compute the aucs
        # resd = __run_classifiers(dspath)

        # holder.train_log.log_metrics(
        #     resd,
        #     prefix="/".join(["test", best_or_last]),
        #     step=holder.state["grad_step"],
        #     epoch=holder.state["epoch"],
        # )
        # holder.train_log.wandb_run.summary.update(
        #     {"/".join(["m", "test", best_or_last, k]): v for k, v in resd.items()}
        # )
        # logger.info(resd)
        # holder.train_log.flush()
    exit(0)


# from subprocess import PIPE, CalledProcessError, Popen
# def __run_classifiers(dspath):
#     resd = {}

#     rpath = Path(conf.path.run_path).absolute()
#     outdir = rpath / "cc_eval/"
#     test_path = "/home/mscham/fgsim/data/calochallange2/dataset_2_2.hdf5"
#     for classifer in "cls-high", "cls-low", "cls-low-normed":
#         logger.info(f"Running classifier {classifer}")
#         cmd = (
#             f"/dev/shm/mscham/fgsim/bin/python evaluate.py -i {dspath} -m"
#             f" {classifer} -r {test_path} -d 2"
#             f" --output_dir {outdir}"
#         )

#         lines = []
#         with Popen(
#             cmd.split(" "),
#             stdout=PIPE,
#             bufsize=1,
#             universal_newlines=True,
#             cwd="/home/mscham/homepage/code/",
#         ) as p:
#             for line in p.stdout:
#                 lines.append(line)
#                 # print(line, end="")

#         if p.returncode != 0:
#             raise CalledProcessError(p.returncode, p.args)

#         # aucidx =
#               lines.index("Final result of classifier test (AUC / JSD):\n") + 1
#         auc = float(lines[-1].split("/")[0].rstrip())
#         resd[classifer] = auc
#         logger.info(f"Classifier {classifer} AUC {auc}")
#     return resd


def __write_dataset(holder, ds, dspath):
    batch_size = conf.loader.batch_size

    y = torch.stack([batch.y for batch in ds.eval_batches])
    n_pointsv = torch.stack([batch.n_pointsv for batch in ds.eval_batches])

    assert (
        len(y) % batch_size == 0
    ), f"Batch size {batch_size} not a multiple of the dataset size {len(y)}"
    y = y.reshape(-1, batch_size, y.shape[-1])
    n_pointsv = n_pointsv.reshape(-1, batch_size)

    x_l = []
    E_l = []

    for ye, e_pointsv in tqdm(
        zip(y, n_pointsv), miniters=100, mininterval=5.0, total=y.shape[0]
    ):
        cond_gen_features = conf.loader.cond_gen_features

        if sum(cond_gen_features) > 0:
            cond = ye[..., cond_gen_features].clone()
        else:
            cond = torch.empty((batch_size, 0)).float()
        gen_batch = holder.generate(cond.to(device), e_pointsv.to(device))

        gen_batch = postprocess(gen_batch, "gen")

        x_l.append(pc_to_voxel(gen_batch).cpu())
        E_l.append(ye.T[0].clone().cpu())

    your_energies = torch.hstack(E_l)
    your_showers = torch.vstack(x_l)

    if dspath.exists():
        dspath.unlink()
    # dspath.touch()
    with h5py.File(dspath.absolute(), "w") as ds:
        ds.create_dataset(
            "incident_energies",
            data=your_energies.reshape(len(your_energies), -1),
            compression="gzip",
        )
        ds.create_dataset(
            "showers",
            data=your_showers.reshape(len(your_showers), -1),
            compression="gzip",
        )
        ds.attrs["hash"] = conf.hash
        ds.attrs["grad_step"] = holder.state["grad_step"]


def __recur_transpant_dict(gsd, ssd):
    for k, v in ssd.items():
        if isinstance(v, dict):
            if k not in gsd:
                gsd[k] = {}
            __recur_transpant_dict(gsd[k], ssd[k])
        elif isinstance(v, torch.Tensor):
            if k not in gsd:
                gsd[k] = v.clone()
        else:
            raise NotImplementedError(f"No definded behaviour for {type(v)}")
