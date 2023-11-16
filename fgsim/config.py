import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from fgsim.cli import get_args
from fgsim.utils.oc_resolvers import register_resolvers
from fgsim.utils.oc_utils import dict_to_keylist, gethash, removekeys

register_resolvers()


def compute_conf(default, *confs):
    default = default.copy()

    # Assert, that only keys existing in the default are overwritten
    default_key_set = set(dict_to_keylist(default))
    for c in confs:
        c_key_set = set(dict_to_keylist(c)) - {
            "/hash",
            "/debug",
            "/command",
            "/work_dir",
        }
        if not c_key_set.issubset(default_key_set):
            raise Exception(
                "Key not present in the default config."
                f" Difference:\n{c_key_set - default_key_set}"
            )

    conf = OmegaConf.merge(*(default, *confs))

    # remove the dependency on hash and loader hash to be able to resolve
    conf_without_paths = removekeys(
        conf,
        [
            "path",
        ],
    )
    OmegaConf.resolve(conf_without_paths)

    # Compute a loader_hash
    # this hash will be part of where the preprocessed
    # dataset is safed to ensure the parameters dont change
    # Exclude the keys that do not affect the training
    exclude_keys = ["preprocess_training", "debug"] + [
        x for x in conf["loader"] if "n_workers" in x
    ]

    loader_params = removekeys(conf_without_paths["loader"], exclude_keys)
    conf["loader_hash"] = gethash(loader_params)

    hyperparameters = removekeys(
        conf_without_paths,
        [
            "command",
            "debug",
            "loglevel",
            "loglevel_qf",
            "remote",
            "path",
            "project_name",
            "ray",
            "hash",
        ]
        + [key for key in conf.keys() if key.endswith("_options")],
    )

    conf["hash"] = gethash(hyperparameters)
    # Infer the parameters here
    OmegaConf.resolve(conf)
    for k in conf.path:
        conf.path[k] = str(Path(conf.path[k]).expanduser())
    # remove the options:
    for key in list(conf.keys()):
        if key.endswith("_options"):
            del conf[key]
    return conf, hyperparameters


defaultconf = OmegaConf.load(Path("fgsim/default.yaml").expanduser())
# Load the default settings, overwrite them
# witht the tag-specific settings and then
# overwrite those with cli arguments.
conf: DictConfig = defaultconf.copy()
hyperparameters: DictConfig({})


def parse_arg_conf(args=None):
    if args is None:
        args = get_args()
    if args.hash is not None:
        if args.ray:
            globstr = f"{args.work_dir}/ray/*/{args.hash}/"
        else:
            globstr = f"{args.work_dir}/*/{args.hash}/"
        globstr = str(Path(globstr).expanduser())
        try:
            folder = Path(glob(globstr)[0])
            assert folder.is_dir()
        except IndexError:
            raise IndexError(
                f"No experiement with hash {args.hash} is set up in {globstr}."
            )
        conf = OmegaConf.load(folder / "conf.yaml")
        hyperparameters = OmegaConf.load(folder / "hyperparameters.yaml")

        conf["command"] = str(args.command)

    else:
        fn = f"{args.work_dir}/{args.tag}/conf.yaml"
        if os.path.isfile(fn):
            tagconf = OmegaConf.load(fn)
        else:
            if args.tag == "default":
                tagconf = OmegaConf.create({})
            else:
                raise FileNotFoundError(f"Tag {args.tag} has no conf.yaml file.")
        conf, hyperparameters = compute_conf(defaultconf, tagconf, vars(args))

    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    random.seed(conf.seed)

    # replace workdir
    if conf.path.run_path.startswith("$WD"):
        conf.path.run_path = conf.path.run_path.replace("$WD", str(args.work_dir))

    return conf, hyperparameters


def get_device():
    # Select the CPU/GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.device_count() - 1))
    else:
        device = torch.device("cpu")
    return device


device = get_device()

plt.rcParams.update(
    {
        "savefig.bbox": "tight",
        "figure.dpi": 150,
        "font.family": "Libertinus Sans",
        "backend": "pgf",
        # "backend": "Agg",
        # "text.usetex": True,  # use inline math for ticks
        # "text.latex.preamble": r"\usepackage{libertinus}",
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                # r"\usepackage[T1]{fontenc}",
                # r"\usepackage[utf8]{inputenc}",
                # r"\usepackage{libertine}",
                r"\AtBeginDocument{\catcode`\&=12\catcode`\#=12}",
                r"\usepackage{unicode-math}",
                r"\setmathfont{Libertinus Math}",
                r"\setmathrm{Libertinus Serif}",
                r"\usepackage{fontspec}",
                r"\setmainfont{Libertinus Sans}",
                r"\setsansfont{Libertinus Sans}",
            ]
        ),
    }
)
np.set_printoptions(formatter={"float_kind": "{:.3g}".format})
