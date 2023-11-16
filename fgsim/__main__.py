"""Main module."""
import importlib
import os
import sys
from pathlib import Path

import pretty_errors  # noqa

# Add the project to the path, -> `import fgsim.x`
sys.path.append(os.path.dirname(os.path.realpath(".")))

from typeguard import install_import_hook  # noqa

install_import_hook("fgsim")


def main():
    from fgsim.cli import get_args

    args = get_args()

    import fgsim.config

    # Overwrite the config with the one from tag or hash
    (
        fgsim.config.conf,
        fgsim.config.hyperparameters,
    ) = fgsim.config.parse_arg_conf()

    # unload utils
    import sys

    del sys.modules["fgsim.utils"]
    # IFF started with hash and not debug:
    # Move python path to wd/tag/hash/fgsim
    # for persisetent models
    overwrite_path_bool = (
        args.command not in ["gethash", "setup", "dump", "overwrite"]
        and not args.debug
        and not args.ray
    )
    if overwrite_path_bool:
        old_path, new_fgsim_path = overwrite_path()

    # from importlib import reload
    # reload(fgsim.utils)

    # Logger setup
    if args.command not in ["gethash", "setup", "dump", "overwrite"]:
        from fgsim.config import conf
        from fgsim.monitoring.logger import init_logger, logger

        init_logger()
        if overwrite_path_bool:
            logger.warning(f"Replaced path {old_path} with {new_fgsim_path}.")

        logger.info(
            f"tag: {conf.tag} hash: {conf.hash} loader_hash: {conf.loader_hash}"
        )
        logger.info(f"Running command {args.command}")

    # Select command
    match args.command:
        case "setup":
            if args.hash is not None:
                raise Exception
            from fgsim.commands.setup import setup_procedure

            print(setup_procedure())

        case "dump":
            if args.hash is None:
                raise Exception
            from fgsim.commands.dump import dump_procedure

            dump_procedure()

        case "train":
            from fgsim.commands.training import training_procedure

            training_procedure()

        case "test":
            from fgsim.commands.testing import test_procedure

            test_procedure()
        case "generate":
            from fgsim.commands.generate import generate_procedure

            generate_procedure()

        case "overwrite":
            from fgsim.commands.overwrite import overwrite_procedure

            overwrite_procedure()

        case "implant_checkpoint":
            import fgsim.commands.implant_checkpoint

        case "loadfile":
            file_name = str(conf.file_to_load)
            import re

            file_name = re.sub(".*fgsim/(.*?).py", ".\\1", file_name)
            file_name = re.sub("/", ".", file_name)
            importlib.import_module(file_name, "fgsim")

        case _:
            raise Exception


def overwrite_path():
    from fgsim.config import conf

    new_fgsim_path = str((Path(conf.path.run_path)).absolute())
    if not (Path(conf.path.run_path) / "fgsim").is_dir():
        raise Exception("setup has not been executed")
    del conf
    del sys.modules["fgsim"]
    #
    pathlist = [e for e in sys.path if e.endswith("fgsim")]
    # make sure that this is unique
    if len({e for e in pathlist}) == 0:
        old_path = ""
    elif len({e for e in pathlist}) == 1:
        # remove the old path
        old_path = pathlist[0]
        for path in pathlist:
            sys.path.remove(path)
    elif len({e for e in pathlist}) > 1:
        raise Exception
    sys.path.insert(0, new_fgsim_path)

    return old_path, new_fgsim_path


if __name__ == "__main__":
    main()
