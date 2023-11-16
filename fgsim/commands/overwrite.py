from pathlib import Path
from shutil import copytree, rmtree

from fgsim.config import conf
from fgsim.monitoring import logger

from .setup import filter_paths


def overwrite_procedure():
    srcpath = Path(conf.path.run_path) / "fgsim"
    assert srcpath.is_dir()
    rmtree(srcpath)
    logger.info(f"Removed {srcpath}")

    copytree(
        "fgsim",
        srcpath,
        ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
        dirs_exist_ok=True,
    )
    logger.info(f"Wrote {srcpath}")
