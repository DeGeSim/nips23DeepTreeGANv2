from pathlib import Path
from typing import List, Tuple

import torch
from jetnet.datasets import JetNet

from fgsim.config import conf
from fgsim.io import FileManager

jn_data = JetNet.getData(
    jet_type=conf.loader.jettype,
    data_dir=Path(conf.loader.dataset_path).expanduser(),
    num_particles=conf.loader.n_points,
)


def path_to_len(fn: Path) -> int:
    return jn_data[0].shape[0]


file_manager = FileManager(path_to_len, files=[Path(conf.loader.jettype)])


def readpath(fn: Path, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    particle_data, jet_data = jn_data
    res = (
        torch.tensor(jet_data[start:end], dtype=torch.float),
        torch.tensor(particle_data[start:end], dtype=torch.float),
    )
    return res


def read_chunks(
    chunks: List[Tuple[Path, int, int]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    chunks_list = []
    for chunk in chunks:
        chunks_list.append(readpath(*chunk))
    res = (
        torch.concat([e[0] for e in chunks_list]),
        torch.concat([e[1] for e in chunks_list]),
    )
    return [(res[0][ievent], res[1][ievent]) for ievent in range(len(res[1]))]
