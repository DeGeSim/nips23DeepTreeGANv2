from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from fgsim.config import conf


# Make sure the readpath takes a path and start and end of the chunk
# It loads a list of files, and then loads the lengths of those files
class FileManager:
    def __init__(
        self, path_to_len: Callable[[Path], int], files: Optional[List[Path]] = None
    ) -> None:
        path_dataset_processed = (
            Path(conf.path.dataset) / f"pkl_{conf.dataset_name}_{conf.loader_hash}"
        )
        if not path_dataset_processed.is_dir():
            path_dataset_processed.mkdir()
        self.path_ds_lenghts = path_dataset_processed / "filelengths.yaml"

        self._path_to_len = path_to_len
        self.files = files
        if self.files is None:
            self.files: List[Path] = self._get_file_list()
        self.file_len_dict: Dict[Path, int] = self._load_len_dict()

    def _get_file_list(self) -> List[Path]:
        ds_path = Path(conf.path.dataset).expanduser()
        assert ds_path.is_dir()
        files = sorted(ds_path.glob(conf.loader.dataset_glob))
        if len(files) < 1:
            raise RuntimeError("No datasets found")
        return [f for f in files]

    def _load_len_dict(self) -> Dict[Path, int]:
        if not self.path_ds_lenghts.is_file():
            self.save_len_dict()
        with open(self.path_ds_lenghts, "r") as f:
            len_dict: Dict[Path, int] = {
                Path(k): int(v)
                for k, v in yaml.load(f, Loader=yaml.SafeLoader).items()
            }
        return len_dict

    def save_len_dict(self) -> None:
        self.len_dict = {}
        for fn in self.files:
            self.len_dict[str(fn.absolute())] = self._path_to_len(fn)
        with open(self.path_ds_lenghts, "w") as f:
            yaml.dump(self.len_dict, f, Dumper=yaml.SafeDumper)
