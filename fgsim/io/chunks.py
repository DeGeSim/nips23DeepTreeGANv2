from pathlib import Path
from typing import List, Tuple

from fgsim.config import conf
from fgsim.monitoring import logger

ChunkType = Tuple[Tuple[Path, int, int]]
chunk_size = conf.loader.batch_size
batch_size = conf.loader.batch_size


def compute_chucks(files, len_dict) -> List[ChunkType]:
    chunk_coords: list[list] = [[]]
    ifile = 0
    ielement = 0
    current_chunck_elements = 0
    while ifile < len(files):
        elem_left_in_cur_file = len_dict[files[ifile]] - ielement
        elem_to_add = chunk_size - current_chunck_elements
        if elem_left_in_cur_file > elem_to_add:
            chunk_coords[-1].append(
                (files[ifile], ielement, ielement + elem_to_add)
            )
            ielement += elem_to_add
            current_chunck_elements += elem_to_add
        else:
            chunk_coords[-1].append(
                (files[ifile], ielement, ielement + elem_left_in_cur_file)
            )
            ielement = 0
            current_chunck_elements += elem_left_in_cur_file
            ifile += 1
        if current_chunck_elements == chunk_size:
            current_chunck_elements = 0
            chunk_coords.append([])

    # remove the last, uneven chunk
    chunk_coords = list(
        filter(
            lambda chunk: sum([part[2] - part[1] for part in chunk]) == chunk_size,
            chunk_coords,
        )
    )
    # convert to list of tuple to make chunks hashable
    return [tuple(e) for e in chunk_coords]


class ChunkManager:
    def __init__(self, file_manager) -> None:
        files = file_manager.files
        len_dict = file_manager.file_len_dict
        # Get access to the postprocess switch for computing the validation dataset
        self.__assign_chunks(files, len_dict)

        self.n_training_events = conf.loader.chunk_size * len(self.training_chunks)
        self.n_grad_steps_per_epoch = self.n_training_events // batch_size

    def __assign_chunks(self, files, len_dict):
        # Make sure the chunks can be split evenly into batches:
        assert chunk_size % batch_size == 0

        assert conf.loader.validation_set_size % batch_size == 0
        n_validation_batches = conf.loader.validation_set_size // batch_size
        n_validation_chunks = conf.loader.validation_set_size // chunk_size

        assert conf.loader.test_set_size % chunk_size == 0
        self.n_test_batches = conf.loader.test_set_size // batch_size
        n_testing_chunks = conf.loader.test_set_size // chunk_size

        logger.info(
            f"Using the first {n_validation_batches} batches for "
            + f"validation and the next {self.n_test_batches} batches for testing."
        )

        if conf.loader.eval_glob is None:
            chunk_coords = compute_chucks(files, len_dict)
            self.validation_chunks = chunk_coords[:n_validation_chunks]
            self.testing_chunks = chunk_coords[
                n_validation_chunks : n_validation_chunks + n_testing_chunks
            ]
            self.training_chunks = chunk_coords[
                n_validation_chunks + n_testing_chunks :
            ]
        else:
            import re

            eval_files = [
                e for e in files if re.search(conf.loader.eval_glob, str(e))
            ]
            train_files = [e for e in files if e not in eval_files]
            self.training_chunks = compute_chucks(train_files, len_dict)

            self.eval_chunks = compute_chucks(eval_files, len_dict)
            self.validation_chunks = self.eval_chunks[:n_validation_chunks]
            self.testing_chunks = self.eval_chunks[
                n_validation_chunks : n_validation_chunks + n_testing_chunks
            ]
        assert len(self.testing_chunks) == n_testing_chunks
        assert len(self.validation_chunks) == n_validation_chunks

        # Check that there is a reasonable amount of data
        assert len(self.validation_chunks) + len(self.testing_chunks) < len(
            self.training_chunks
        ), "Dataset to small"
