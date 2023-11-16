"""
Provides the `QueuedDataLoader` class. The definded sequence of qf steps is \
loaded depending on `conf.loader.name`.
"""

from pathlib import Path

import numpy as np
import queueflow as qf
import torch

from fgsim.config import conf
from fgsim.io import LoaderInfo
from fgsim.io.chunks import compute_chucks
from fgsim.io.preprocessed_seq import preprocessed_seq

# from fgsim.io.sel_loader import (
#     DataSetType,
#     files,
#     len_dict,
#     shared_postprocess_switch,
#     process_seq,
# )
from fgsim.monitoring import logger

chunk_size = conf.loader.chunk_size
batch_size = conf.loader.batch_size


class QueuedDataset:
    """
`QueuedDataset` makes `validation_batches` \
and `testing_batches` available as properties; to load training batches, one \
must queue an epoch via `queue_epoch()` and iterate over the instance of the class.
    """

    def __init__(self, loader: LoaderInfo):
        files = loader.file_manager.files
        len_dict = loader.file_manager.file_len_dict
        # Get access to the postprocess switch for computing the validation dataset
        self.shared_postprocess_switch = loader.shared_postprocess_switch
        self.shared_batch_size = loader.shared_batch_size
        process_seq = loader.process_seq

        self.__assign_chunks(files, len_dict)

        self.n_training_events = conf.loader.chunk_size * len(self.training_chunks)
        self.n_grad_steps_per_epoch = self.n_training_events // batch_size

        self.qfseq: qf.Sequence
        if conf.loader.preprocess_training and conf.command not in [
            "preprocess",
            "generate",
        ]:
            qf.init(False)
            self.qfseq = qf.Sequence(*preprocessed_seq())
        else:
            qf.init(False)
            self.qfseq = qf.Sequence(*process_seq())

        if conf.command != "preprocess":
            # In all cases training and test set must be available
            # if the current command is not  preprocessing
            # if (
            #     not os.path.isfile(conf.path.validation)
            #     or not os.path.isfile(conf.path.test)
            # ):
            #     raise FileNotFoundError
            if conf.loader.preprocess_training:
                self.preprocessed_files = list(
                    sorted(Path(conf.path.training).glob(conf.path.training_glob))
                )
                if len(self.preprocessed_files) == 0:
                    raise FileNotFoundError("Couldn't find preprocessed dataset.")

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

    @property
    def validation_batches(self):
        if not hasattr(self, "_validation_batches"):
            logger.debug("Validation batches not loaded, loading from disk.")
            self._validation_batches = torch.load(
                conf.path.validation, map_location=torch.device("cpu")
            )
            logger.debug(
                f"Finished loading. Type is {type(self._validation_batches)}"
            )
        return self._validation_batches

    @property
    def testing_batches(self):
        if not hasattr(self, "_testing_batches"):
            logger.debug("Testing batches not loaded, loading from disk.")
            self._testing_batches = torch.load(
                conf.path.test, map_location=torch.device("cpu")
            )
            logger.debug("Finished loading.")
        return self._testing_batches

    @property
    def eval_batches(self):
        batch_list = self.validation_batches + self.testing_batches
        if conf.loader.eval_glob is None:
            return batch_list
        rest_eval_path = Path(conf.path.dataset_processed) / "rest_eval.pt"
        if not hasattr(self, "_rest_eval_batches"):
            logger.debug("Remainder eval batches not loaded, loading from disk.")
            self._rest_eval_batches = torch.load(
                rest_eval_path, map_location=torch.device("cpu")
            )
            logger.debug("Finished loading.")
        batch_list += self._rest_eval_batches
        return batch_list

    def queue_epoch(self, n_skip_events=0) -> None:
        if not self.qfseq.started:
            self.qfseq.start()

        # Calculate the epoch skip
        n_skip_epochs = n_skip_events // self.n_training_events
        n_skip_events = n_skip_events % self.n_training_events

        # Compute the batches on the fly
        if not conf.loader.preprocess_training or conf.command == "preprocess":
            # Repeat the shuffeling to get the same list
            for _ in range(n_skip_epochs):
                np.random.shuffle(self.training_chunks)
            # Calculate the chunk skip
            n_skip_chunks = (n_skip_events // conf.loader.chunk_size) % len(
                self.training_chunks
            )
            n_skip_events = n_skip_events % conf.loader.chunk_size

            if n_skip_chunks != 0:
                logger.warning(f"Skipped {n_skip_chunks} chunks")
            # Only queue to the chucks that are still left
            self.qfseq.queue_iterable(self.training_chunks[n_skip_chunks:])
            np.random.shuffle(self.training_chunks)

        # Load the preprocessed batches
        else:
            # Repeat the shuffeling to get the same list
            for _ in range(n_skip_epochs):
                np.random.shuffle(self.preprocessed_files)
            # # Calculate the preprocessed file skip
            # n_skip_files = (
            #     n_skip_events
            #     // conf.loader.events_per_file  # by the number of batches per file
            # )
            # n_skip_events = n_skip_events % conf.loader.events_per_file
            # if n_skip_files != 0:
            #     logger.warning(f"Skipped {n_skip_files} pickled files")
            # self.qfseq.queue_iterable(self.preprocessed_files[n_skip_files:])
            self.qfseq.queue_iterable(self.preprocessed_files)
            np.random.shuffle(self.preprocessed_files)

        # Now calculate the number of batches that we still have to skip,
        # because a chunk may be multiple batches and we need to skip
        # the ones that are alread processed
        n_skip_batches = n_skip_events // conf.loader.batch_size

        if n_skip_batches > 0:
            total_batches = self.n_training_events // conf.loader.batch_size
            logger.warning(f"Skipping {n_skip_batches}(/{total_batches}) batches.")
        for ibatch in range(n_skip_batches):
            _ = next(self.qfseq)
            logger.debug(f"Skipped batch({ibatch}).")

    def __iter__(self) -> qf.Sequence:
        return iter(self.qfseq)
