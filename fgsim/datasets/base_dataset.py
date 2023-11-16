from pathlib import Path

import torch
from tqdm import tqdm

from fgsim.config import conf
from fgsim.io.chunks import ChunkManager
from fgsim.monitoring import logger


class BaseDS:
    def __init__(self, file_manager):
        self.chunk_manager = ChunkManager(file_manager)

    def _provide_batches(self, dsname):
        attr_name = f"_{dsname}_batches"
        pickle_path = (
            Path(conf.path.dataset)
            / f"pkl_{conf.dataset_name}_{conf.loader_hash}"
            / f"{dsname}.pt"
        )
        chunks = getattr(self.chunk_manager, f"{dsname}_chunks")
        return self._provide_batches_args(attr_name, pickle_path, chunks)

    def _provide_batches_args(self, attr_name, pickle_path, chunks):
        if not hasattr(self, attr_name):
            logger.debug(f"{attr_name} batches not loaded")
            if pickle_path.is_file():
                batch_list = torch.load(
                    pickle_path, map_location=torch.device("cpu")
                )
            else:
                logger.info(f"Processing {attr_name} for {pickle_path}")
                batch_list = self.__process_ds(chunks)
                torch.save(batch_list, pickle_path)
            setattr(self, attr_name, batch_list)
        return getattr(self, attr_name)

    @property
    def training_batches(self):
        return self._provide_batches("training")

    @property
    def validation_batches(self):
        return self._provide_batches("validation")

    @property
    def testing_batches(self):
        return self._provide_batches("testing")

    @property
    def eval_batches(self):
        batch_list = self.validation_batches + self.testing_batches
        if conf.loader.eval_glob is None:
            return batch_list
        rest_eval_path = (
            Path(conf.path.dataset)
            / f"pkl_{conf.dataset_name}_{conf.loader_hash}"
            / "rest_eval.pt"
        )

        rest_eval_chunks = (
            set(self.chunk_manager.eval_chunks)
            - set(self.chunk_manager.testing_chunks)
            - set(self.chunk_manager.validation_chunks)
        )
        batch_list += self._provide_batches_args(
            "_rest_eval_batches",
            rest_eval_path,
            rest_eval_chunks,
        )
        return batch_list

    def __process_ds(self, chunks_list):
        batch_list = [
            self._chunk_to_batch(e) for e in tqdm(chunks_list, postfix="Batches")
        ]
        # batch_list = []
        # with Pool(2) as p:
        #     with tqdm(total=len(chunks_list)) as pbar:
        #         # for b in p.imap_unordered(self._chunk_to_batch, chunks_list):
        #         for b in (self._chunk_to_batch(e) for e in chunks_list):
        #             batch_list.append(b.clone())
        #             pbar.update()

        return batch_list

    def _chunk_to_batch(self, chunks):
        raise NotImplementedError

    def queue_epoch(self, n_skip_events=0) -> None:
        pass

    def __iter__(self):
        return iter(self.training_batches)
