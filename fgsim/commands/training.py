"""Contain the training procedure, access point from __main__"""

import signal
import sys
import traceback

import wandb
from fgsim.config import conf, device
from fgsim.ml import Holder, Trainer
from fgsim.monitoring import logger
from fgsim.utils.senderror import send_error, send_exit


def training_procedure() -> None:
    holder = Holder(device)
    trainer = Trainer(holder)
    term_handler = SigTermHander(holder, trainer.loader)
    # Regular run
    if sys.gettrace() is not None or conf.debug:
        trainer.training_loop()
    # Debugger is running
    else:
        exitcode = 0
        try:
            trainer.training_loop()
            send_exit()
        except Exception:
            exitcode = 1
            tb = traceback.format_exc()
            send_error(tb)
            logger.error(tb)
        finally:
            logger.error("Error detected, stopping qfseq.")
            if hasattr(trainer.loader, "qfseq") and trainer.loader.qfseq.started:
                trainer.loader.qfseq.stop()
            exit(exitcode)
    del term_handler


class SigTermHander:
    def __init__(self, holder: Holder, loader) -> None:
        self.holder = holder
        self.loader = loader
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, _signo, _stack_frame):
        self.holder.checkpoint_manager.save_checkpoint()
        if hasattr(self.loader, "qfseq"):
            self.loader.qfseq.stop()
        if not conf.debug:
            wandb.mark_preempting()
        exit()
