import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler
from tqdm.contrib.logging import logging_redirect_tqdm

from fgsim.config import conf

logger = logging.getLogger("fgsim")
logger.setLevel(logging.DEBUG)


def init_logger():
    if not logger.handlers:
        log_path = f"{conf.path.run_path}/{conf.command}.log"

        format = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%y-%m-%d %H:%M",
        )
        format_plain = logging.Formatter(
            fmt="%(message)s",
            datefmt="%y-%m-%d %H:%M",
        )

        stream_handler = RichHandler()
        stream_handler.setFormatter(format_plain)
        stream_handler.setLevel(logging.DEBUG if conf.debug else conf.loglevel)
        logger.addHandler(stream_handler)

        if not conf.debug:
            file_handler = RotatingFileHandler(filename=log_path, backupCount=10)
            file_handler.setFormatter(format)
            file_handler.setLevel(logging.INFO)
            file_handler.doRollover()
            logger.addHandler(file_handler)

        # qf logger
        # qf_stream_handler = RichHandler()
        # qf_stream_handler.setFormatter(format_plain)
        # qf_stream_handler.setLevel(conf.loglevel_qf)
        # if not conf.debug:
        #     loader_log = f"{conf.path.run_path}/{conf.command}_loader.log"
        #     qf_file_handler = RotatingFileHandler(
        #         filename=loader_log, backupCount=10
        #     )
        #     qf_file_handler.setFormatter(format)
        #     qf_file_handler.setLevel(min(logging.INFO, conf.loglevel_qf))

        logging_redirect_tqdm([logger])
