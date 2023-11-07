import logging

import colorlog

LOG_LEVEL = logging.INFO
LOGFORMAT = (
    "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
)


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(LOGFORMAT)
    handler.setFormatter(formatter)
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    # for level in logging_levels:
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
