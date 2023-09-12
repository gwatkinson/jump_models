from typing import Optional

import hydra
import pyrootutils
from omegaconf import DictConfig

from src import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> Optional[float]:
    cfg.extras.print_eval = True

    # overwrite task name so debugging logs are stored in separate folder
    cfg.task_name = "print_cfg"

    # disable callbacks and loggers during debugging
    cfg.callbacks = None
    cfg.logger = None

    utils.extras(cfg)


if __name__ == "__main__":
    main()
