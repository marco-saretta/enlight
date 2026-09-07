import logging

import hydra
from omegaconf import DictConfig

from enlight.runner import EnlightRunner

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Starting ENLIGHT")

    runner = EnlightRunner(cfg)
    runner.run()

    log.info("Run completed.")


if __name__ == "__main__":
    main()
