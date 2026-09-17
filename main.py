import gc

import hydra
from omegaconf import DictConfig

from enlight.runner import EnlightRunner


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    EnlightRunner(cfg).run()
    # --multirun runs every job in this same process; free this job's model
    # before the next one starts, or their memory adds up.
    gc.collect()


if __name__ == "__main__":
    main()
