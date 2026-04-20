import hydra
from omegaconf import DictConfig
from enlight.runner import EnlightRunner
from pathlib import Path


@hydra.main(config_path="configs", config_name="default_config")
def main(cfg: DictConfig):

    # Create an instance of the EnlightRunner
    enlight_runner = EnlightRunner(config=cfg)

    # Creates instance of the DataProcessor:
    enlight_runner.run_scenario("scenario_1", dry_run=False)


if __name__ == "__main__":
    main()
