import hydra
from omegaconf import DictConfig
from enlight.runner import EnlightRunner


@hydra.main(config_path="configs", config_name="default_config", version_base=None)
def main(config: DictConfig) -> None:
    runner = EnlightRunner(config=config)
    runner.run(dry_run=False)


if __name__ == "__main__":
    main()

# uv run main.py --run simulations=sim_1
# uv run main.py --multirun simulations=_template,sim_1
