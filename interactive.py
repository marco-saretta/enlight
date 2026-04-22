# %%
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from enlight.runner import EnlightRunner

GlobalHydra.instance().clear()

with initialize(config_path="configs", version_base=None):
    cfg = compose(config_name="default_config", overrides=["simulations=sim_1"])

# %%
runner = EnlightRunner(config=cfg)
runner.run(dry_run=False)
# %%
