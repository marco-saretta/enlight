# %%
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from enlight.runner import EnlightRunner

GlobalHydra.instance().clear()

with initialize(config_path="config", version_base=None):
    cfg = compose(config_name="config", overrides=["simulations=sim_1"])

# %%
runner = EnlightRunner(config=cfg)
runner.run(dry_run=False)
# %%
