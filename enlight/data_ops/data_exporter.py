from pathlib import Path
from logging import Logger
from dataclasses import dataclass
from enlight.model import EnlightModel
import pandas as pd

# At first a lot of the functions from utils.py should be moved here.

@dataclass
class DataExporter:
    """
    This class exports the results of the DA market clearing to dataframes.
    The purpose of the class is to make the results accessible without having
    to run the model each time around.
    It only requires the an instance of the EnlightModel class after calling
    .run_single_simulation() or .run_all_simulations().
    """

    enlight_model_obj: EnlightModel
    logger: Logger
    output_path: Path
    overwrite_exported_data: bool = True  # we don't have to keep this