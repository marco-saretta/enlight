from pathlib import Path
from logging import Logger
from dataclasses import dataclass
from typing import Dict, Any
import enlight.utils as utils
import pandas as pd
import yaml


@dataclass
class DataProcessor:
    """Data processor for energy system scenarios."""

    scenario_name: str
    config_yaml: dict
    logger: Logger
    base_config_path: Path = Path("config")
    base_data_path: Path = Path("data")

    def __post_init__(self) -> None:
        """
        Eagerly load and process all required data, including all networks, upon instantiation.

        Note: This method performs all data loading and processing steps immediately when the object is created,
        which may have side effects (such as file I/O) and performance implications.
        """
        # Initialize logger
        self.logger.info(f"INITIALIZING DATA PROCESSOR: {self.scenario_name}")

        # Initialize auxiliary data dictionary
        self.aux_data_dict: Dict[str, Any] = {"scenario_name": self.scenario_name}

        self._init_data_paths()
        self._write_bidding_zones()
        self._write_generators()  # NEWLY ADDED
        self._load_scenarios_config()
        self._load_setup()
        self._prepare_all_renewable_sources()
        self._load_hydro_reservoir_data()
        self._load_hydro_pumped_storage()
        self._load_conventional_thermal_units_data()
        self._load_ptx_plants()
        self._load_dh_plants()
        self._load_fuel_prices()
        self._load_transmission_lines_data()
        self._prepare_inflexible_demand_sources()
        # Save aux data to YAML after all processing
        self._save_aux_data_to_yaml()

    def _init_data_paths(self) -> None:
        """Initialize all data directory paths according to the updated folder structure."""
        # Demand paths
        self.path_demand_inflex_classic = (
            self.base_data_path / "demand_inflexible_classic"
        )
        self.path_demand_flex_classic = self.base_data_path / "demand_flexible_classic"
        self.path_demand_inflex_ev = self.base_data_path / "demand_inflexible_ev"
        self.path_demand_flex_ev = self.base_data_path / "demand_flexible_ev"

        # Market data paths
        self.path_fuel_projections = self.base_data_path / "fuel_price_projections"

        # Hydro paths
        # self.path_hydro_ror = self.base_data_path / "hydro_run_of_river"
        self.path_hydro_reservoir = self.base_data_path / "hydro_reservoir"
        self.path_hydro_pumped = self.base_data_path / "hydro_pumped_storage"

        # Renewable paths
        # self.path_solar_pv = self.base_data_path / "solar_pv"
        # self.path_wind_onshore = self.base_data_path / "wind_onshore"
        # self.path_wind_offshore = self.base_data_path / "wind_offshore"

        # Thermal and other paths
        self.path_thermal_plants = self.base_data_path / "thermal_plants"
        self.path_district_heating = self.base_data_path / "district_heating"
        self.path_ptx = self.base_data_path / "ptx"
        self.path_lines = self.base_data_path / "lines"

        # Common subdirectories
        self.capacity_projections_subdir = "capacity_projections"
        self.weather_years_subdir = "weather_years"
        self.profile_years_subdir = "profile_years"

        self.output_path = (
            Path("simulations") / self.scenario_name / "data"
        )  # Output path for the scenario data

    def _load_scenarios_config(self) -> None:
        """
        Load the system configuration for various scenarios from an Excel file.

        This method reads the 'scenarios_config.xlsx' file located in the base configuration path
        and stores its content in the 'scenario_config_df' attribute for later use.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        # Define the path to the scenarios configuration Excel file
        scenario_config_path = self.base_config_path / "scenarios_config.xlsx"

        # Check if the configuration file exists
        if not scenario_config_path.exists():
            raise FileNotFoundError(
                f"System configuration not found: {scenario_config_path}"
            )

        # Load the configuration file into a DataFrame
        self.scenario_config_df = pd.read_excel(
            scenario_config_path, index_col=0, sheet_name="Python"
        )

    def _write_bidding_zones(self) -> None:
        self.bidding_zones_list = self.config_yaml.get("bidding_zones", [])

        self.aux_data_dict["bidding_zones"] = self.bidding_zones_list

    def _write_generators(self) -> None:
        # The list of conventional generators is not yet useful but included for completeness.
        # self.conventional_generators = self.config_yaml.get("conventional_generators", [])
        self.VRE_generators = self.config_yaml.get("VRE_generators", [])

    def _load_setup(self) -> None:
        """
        Load scenario setup configuration from the loaded configuration DataFrame.

        This method extracts the 'SETUP' section from the 'scenario_config_df' DataFrame,
        sets up the configurations for the scenario, and determines the prediction year.

        Raises:
            ValueError: If the 'SETUP' section is missing in the configuration DataFrame.
        """
        # Label for the setup configuration section
        setup_label = "SETUP"

        # Check if the 'SETUP' section is present in the configuration DataFrame
        if setup_label not in self.scenario_config_df.index:
            raise ValueError("Missing SETUP section in system config.")

        # Extract and copy the setup configuration section from the DataFrame
        self.setup_config_df = self.scenario_config_df.loc[setup_label].copy()

        # Store the setup configuration and determine the prediction year from the scenario name
        self.prediction_year = int(self.setup_config_df[self.scenario_name])  # type: ignore

        self.aux_data_dict["prediction_year"] = self.prediction_year

    def calculate_renewable_profiles(self, config: dict) -> pd.DataFrame:
        """
        Compute renewable generation profiles by combining weather-based per-unit profiles
        with installed capacity projections.

        Args:
            config (dict): Expected keys:
                - 'label': Section label in the scenario config sheet
                - 'data_path': Base data folder for this technology
                - 'wy_key': Config key for weather year
                - 'cap_file_key': Config key for capacity projection file
                - 'bid_price_key': Config key for bid price
                - 'wy_subdir': Subfolder under data_path containing profile files
                - 'wy_label': File prefix for profile files

        Returns:
            pd.DataFrame: Hourly renewable production [MW] for the prediction year,
                        indexed by time, with columns as bidding zones.
        """

        label = config["label"]
        scenario = self.scenario_name

        # --- Extract scenario-specific parameters ---
        section = self.scenario_config_df.loc[label].copy().set_index("key")
        weather_year = section.at[config["wy_key"], scenario]
        cap_file = section.at[config["cap_file_key"], scenario]
        bid_price = float(section.at[config["bid_price_key"], scenario])

        # Store bid price and weather year in aux data (for later use/reporting)

        # Store in aux data
        aux = self.aux_data_dict.setdefault(label.lower(), {})
        aux.update({"bid_prices": bid_price, "weather_year": weather_year})

        # --- Load weather-based per-unit production profile ---
        profile_file = f"{config['wy_label']}_{weather_year}.csv"
        profile_path = config["data_path"] / config["wy_subdir"] / profile_file
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_path}")

        profile_df = pd.read_csv(profile_path, index_col=0)
        utils.validate_df_positive_numeric(profile_df, f"{label.lower()}_profile_pu")

        # --- Load installed capacity projections ---
        cap_path = (
            config["data_path"] / self.capacity_projections_subdir / f"{cap_file}.csv"
        )
        if not cap_path.exists():
            raise FileNotFoundError(f"Capacity file not found: {cap_path}")

        cap_df = pd.read_csv(cap_path, index_col=0)

        # Filter to prediction year
        try:
            cap_year = cap_df.loc[[self.prediction_year]]
        except KeyError:
            raise KeyError(
                f"Prediction year '{self.prediction_year}' not found in {cap_path}"
            )

        utils.validate_df_positive_numeric(cap_year, f"{label.lower()}_cap")

        # --- Compute renewable production ---
        # Initialize output with all bidding zones (set to 0 if no data)
        production_df = pd.DataFrame(
            index=profile_df.index,
            columns=self.bidding_zones_list,
            data=0.0,
        )

        # --- Compute renewable production ---
        # Find columns present in all three: capacity, profile, and output template
        common_cols = cap_year.columns.intersection(profile_df.columns).intersection(
            production_df.columns
        )

        if common_cols.empty:
            raise ValueError(
                f"No matching bidding zone columns between capacity, profile, and production for {label}"
            )

        # Multiply profile (per-unit) by installed capacity (MW)
        production_df[common_cols] = (
            profile_df[common_cols] * cap_year[common_cols].values
        )

        # Add the week column
        production_df["Week"] = profile_df["Week"]

        return production_df

    def _prepare_all_renewable_sources(self) -> None:
        """
        Load, process, and store renewable generation profiles and capacities.
        Covers: Wind Onshore, Wind Offshore, Solar PV, Hydro ROR.
        """
        self.scenario_config_df
        sources_dict = {}  # create a dictionary of dictionaries to dynamically create the sources list used to load the VRE data from the excel-configuration file.
        for vre_gen in self.VRE_generators:
            label, tech = (
                vre_gen.values()
            )  # load the label and tech from the .yaml configuration file e.g. WIND_ON and wind_onshore
            sources_dict[label] = {}
            sources_dict[label]["label"] = label
            sources_dict[label]["data_path"] = self.base_data_path / tech
            sources_dict[label]["wy_subdir"] = self.weather_years_subdir
            sources_dict[label]["wy_label"] = tech + "_wy"
            sources_dict[label]["output_file"] = tech + "_production.csv"
            for subkey in self.scenario_config_df.loc[label, "key"]:
                sources_dict[label][subkey.replace(tech + "_", "") + "_key"] = subkey
        sources = list(sources_dict.values())

        for source in sources:
            prod_df = self.calculate_renewable_profiles(source)
            utils.save_data(
                prod_df,
                source["output_file"],
                output_dir=self.output_path,
                logger=self.logger,
            )

    def _load_hydro_reservoir_data(self) -> None:
        """
        Load and process data for hydro reservoir units.

        This method retrieves the hydro reservoir units data from a specified CSV file,
        checks for its existence, loads it into a DataFrame, validates the data, and saves
        it to the designated output path.

        Raises:
            FileNotFoundError: If the hydro reservoir units file does not exist.
        """

        # Load the configuration section for hydro reservoir
        hydro_res_df = self.scenario_config_df.loc["HYDRO_RES"].copy()
        hydro_res_df.set_index("key", inplace=True)

        # Extract hydro reservoir configuration values
        hydro_res_units_file = hydro_res_df.loc[
            "hydro_res_units_file", self.scenario_name
        ]
        hydro_res_energy_wy = hydro_res_df.loc[
            "hydro_res_energy_wy", self.scenario_name
        ]

        # Define file paths
        hydro_res_units_filepath = (
            self.path_hydro_reservoir / "units" / f"{hydro_res_units_file}.csv"
        )
        hydro_res_energy_wy_filepath = (
            self.path_hydro_reservoir
            / "energy_availability"
            / f"hydro_res_energy_wy_{hydro_res_energy_wy}.csv"
        )

        # Check if the hydro reservoir units file exists
        if not hydro_res_units_filepath.exists():
            raise FileNotFoundError(
                f"Hydro reservoir units file not found: {hydro_res_units_filepath}"
            )

        # Check if the hydro reservoir energy availability file exists
        if not hydro_res_energy_wy_filepath.exists():
            raise FileNotFoundError(
                f"Hydro reservoir energy availability file not found: {hydro_res_energy_wy_filepath}"
            )

        # Load the hydro reservoir data into DataFrames
        self.hydro_reservoir_units_df_raw = pd.read_csv(
            hydro_res_units_filepath, index_col=0
        )
        self.hydro_res_energy_wy_df_raw = pd.read_csv(
            hydro_res_energy_wy_filepath, index_col=0
        )

        # Filter the hydro reservoir data to only include generators and energy availability in the selected bidding zones
        self.hydro_reservoir_units_df = self.hydro_reservoir_units_df_raw[
            self.hydro_reservoir_units_df_raw['zone_el'].isin(self.bidding_zones_list)
            ].copy()  # .copy() used to avoid SettingWithCopyWarning
        self.hydro_res_energy_wy_df = self.hydro_res_energy_wy_df_raw[self.bidding_zones_list].copy()

        # Validate the loaded data
        utils.validate_df_positive_numeric(
            self.hydro_res_energy_wy_df, "hydro_res_energy_availability"
        )

        # Save the loaded and validated hydro reservoir data to the designated output path
        utils.save_data(
            self.hydro_reservoir_units_df,
            "hydro_reservoir_units.csv",
            output_dir=self.output_path,
            logger=self.logger,
        )
        utils.save_data(
            self.hydro_res_energy_wy_df,
            "hydro_reservoir_energy.csv",
            output_dir=self.output_path,
            logger=self.logger,
        )

    def _load_hydro_pumped_storage(self) -> None:
        """
        Load and process data for hydro pumped storage units.

        This method retrieves the hydro pumped storage units data from a specified CSV file,
        checks for its existence, loads it into a DataFrame, and saves it to the designated output path.

        Raises:
            FileNotFoundError: If the hydro pumped storage units file does not exist.
        """
        hydro_ps_filepath = self.path_hydro_pumped / "hydro_pumped_storage_units.csv"

        # Check if the hydro reservoir energy availability file exists
        if not hydro_ps_filepath.exists():
            raise FileNotFoundError(
                f"Hydro pumped storage units file not found: {hydro_ps_filepath}"
            )

        self.hydro_pumped_storage_units_df = pd.read_csv(hydro_ps_filepath, index_col=0)
        utils.save_data(
            self.hydro_pumped_storage_units_df,
            "hydro_pumped_storage_units.csv",
            output_dir=self.output_path,
            logger=self.logger,
        )

    def _load_conventional_thermal_units_data(self) -> None:
        """
        Load data for thermal generation units.

        This method retrieves the thermal plant units data from a specified CSV file,
        checks for its existence, and loads it into a DataFrame. The data is then saved
        to a specified output path.

        Raises:
            FileNotFoundError: If the thermal units file does not exist.
        """

        # Load the configuration section for hydro reservoir
        thermal_df = self.scenario_config_df.loc["THERMAL"].copy()
        thermal_df.set_index("key", inplace=True)

        # Extract thermal units configuration values
        thermal_units_file = thermal_df.loc["thermal_units_file", self.scenario_name]

        # Define the path to the thermal plant units CSV file
        thermal_units_filepath = (
            self.path_thermal_plants / "units" / f"{thermal_units_file}.csv"
        )

        # Check if the thermal units file exists
        if not thermal_units_filepath.exists():
            raise FileNotFoundError(
                f"Thermal units file not found: {thermal_units_filepath}"
            )

        # Load the thermal plant units data into a DataFrame
        self.thermal_units_raw = pd.read_csv(thermal_units_filepath, index_col=0)

        # Filter the thermal units to only include those in the selected bidding zones
        self.thermal_units = self.thermal_units_raw[
            self.thermal_units_raw["zone_el"].isin(self.bidding_zones_list)
        ]

        # Save the loaded thermal plant units data to the designated output path
        utils.save_data(
            self.thermal_units,
            "conventional_thermal_units.csv",
            output_dir=self.output_path,
            logger=self.logger,
        )

    def _load_ptx_plants(self) -> None:
        pass

    def _load_dh_plants(self) -> None:
        pass

    def _load_fuel_prices(self) -> None:
        pass

    def _load_transmission_lines_data(self) -> None:
        """
        Load data for transmission lines.

        This method retrieves the transmission lines data from specified CSV files,
        checks for their existence, and loads them into DataFrames. The data is then saved
        to a specified output path.

        Raises:
            FileNotFoundError: If neither of the transmission lines files exist.
        """

        # Label for the setup configuration section
        lines_label = "LINES"

        # Check if the 'SETUP' section is present in the configuration DataFrame
        if lines_label not in self.scenario_config_df.index:
            raise ValueError("Missing LINES section in system config.")

        # Extract and copy the setup configuration section from the DataFrame
        self.lines_config_df = self.scenario_config_df.loc[lines_label].copy()

        # Store the setup configuration and determine the prediction year from the scenario name
        self.lines_selection = str(self.lines_config_df[self.scenario_name])

        # Define the paths to the transmission lines CSV files
        lines_a_b_file = self.path_lines / self.lines_selection / "lines_a_b.csv"
        lines_b_a_file = self.path_lines / self.lines_selection / "lines_b_a.csv"

        # Check if both transmission lines files exist
        if lines_a_b_file.exists() and lines_b_a_file.exists():
            # self.lines_a_b_raw = pd.read_csv(lines_a_b_file, index_col=0)
            # self.lines_b_a_raw = pd.read_csv(lines_b_a_file, index_col=0)

            # self.lines_a_b = self.lines_a_b_raw[self.lines_a_b_raw.index.isin(self.bidding_zones_list)]
            # self.lines_b_a = self.lines_b_a_raw[self.lines_b_a_raw.index.isin(self.bidding_zones_list)]

            # New code below ensures that the source and destination zone have both been chosen as bidding zones.
            self.lines_a_b_raw = pd.read_csv(lines_a_b_file)  # , index_col=0)
            self.lines_b_a_raw = pd.read_csv(lines_b_a_file)  # , index_col=0)

            self.lines_a_b = self.lines_a_b_raw[
                self.lines_a_b_raw[["from_zone", "to_zone"]]
                .isin(self.bidding_zones_list)
                .all(axis=1)
            ]
            self.lines_b_a = self.lines_b_a_raw[
                self.lines_b_a_raw[["to_zone", "from_zone"]]
                .isin(self.bidding_zones_list)
                .all(axis=1)
            ]

            utils.save_data(
                self.lines_a_b,
                "lines_a_b.csv",
                output_dir=self.output_path,
                logger=self.logger,
            )
            utils.save_data(
                self.lines_b_a,
                "lines_b_a.csv",
                output_dir=self.output_path,
                logger=self.logger,
            )
        else:
            raise FileNotFoundError(
                f"Line files not found: {lines_a_b_file} or {lines_b_a_file}"
            )

    def calculate_inflexible_demand(self, config: dict) -> pd.DataFrame:
        """
        Load and scale inflexible electricity demand using profile and projection.

        Args:
            config (dict): Expected keys:
                - 'label': Section label in the config sheet
                - 'profile_year_key': Config key for profile year
                - 'amount_file_key': Config key for total demand projection
                - 'voll_key': Config key for VOLL (Value of Lost Load)
                - 'aux_label': Key under which VOLL is stored in aux_data_dict
                - 'base_path': Base folder for profile and projection files

        Returns:
            pd.DataFrame: Scaled demand profile (MW) for prediction year
        """
        scenario = self.scenario_name
        section = self.scenario_config_df.loc[config["label"]].copy().set_index("key")

        # --- Extract configuration values ---
        profile_year = section.at[config["profile_year_key"], scenario]
        amount_file = section.at[config["amount_file_key"], scenario]
        voll = float(section.at[config["voll_key"], scenario])

        # Store VOLL in auxiliary data
        self.aux_data_dict.setdefault(config["aux_label"], {})["voll"] = voll

        # --- Build file paths ---
        profile_file = f"{config['profile_year_key']}_{profile_year}.csv"
        profile_path = config["base_path"] / self.profile_years_subdir / profile_file
        projection_path = (
            config["base_path"] / "demand_projection" / f"{amount_file}.csv"
        )

        # --- Load data ---
        profile_df = utils.load_csv_if_exists(profile_path)
        projection_df = utils.load_csv_if_exists(projection_path)

        # --- Extract projection for prediction year ---
        if self.prediction_year not in projection_df.index:
            raise KeyError(
                f"Prediction year '{self.prediction_year}' not found in projection file: {projection_path}"
            )
        projection_row = projection_df.loc[self.prediction_year]

        # --- Compute demand by scaling profile with projection ---
        common_cols = profile_df.columns.intersection(projection_row.index)
        if common_cols.empty:
            raise ValueError(
                f"No matching columns between profile and projection for {config['label']}"
            )

        demand_profile = profile_df.copy()
        demand_profile[common_cols] = (
            profile_df[common_cols] * projection_row[common_cols].values
        )

        # --- Validation ---
        utils.validate_df_positive_numeric(demand_profile, profile_file)

        # Filter the bidding zones chosen in config.yaml
        demand_profile = demand_profile[self.bidding_zones_list + ["Week"]]

        return demand_profile

    def _prepare_inflexible_demand_sources(self) -> None:
        """
        Load, scale, and save inflexible demand profiles from sources.
        """
        source_configs = [
            {
                "label": "DEMAND_INF_CLA",
                "aux_label": "demand_inflexible_classic",
                "profile_year_key": "demand_inf_cla_py",
                "amount_file_key": "demand_inf_cla_amount_file",
                "voll_key": "demand_inf_cla_voll",
                "base_path": self.path_demand_inflex_classic,
                "output_file": "demand_inflexible_classic.csv",
            },
            {
                "label": "DEMAND_INF_EV",
                "aux_label": "demand_inflexible_ev",
                "profile_year_key": "demand_inf_ev_py",
                "amount_file_key": "demand_inf_ev_amount_file",
                "voll_key": "demand_inf_ev_voll",
                "base_path": self.path_demand_inflex_ev,
                "output_file": "demand_inflexible_ev.csv",
            },
        ]

        for config in source_configs:
            scaled_df = self.calculate_inflexible_demand(config)
            utils.save_data(
                scaled_df,
                config["output_file"],
                output_dir=self.output_path,
                logger=self.logger,
            )

    def _save_aux_data_to_yaml(self) -> None:
        """Save auxiliary data dictionary to a YAML file."""
        yaml_path = Path(self.output_path) / f"{self.scenario_name}_aux_data.yaml"

        # Ensure output directory exists
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to YAML file
        with open(yaml_path, "w") as f:
            yaml.dump(
                self.aux_data_dict,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
            )

        self.logger.info(f"Auxiliary data saved to: {yaml_path}")
