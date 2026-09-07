from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import linopy

from enlight.utils import get_logger

log = get_logger(__name__)


class DataExporter:
    """
    Export optimization results from Enlight models.
    
    Supports both yearly and weekly run modes with automatic detection
    and appropriate handling of result concatenation.
    
    Attributes:
        model: EnlightModel or linopy.Model instance
        scenario_name: Name of the scenario
        scenario_config: Configuration dictionary
        root_path: Project root path
        results_path: Path where results will be saved
        results_dict: Dictionary storing extracted results
        exported_files: List of successfully exported files
    """
    
    def __init__(
        self,
        enlight_model,
        scenario_name: str,
        scenario_config: Dict,
        root_path: Path,
    ):
        self.enlight_model = enlight_model
        self.scenario_name = scenario_name
        self.scenario_config = scenario_config
        self.root_path = root_path
        
        # Initialize attributes
        self.results_dict = {}
        self.exported_files = []
        
        # Setup paths
        self._setup_paths()
        
        # Extract model if needed
        self._extract_model_object()
        
        log.info(f"DataExporter initialized for '{scenario_name}'")
        log.info(f"Results directory: {self.results_path}")
    
    def _setup_paths(self) -> None:
        """Create output directory structure."""
        self.results_path = (
            self.root_path / 'simulations' / self.scenario_name / "results"
        )
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def _extract_model_object(self) -> None:
        """Extract the linopy model from EnlightModel."""
        if isinstance(self.enlight_model, linopy.model.Model):
            log.info("Model is already a linopy.Model instance")
            self.model = self.enlight_model
            self.enlight_model = None  # No EnlightModel available
            
        elif hasattr(self.enlight_model, 'model'):  # ✓ Fixed
            log.info("Extracting linopy model from EnlightModel")
            self.model = self.enlight_model.model
            
        else:
            raise TypeError(
                f"Model must be linopy.Model or EnlightModel, "
                f"got {type(self.enlight_model)}"
            )
            
    def _get_solution(self, var):
        return var.solution.to_dataframe().squeeze().unstack()
    
    @staticmethod
    def _to_dataframe(data) -> Optional[pd.DataFrame]:
        """Convert xarray/solution to DataFrame."""
        try:
            df = data.to_dataframe()
            
            # Handle Series
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            # Unstack if multi-indexed
            if isinstance(df.index, pd.MultiIndex) and len(df.columns) == 1:
                col = df.columns[0]
                df = df[col].unstack()
            
            return df
        except:
            return None
    
    
    def export_solution(self) -> Dict[str, pd.DataFrame]:
        """
        Export main model results to CSV files.
        
        Returns:
            Dictionary of DataFrames with results
        """
        log.info("Exporting model results...")
        
        if self.model.status != 'ok':
            log.warning(f"Model status: {self.model.status}")
        
        # Extract main variables
        self._extract_main_variables()
        
        # Extract prices
        self._extract_dual_values()
        
        # Save to CSV
        self._save_to_csv()
        
        # Save metadata
        self._save_metadata()
        
        log.info(f"Exported {len(self.exported_files)} files")
        
        return self.results_dict
    
    # ========================================================================
    # VARIABLE SOLUTION EXTRACTION
    # ========================================================================
    
    def _extract_main_variables(self) -> None:
        """Extract only the most important variables."""
        if self.model is None:
            log.warning("No EnlightModel - skipping variable extraction")
            return
        
        # Extract all variables
        for var_name, var in self.model.variables.items():
            try:
                if hasattr(var, 'solution') and var.solution is not None:
                    df = self._get_solution(var)
                    if df is not None:
                        self.results_dict[var_name] = df
                        log.info(f"Exported {var_name}: {df.shape}")
            except Exception as e:
                log.debug(f"Skipping {var_name}: {e}")

    
    # ========================================================================
    # DUAL VALUE EXTRACTION
    # ========================================================================
    
    def _extract_dual_values(self) -> None:
        """Extract dual values (shadow prices) from constraints."""
        if self.enlight_model is None:
            log.warning(
                "Cannot extract duals - EnlightModel not available"
            )
            return
        
        m = self.enlight_model
        
        # Define constraints to extract duals from
        constraints_map = {
            "zonal_prices": (m.power_balance, "Power balance duals (zonal prices)"),
            "export_duals": (m.electricity_exports, "Export constraint duals"),
            "flexible_demand_duals": (m.demand_flexible_classic_limit, "Flexible demand duals"),
        }
        
        # Extract each constraint's dual
        for dual_name, (constraint, description) in constraints_map.items():
            self._extract_single_dual(dual_name, constraint, description)
    
    def _extract_single_dual(
        self,
        dual_name: str,
        constraint,
        description: str
    ) -> None:
        """
        Extract dual values from a single constraint.
        
        Args:
            dual_name: Name for the dual values
            constraint: Linopy constraint object
            description: Human-readable description
        """
        if constraint is None:
            log.debug(f"  Skipping {dual_name} (None)")
            return
        
        try:
            if not hasattr(constraint, 'dual') or constraint.dual is None:
                log.debug(f"  Skipping {dual_name} (no dual values)")
                return
            
            # Convert to DataFrame
            dual_df = constraint.dual.to_dataframe()
            
            if isinstance(dual_df, pd.Series):
                dual_df = dual_df.to_frame(name=dual_name)
            
            # Unstack if multi-indexed
            if isinstance(dual_df.index, pd.MultiIndex):
                value_col = dual_df.columns[0]
                dual_df = dual_df[value_col].unstack()
            
            self.results_dict[dual_name] = dual_df
            log.info(f"Extracted {description}: {dual_df.shape}")
            
        except Exception as e:
            log.warning(f"Failed to extract {dual_name}: {e}")
    
    
    # ========================================================================
    # SAVE METHODS
    # ========================================================================
    
    def _save_to_csv(self) -> None:
        """Save all results to CSV files."""
        for name, df in self.results_dict.items():
            try:
                filepath = self.results_path / f"{name}.csv"
                df.to_csv(filepath)
                self.exported_files.append(str(filepath))
            except Exception as e:
                log.warning(f"Failed to save {name}: {e}")
    
    def _save_metadata(self) -> None:
        """Save basic run metadata to YAML file."""
        try:
            metadata = {
                'scenario': self.scenario_name,
                'status': self.model.status,
                'objective': float(self.model.objective.value),
                'timestamp': datetime.now().isoformat(),
            }
            
            if self.enlight_model and hasattr(self.enlight_model.data, 'week'):
                metadata['week'] = self.enlight_model.data.week
            
            # ✓ Changed to YAML
            filepath = self.results_path / 'metadata.yaml'  # .yaml instead of .json
            with open(filepath, 'w') as f:
                yaml.dump(
                    metadata, 
                    f, 
                    default_flow_style=False,  # Makes it more readable
                    sort_keys=False            # Keeps original order
                )
            
            self.exported_files.append(str(filepath))
            
        except Exception as e:
            log.debug(f"Could not save metadata: {e}")
