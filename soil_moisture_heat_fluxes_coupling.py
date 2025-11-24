#!/usr/bin/env python
# coding: utf-8

"""
Soil Moisture-surface heat flux Coupling Analysis


This script computes error-adjusted coupling indices between soil moisture and
surface heaf fluxes (evaporative fraction, sensible heat flux, evaporation) by:
  1. Loading soil moisture,heaf fluxes, and error parameters
  2. Computing error-adjusted correlations between variables
  3. Deriving terrestrial coupling indices
  4. Computing statistical significance (p-values with effective DOF)
  5. Saving results to NetCDF with comprehensive metadata

Workflow:
  - Input: Seasonal soil moisture data, heaf fluxes, and SMM error parameters
  - Output: Error-adjusted correlation (R), coupling index (I), standard deviation, 
            sample size (N), and p-values for each season and variable

Usage:
    python soil_moisture_heat_fluxes_coupling.py
"""

import os
import logging
import warnings
from time import perf_counter

import numpy as np
import xarray as xr
from scipy.stats import t as tdist
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# ──────────────────────────────────────────────────────────────────────────────
# Configuration and Setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings("ignore")

# Base output directory and dataset name for filenames
BASE_OUTPUT_DIRECTORY = './output/coupling_indices'
DATASET_IDENTIFIER = "your_dataset_name_here"  # e.g., "SMAP_GLEAM"
#DATASET_IDENTIFIER = 'SMAP_GLEAM'

# ──────────────────────────────────────────────────────────────────────────────
# Core Processing Functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_adjusted_coupling_metrics(input_soil_moisture, input_atmosphere_var, 
                                       input_error_parameters, atmosphere_var_name):
    """
    Compute error-adjusted correlation and coupling indices.
    
    This function loads soil moisture, atmospheric variables, and error correction
    parameters, then computes the error-adjusted Pearson correlation coefficient
    and derives the terrestrial coupling index.
    
    Parameters
    ----------
    input_soil_moisture : str
        Path to NetCDF file containing soil moisture data
    input_atmosphere_var : str
        Path to NetCDF file containing atmospheric variable data
    input_error_parameters : str
        Path to NetCDF file containing error correction parameters (alpha, SMM)
    atmosphere_var_name : str
        Variable name in the atmosphere NetCDF file
    
    Returns
    -------
    tuple of DataArray
        (std_atmosphere, R_adjusted, coupling_index, sample_count, SMM)
        where each element represents the corresponding metric
    
    Notes
    -----
    Error correction removes measurement error from soil moisture variance:
    delta² = var(SM) * α / (1 + α)
    R_adjusted = cov(SM, VAR) / sqrt((var(SM) - delta²) * var(VAR))
    """
    try:
        # Load datasets
        ds_soil_moisture = xr.open_dataset(input_soil_moisture)
        ds_atmosphere = xr.open_dataset(input_atmosphere_var)
        ds_error_params = xr.open_dataset(input_error_parameters)

        # Extract variables
        soil_moisture = ds_soil_moisture['soil_moisture']
        atmosphere_variable = ds_atmosphere[atmosphere_var_name]
        alpha = ds_error_params['alpha']
        smm = ds_error_params['SMM']

        # Harmonize coordinates to avoid floating-point misalignment
        soil_moisture = soil_moisture.assign_coords(
            lat=soil_moisture.lat.round(4),
            lon=soil_moisture.lon.round(4)
        )
        atmosphere_variable = atmosphere_variable.assign_coords(
            lat=atmosphere_variable.lat.round(4),
            lon=atmosphere_variable.lon.round(4)
        )
        alpha = alpha.assign_coords(lat=alpha.lat.round(4), lon=alpha.lon.round(4))
        smm = smm.assign_coords(lat=smm.lat.round(4), lon=smm.lon.round(4))

        # Align all variables on common grid
        soil_moisture, atmosphere_variable, alpha, smm = xr.align(
            soil_moisture, atmosphere_variable, alpha, smm,
            join='inner'
        )

        # Configure chunking for Dask parallelization
        chunk_temporal = {'time': -1, 'lat': 50, 'lon': 50}
        chunk_spatial = {'lat': 50, 'lon': 50}
        soil_moisture = soil_moisture.chunk(chunk_temporal)
        atmosphere_variable = atmosphere_variable.chunk(chunk_temporal)
        alpha = alpha.chunk(chunk_spatial)
        smm = smm.chunk(chunk_spatial)

        logging.info(
            f"Data chunks configured — "
            f"SM:{soil_moisture.chunks}, "
            f"{atmosphere_var_name}:{atmosphere_variable.chunks}, "
            f"alpha:{alpha.chunks}, SMM:{smm.chunks}"
        )

        # ─────────────────────────────────────────────────────────────────
        # Create valid data mask and compute statistics
        # ─────────────────────────────────────────────────────────────────
        valid_data_mask = xr.ufuncs.logical_and(
            soil_moisture.notnull(),
            atmosphere_variable.notnull()
        )
        soil_moisture_valid = soil_moisture.where(valid_data_mask)
        atmosphere_variable_valid = atmosphere_variable.where(valid_data_mask)
        sample_count = soil_moisture_valid.notnull().sum(dim='time')

        # Compute variances and covariance
        variance_soil_moisture = soil_moisture_valid.var(dim='time', ddof=1, skipna=True)
        variance_atmosphere = atmosphere_variable_valid.var(dim='time', ddof=1, skipna=True)
        covariance = xr.cov(soil_moisture_valid, atmosphere_variable_valid, dim='time')

        # ─────────────────────────────────────────────────────────────────
        # Apply error correction
        # ─────────────────────────────────────────────────────────────────
        # Error variance of soil moisture: δ² = var(SM) * α / (1 + α)
        error_variance_sm = variance_soil_moisture * (alpha / (1.0 + alpha))

        # Error-adjusted correlation (clipped to valid range)
        R_adjusted = covariance / np.sqrt(
            (variance_soil_moisture - error_variance_sm) * variance_atmosphere
        )
        R_adjusted = xr.where(
            (R_adjusted >= -1) & (R_adjusted <= 1),
            R_adjusted,
            np.nan
        )

        # Coupling index: I = R * std(VAR)
        std_atmosphere = np.sqrt(variance_atmosphere)
        coupling_index = R_adjusted * std_atmosphere

        return std_atmosphere, R_adjusted, coupling_index, sample_count, smm

    except Exception as e:
        logging.error(
            f"Error in compute_adjusted_coupling_metrics for {atmosphere_var_name}: {e}"
        )
        raise


def process_seasonal_coupling(season_config, variable_config, dask_client):
    """
    Process one season and one atmospheric variable to compute coupling metrics.
    
    This function orchestrates the complete analysis for a single season-variable
    combination by:
      1. Computing error-adjusted correlations and coupling indices
      2. Computing statistical significance using effective degrees of freedom
      3. Assembling results into a NetCDF dataset with comprehensive metadata
      4. Saving to disk with compression
    
    Parameters
    ----------
    season_config : dict
        Configuration for one season containing keys:
        - 'season': season code (e.g., 'DJF')
        - 'input_soil_moisture': path to soil moisture file
        - 'input_atmosphere_var': path to atmosphere variable file
        - 'input_error_parameters': path to error parameters file
    variable_config : dict
        Configuration for atmospheric variable containing keys:
        - 'var_label': short variable label (e.g., 'EF', 'H', 'E')
        - 'var_name': variable name in NetCDF file
    dask_client : dask.distributed.Client
        Active Dask client for parallel computation
    """
    season_code = season_config['season']
    var_label = variable_config['var_label']
    var_name = variable_config['var_name']

    input_soil_moisture = season_config['input_soil_moisture']
    input_atmosphere_var = season_config['input_atmosphere_var']
    input_error_parameters = season_config['input_error_parameters']

    # Create output directory (organized by season)
    output_directory = os.path.join(BASE_OUTPUT_DIRECTORY, season_code)
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(
        output_directory,
        f"Coupling_SM_{var_label}_{DATASET_IDENTIFIER}.nc"
    )

    logging.info(f"Processing {var_label} for {season_code}")
    logging.info(f"Output file: {output_file}")

    # ─────────────────────────────────────────────────────────────────────
    # Variable-specific metadata
    # ─────────────────────────────────────────────────────────────────────
    variable_metadata = {
        'EF': {
            'units': 'dimensionless',
            'long_name': 'evaporative fraction',
            'full_name_with_abbr': 'evaporative fraction (EF)'
        },
        'H': {
            'units': 'W m-2',
            'long_name': 'sensible heat flux',
            'full_name_with_abbr': 'sensible heat flux (H)'
        },
        'E': {
            'units': 'mm day-1',
            'long_name': 'evaporation',
            'full_name_with_abbr': 'evaporation (E)'
        }
    }

    var_info = variable_metadata.get(
        var_label,
        {'units': 'unknown', 'long_name': var_label, 'full_name_with_abbr': var_label}
    )
    var_units = var_info['units']
    var_long_name = var_info['long_name']
    var_full_name_abbr = var_info['full_name_with_abbr']

    try:
        # ─────────────────────────────────────────────────────────────────
        # Compute coupling metrics
        # ─────────────────────────────────────────────────────────────────
        start_time = perf_counter()
        
        with ProgressBar():
            std_atmosphere, R_adjusted, coupling_index, sample_count, smm = \
                compute_adjusted_coupling_metrics(
                    input_soil_moisture,
                    input_atmosphere_var,
                    input_error_parameters,
                    var_name
                )
            # Trigger computation
            std_atmosphere = std_atmosphere.compute()
            R_adjusted = R_adjusted.compute()
            coupling_index = coupling_index.compute()
            sample_count = sample_count.compute()
            smm = smm.compute()

        # ─────────────────────────────────────────────────────────────────
        # Compute statistical significance
        # ─────────────────────────────────────────────────────────────────
        # Valid correlation mask (valid range and sufficient samples)
        valid_correlation_mask = (R_adjusted > -1) & (R_adjusted < 1) & (sample_count > 2)
        
        # Effective degrees of freedom: DOF = N / (SMM + 1) - 2
        effective_dof = (sample_count.where(valid_correlation_mask) / 
                        (smm.where(valid_correlation_mask) + 1.0)) - 2.0
        
        valid_mask_final = valid_correlation_mask & (effective_dof > 0)
        
        # t-statistic: t = R * sqrt(DOF / (1 - R²))
        t_statistic = (R_adjusted.where(valid_mask_final) * 
                      np.sqrt(effective_dof.where(valid_mask_final) / 
                              (1.0 - R_adjusted.where(valid_mask_final)**2)))

        # Two-sided p-value from t-distribution
        t_statistic_np = np.where(np.isfinite(t_statistic), t_statistic.values, np.nan)
        dof_np = np.where(np.isfinite(effective_dof), effective_dof.values, np.nan)
        cdf_np = tdist.cdf(np.abs(t_statistic_np), df=dof_np)
        p_value_np = 2.0 * (1.0 - cdf_np)
        p_value = xr.DataArray(
            p_value_np,
            coords=R_adjusted.coords,
            dims=R_adjusted.dims,
            name='p_value'
        )

        # ─────────────────────────────────────────────────────────────────
        # Assemble output dataset
        # ─────────────────────────────────────────────────────────────────
        ds_output = xr.Dataset(
            data_vars={
                f"std_{var_label}": std_atmosphere,
                "R": R_adjusted,
                "I": coupling_index,
                "p_value": p_value,
                "N": sample_count
            }
        )

        # Replace non-finite values with NaN
        ds_output = ds_output.where(np.isfinite(ds_output), np.nan)

        # ─────────────────────────────────────────────────────────────────
        # Add coordinate metadata
        # ─────────────────────────────────────────────────────────────────
        if 'lat' in ds_output.coords:
            ds_output.coords['lat'].attrs.update({
                "long_name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "axis": "Y"
            })
        if 'lon' in ds_output.coords:
            ds_output.coords['lon'].attrs.update({
                "long_name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "axis": "X"
            })

        # ─────────────────────────────────────────────────────────────────
        # Add global (file-level) attributes
        # ─────────────────────────────────────────────────────────────────
        ds_output.attrs.update({
            "Institution": "George Mason University, Virginia, USA",
            "Contact": "Nazanin Tavakoli (ntavakol@gmu.edu), Paul A Dirmeyer (pdirmeye@gmu.edu)",
            "Description": (
                "Random error adjusted coupling indices derived from Soil Moisture Active Passive (SMAP) "
                "and Global Land Evaporation Amsterdam Model (GLEAM) "
                f"for {season_code} from April 2015 to March 2023."
            ),
            "Reference": "Tavakoli and Dirmeyer",
            "Creation date": "2025"
        })

        # ─────────────────────────────────────────────────────────────────
        # Add variable-specific attributes
        # ─────────────────────────────────────────────────────────────────
        ds_output[f"std_{var_label}"].attrs.update({
            "long_name": f"Standard deviation of {var_full_name_abbr} for {season_code}",
            "units": var_units
        })

        ds_output["R"].attrs.update({
            "long_name": (
                f"Random error adjusted correlation between soil moisture "
                f"and {var_long_name} for {season_code}"
            ),
            "standard_name": f"Radjusted(SM,{var_label})",
            "units": "dimensionless",
            "statistic": "Pearson Correlation Coefficient",
            "comment": (
                f"R = cov(SM, {var_label}) / sqrt((var(SM) - delta²) * var({var_label})); "
                "delta² = var(SM) * alpha / (1 + alpha); "
                "R is the random error adjusted correlation; "
                "delta² is the random-error variance of soil moisture."
            )
        })

        ds_output["I"].attrs.update({
            "long_name": (
                f"Random error adjusted terrestrial coupling index between "
                f"soil moisture and {var_long_name} for {season_code}"
            ),
            "standard_name": f"Iadjusted(SM,{var_label})",
            "units": var_units,
            "comment": (
                f"Computed as I = R * std({var_label}), where R is the random error "
                f"adjusted correlation and std({var_label}) is the standard deviation "
                f"of {var_long_name}."
            )
        })

        ds_output["N"].attrs.update({
            "long_name": f"Sample size used to calculate R for {season_code}",
            "units": "dimensionless",
            "comment": (
                f"Count of non-NaN soil moisture and {var_label} pairs after alignment on time, "
                f"used to compute effective degrees of freedom and p-value."
            )
        })

        ds_output["p_value"].attrs.update({
            "long_name": f"Two-sided p-value for adjusted correlation R during {season_code}",
            "units": "dimensionless",
            "comment": (
                "Two-sided p-value under H0: true correlation = 0 using Student t-test "
                "with effective degrees of freedom DOF = N/(SMM + 1) - 2; "
                "t = R * sqrt(DOF / (1 - R²))."
            )
        })

        # ─────────────────────────────────────────────────────────────────
        # Configure compression and save
        # ─────────────────────────────────────────────────────────────────
        encoding = {
            var: {'zlib': True, 'complevel': 1}
            for var in ds_output.data_vars
        }

        ds_output.to_netcdf(
            output_file,
            encoding=encoding,
            engine='netcdf4',
            format='NETCDF4'
        )

        elapsed_time = perf_counter() - start_time
        logging.info(
            f"Finished {var_label} {season_code} → {output_file} in {elapsed_time:.2f}s"
        )

    except Exception as e:
        logging.error(f"Error while processing {var_label} in {season_code}: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Define atmospheric variables to analyze
    atmosphere_variables = [
        {'var_label': 'EF', 'var_name': 'EF'},
        {'var_label': 'H', 'var_name': 'H'},
        {'var_label': 'E', 'var_name': 'E'},
    ]

    # ════════════════════════════════════════════════════════════════════
    # USER CONFIGURATION: Modify these paths for your data
    # ════════════════════════════════════════════════════════════════════
    seasons_configuration = [
        {
            'season': 'DJF',
            'input_soil_moisture': './data/soil_moisture_DJF.nc',
            'input_EF': './data/EF_DJF.nc',
            'input_H': './data/H_DJF.nc',
            'input_E': './data/E_DJF.nc',
            'input_error_parameters': './data/error_parameters_DJF.nc',
        },
        {
            'season': 'JJA',
            'input_soil_moisture': './data/soil_moisture_JJA.nc',
            'input_EF': './data/EF_JJA.nc',
            'input_H': './data/H_JJA.nc',
            'input_E': './data/E_JJA.nc',
            'input_error_parameters': './data/error_parameters_JJA.nc',
        },
        {
            'season': 'MAM',
            'input_soil_moisture': './data/soil_moisture_MAM.nc',
            'input_EF': './data/EF_MAM.nc',
            'input_H': './data/H_MAM.nc',
            'input_E': './data/E_MAM.nc',
            'input_error_parameters': './data/error_parameters_MAM.nc',
        },
        {
            'season': 'SON',
            'input_soil_moisture': './data/soil_moisture_SON.nc',
            'input_EF': './data/EF_SON.nc',
            'input_H': './data/H_SON.nc',
            'input_E': './data/E_SON.nc',
            'input_error_parameters': './data/error_parameters_SON.nc',
        }
    ]

    # ════════════════════════════════════════════════════════════════════
    # Configure and execute processing with Dask
    # ════════════════════════════════════════════════════════════════════
    cluster = None
    client = None
    
    try:
        # Initialize Dask cluster
        cluster = LocalCluster(
            n_workers=4,
            threads_per_worker=8,
            memory_limit='32GB'
        )
        client = Client(cluster)

        if client.status == "running":
            logging.info("Dask client connected successfully.")
            logging.info(f"Dask dashboard: {client.dashboard_link}")
        else:
            logging.warning("Dask client did not establish connection.")

        # ────────────────────────────────────────────────────────────────
        # Loop over seasons and variables
        # ────────────────────────────────────────────────────────────────
        for season in seasons_configuration:
            for variable in atmosphere_variables:
                # Map variable label to input file key
                if variable['var_label'] == 'EF':
                    input_var_file = season['input_EF']
                elif variable['var_label'] == 'H':
                    input_var_file = season['input_H']
                elif variable['var_label'] == 'E':
                    input_var_file = season['input_E']
                else:
                    logging.warning(f"Unknown var_label: {variable['var_label']}")
                    continue

                # Configure for this season-variable combination
                season_config = {
                    'season': season['season'],
                    'input_soil_moisture': season['input_soil_moisture'],
                    'input_atmosphere_var': input_var_file,
                    'input_error_parameters': season['input_error_parameters'],
                }

                # Process this combination
                process_seasonal_coupling(season_config, variable, client)

    except Exception as e:
        logging.error(f"Error in Dask setup or processing pipeline: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            if client is not None:
                client.close()
        except Exception:
            pass
        try:
            if cluster is not None:
                cluster.close()
        except Exception:
            pass
        
        