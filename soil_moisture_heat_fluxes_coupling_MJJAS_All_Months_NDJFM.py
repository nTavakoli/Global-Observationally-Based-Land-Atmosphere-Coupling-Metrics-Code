#!/usr/bin/env python
# coding: utf-8

"""
Soil Moisture-surface heat flux Coupling Analysis

This script computes error-corrected correlation coefficients and coupling indices
between soil moisture and surface heat fluxes (EF, H, E) across three latitude
belts with region-specific seasonal aggregations:

Regional Configuration:
  • Northern:  23°N-60°N,   May-September (MJJAS, warm season)
  • Tropical:  23°S-23°N,   All months (year-round)
  • Southern:  23°S-60°S,   November-March (NDJFM, warm season)

Workflow:
  1. Load seasonal soil moisture and surface heat flux data (NetCDF)
  2. Concatenate seasonal files and filter by region/months
  3. Apply error adjusted using regional α parameters (from SMM analysis)
  4. Compute bias-corrected correlation and coupling indices
  5. Perform statistical significance testing (t-test with DOF adjusted)
  6. Save results to NetCDF with comprehensive metadata

Output Variables:
  - std_VAR: Standard deviation of surface heat flux
  - std_sm: Standard deviation of soil moisture
  - R_un: Not-Adjusted Pearson correlation coefficient
  - R: Adjusted Pearson correlation coefficient
  - I_un: Not-Adjusted coupling index (r × σ_VAR)
  - I: Adjusted coupling index (r × σ_VAR)
  - p_value: Two-sided p-value for significance of R
  - N: Number of valid time samples

Atmospheric Variables Processed:
  - EF: Evaporative Fraction (dimensionless)
  - H: Sensible Heat Flux (W m⁻²)
  - E: Evaporation Rate (mm day⁻¹)

Usage:
    python soil_moisture_heat_fluxes_coupling_MJJAS_All_Months_NDJFM.py
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

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# Static Configuration
# ──────────────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# USER CONFIGURATION: Modify these paths for your data
# ════════════════════════════════════════════════════════════════════

# Base directory for seasonal data files
BASE_INPUT_DIRECTORY = "./data/soilmoisture_atmosphere"

# Seasonal file paths for each variable
SEASON_PATHS = {
    "SoMo": {
        "DJF": os.path.join(BASE_INPUT_DIRECTORY, "layer1_DJF.nc"),
        "MAM": os.path.join(BASE_INPUT_DIRECTORY, "layer1_MAM.nc"),
        "JJA": os.path.join(BASE_INPUT_DIRECTORY, "layer1_JJA.nc"),
        "SON": os.path.join(BASE_INPUT_DIRECTORY, "layer1_SON.nc"),
    },
    "EF": {
        "DJF": os.path.join(BASE_INPUT_DIRECTORY, "EF_DJF.nc"),
        "MAM": os.path.join(BASE_INPUT_DIRECTORY, "EF_MAM.nc"),
        "JJA": os.path.join(BASE_INPUT_DIRECTORY, "EF_JJA.nc"),
        "SON": os.path.join(BASE_INPUT_DIRECTORY, "EF_SON.nc"),
    },
    "H": {
        "DJF": os.path.join(BASE_INPUT_DIRECTORY, "H_DJF.nc"),
        "MAM": os.path.join(BASE_INPUT_DIRECTORY, "H_MAM.nc"),
        "JJA": os.path.join(BASE_INPUT_DIRECTORY, "H_JJA.nc"),
        "SON": os.path.join(BASE_INPUT_DIRECTORY, "H_SON.nc"),
    },
    "E": {
        "DJF": os.path.join(BASE_INPUT_DIRECTORY, "E_DJF.nc"),
        "MAM": os.path.join(BASE_INPUT_DIRECTORY, "E_MAM.nc"),
        "JJA": os.path.join(BASE_INPUT_DIRECTORY, "E_JJA.nc"),
        "SON": os.path.join(BASE_INPUT_DIRECTORY, "E_SON.nc"),
    },
}

# Variable names in each NetCDF file
VARIABLE_NAMES = {
    "SoMo": "layer1",
    "EF": "EF",
    "H": "H",
    "E": "E"
}

# Month-to-season mapping
MONTH_TO_SEASON_MAP = {
    1: "DJF", 2: "DJF", 3: "MAM",
    4: "MAM", 5: "MAM", 6: "JJA",
    7: "JJA", 8: "JJA", 9: "SON",
    10: "SON", 11: "SON", 12: "DJF",
}

# Path to error adjusted parameters (α, SMM) for each region
def get_alpha_path(region_label):
    """
    Get path to α/SMM file for a given region.
    
    Parameters
    ----------
    region_label : str
        Region identifier (e.g., 'North_MJJAS', 'Tropics_AllMonths')
    
    Returns
    -------
    str
        Path to NetCDF file containing α and SMM
    """
    return f"./data/memory_parameters/SMM_{region_label}.nc"


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading Functions
# ──────────────────────────────────────────────────────────────────────────────
def load_seasonal_data(variable_key, months_to_select):
    """
    Load and concatenate seasonal files for a given variable.
    
    Parameters
    ----------
    variable_key : str
        Variable identifier ('SoMo', 'EF', 'H', or 'E')
    months_to_select : list or str
        List of month numbers (1-12) or 'ALL' for all months
    
    Returns
    -------
    xr.DataArray
        Concatenated and time-sorted data array
    """
    # Determine which seasons to load
    if months_to_select == "ALL":
        seasons = ["DJF", "MAM", "JJA", "SON"]
    else:
        seasons = sorted({MONTH_TO_SEASON_MAP[m] for m in months_to_select})
    
    # Load seasonal files
    arrays = []
    for season in seasons:
        file_path = SEASON_PATHS[variable_key][season]
        dataset = xr.open_dataset(file_path, engine="h5netcdf")
        var_name = VARIABLE_NAMES[variable_key]
        arrays.append(dataset[var_name])
    
    # Concatenate and sort by time
    combined = xr.concat(arrays, dim="time").sortby("time")
    
    # Filter to requested months if not all months
    if months_to_select != "ALL":
        combined = combined.sel(time=combined["time.month"].isin(months_to_select))
    
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Core Processing Function
# ──────────────────────────────────────────────────────────────────────────────
def compute_coupling_indices(soil_moisture, atmosphere_variable, alpha, smm,
                             variable_unit, variable_label):
    """
    Compute error-corrected correlation and coupling indices between
    soil moisture and atmospheric variables.
    
    This function:
      1. Aligns datasets and handles missing values
      2. Computes variance/covariance
      3. Applies error adjusted using α parameter
      4. Calculates bias-corrected correlation coefficient (R)
      5. Derives coupling index (I = r × σ)
      6. Performs statistical significance testing with DOF adjusted
    
    Parameters
    ----------
    soil_moisture : xr.DataArray
        Soil moisture time series (time, lat, lon)
    atmosphere_variable : xr.DataArray
        Atmospheric variable time series (time, lat, lon)
    alpha : xr.DataArray
        Error adjusted factor (lat, lon) - dimensionless
    smm : xr.DataArray
        Soil moisture memory in days (lat, lon) - used for DOF calculation
    variable_unit : str
        Unit of atmospheric variable (e.g., 'W m⁻²', 'mm day⁻¹')
    variable_label : str
        Label for atmospheric variable (e.g., 'EF', 'H', 'E')
    
    Returns
    -------
    tuple of (xr.Dataset, dict)
        - Dataset containing all output variables with attributes
        - Encoding dictionary for compression
    
    Notes
    -----
    Error adjusted follows:
      δ = √(α / (1 + α)) × σ_SM
      R_corrected = cov / √((σ_SM² - δ²) × σ_VAR²)
    
    DOF adjustment accounts for autocorrelation via SMM:
      DOF = (N / (SMM + 1)) - 2
    """
    # ─────────────────────────────────────────────────────────────────
    # Data alignment and chunking
    # ─────────────────────────────────────────────────────────────────
    soil_moisture, atmosphere_variable, alpha, smm = xr.align(
        soil_moisture, atmosphere_variable, alpha, smm,
        join="inner"
    )
    
    # Chunk for efficient Dask processing
    time_chunks = {"time": -1, "lat": 50, "lon": 50}
    spatial_chunks = {"lat": 50, "lon": 50}
    
    soil_moisture = soil_moisture.chunk(time_chunks)
    atmosphere_variable = atmosphere_variable.chunk(time_chunks)
    alpha = alpha.chunk(spatial_chunks)
    smm = smm.chunk(spatial_chunks)
    
    # ─────────────────────────────────────────────────────────────────
    # Mask for valid pairs
    # ─────────────────────────────────────────────────────────────────
    valid_mask = xr.ufuncs.logical_and(
        soil_moisture.notnull(),
        atmosphere_variable.notnull()
    )
    soil_moisture_valid = soil_moisture.where(valid_mask)
    atmosphere_variable_valid = atmosphere_variable.where(valid_mask)
    
    # ─────────────────────────────────────────────────────────────────
    # Count valid samples
    # ─────────────────────────────────────────────────────────────────
    sample_count = soil_moisture_valid.notnull().sum(dim="time")
    
    # ─────────────────────────────────────────────────────────────────
    # Variance and covariance calculations
    # ─────────────────────────────────────────────────────────────────
    variance_soil_moisture = soil_moisture_valid.var(dim="time", ddof=1, skipna=True)
    variance_atmosphere = atmosphere_variable_valid.var(dim="time", ddof=1, skipna=True)
    covariance = xr.cov(soil_moisture_valid, atmosphere_variable_valid, dim="time")
    
    # ─────────────────────────────────────────────────────────────────
    # Error adjusted
    # ─────────────────────────────────────────────────────────────────
    # δ = measurement error std = √(var_error)
    # var_error = (α / (1 + α)) × var_SM
    delta_variance = variance_soil_moisture * (alpha / (1.0 + alpha))
    
    # ─────────────────────────────────────────────────────────────────
    # Correlation coefficients
    # ─────────────────────────────────────────────────────────────────
    # Uncorrected: R = cov / √(var_SM × var_VAR)
    correlation_uncorrected = covariance / np.sqrt(
        variance_soil_moisture * variance_atmosphere
    )
    
    # Corrected: R = cov / √((var_SM - δ²) × var_VAR)
    correlation_corrected = covariance / np.sqrt(
        (variance_soil_moisture - delta_variance) * variance_atmosphere
    )
    
    # Validate bounds [-1, 1]
    correlation_uncorrected = xr.where(
        (correlation_uncorrected >= -1) & (correlation_uncorrected <= 1),
        correlation_uncorrected,
        np.nan
    )
    correlation_corrected = xr.where(
        (correlation_corrected >= -1) & (correlation_corrected <= 1),
        correlation_corrected,
        np.nan
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Coupling indices: I = R × σ_VAR
    # ─────────────────────────────────────────────────────────────────
    std_atmosphere = np.sqrt(variance_atmosphere)
    coupling_uncorrected = correlation_uncorrected * std_atmosphere
    coupling_corrected = correlation_corrected * std_atmosphere
    
    std_soil_moisture = np.sqrt(variance_soil_moisture)
    
    # ─────────────────────────────────────────────────────────────────
    # Statistical significance testing
    # ─────────────────────────────────────────────────────────────────
    # Validity check
    valid_for_test = (
        (sample_count > 2) &
        (correlation_corrected > -1) &
        (correlation_corrected < 1)
    )
    
    # Degrees of freedom adjusted for autocorrelation
    # DOF = (N / (SMM + 1)) - 2
    degrees_of_freedom = (
        (sample_count.where(valid_for_test) /
         (smm.where(valid_for_test) + 1.0)) - 2.0
    )
    
    # t-statistic: t = r × √(DOF / (1 - r²))
    t_statistic = (
        correlation_corrected.where(valid_for_test) *
        np.sqrt(degrees_of_freedom /
                (1.0 - correlation_corrected.where(valid_for_test)**2))
    )
    
    # p-value: two-sided test using Dask-safe apply_ufunc
    p_value = xr.apply_ufunc(
        lambda t, df: 2.0 * (1.0 - tdist.cdf(np.abs(t), df)),
        t_statistic,
        degrees_of_freedom,
        input_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Build output dataset
    # ─────────────────────────────────────────────────────────────────
    output_dataset = xr.Dataset({
        f"std_{variable_label}": std_atmosphere.assign_attrs(
            long_name=f"Standard deviation of {variable_label}",
            units=variable_unit,
        ),
        "std_sm": std_soil_moisture.assign_attrs(
            long_name="Standard deviation of soil moisture",
            units="m³ m⁻³",
        ),
        "R_un": correlation_uncorrected.assign_attrs(
            long_name=f"Uncorrected Pearson correlation (SM vs. {variable_label})",
            units="1",
        ),
        "R": correlation_corrected.assign_attrs(
            long_name=f"Bias-corrected Pearson correlation (SM vs. {variable_label})",
            units="1",
        ),
        "I_un": coupling_uncorrected.assign_attrs(
            long_name=f"Uncorrected coupling index (r × σ_{variable_label})",
            units=variable_unit,
        ),
        "I": coupling_corrected.assign_attrs(
            long_name=f"Bias-corrected coupling index (r × σ_{variable_label})",
            units=variable_unit,
        ),
        "p_value": p_value.assign_attrs(
            long_name="Two-sided p-value for significance of R",
            units="1",
        ),
        "N": sample_count.assign_attrs(
            long_name="Number of valid time samples",
            units="1",
        ),
    })
    
    # ─────────────────────────────────────────────────────────────────
    # Compression encoding
    # ─────────────────────────────────────────────────────────────────
    encoding = {
        var: {"zlib": True, "complevel": 1}
        for var in output_dataset.data_vars
    }
    
    return output_dataset, encoding


# ──────────────────────────────────────────────────────────────────────────────
# Regional Configurations
# ──────────────────────────────────────────────────────────────────────────────
REGIONAL_TASKS = [
    {
        "region_label": "North_MJJAS",
        "latitude_min": 60,
        "latitude_max": 23,
        "months": [5, 6, 7, 8, 9],
        "description": "Northern (23°N-60°N), May-September"
    },
    {
        "region_label": "Tropics_AllMonths",
        "latitude_min": 23,
        "latitude_max": -23,
        "months": "ALL",
        "description": "Tropical (23°S-23°N), All months"
    },
    {
        "region_label": "South_NDJFM",
        "latitude_min": -23,
        "latitude_max": -60,
        "months": [11, 12, 1, 2, 3],
        "description": "Southern (60°S-23°S), November-March"
    },
]

ATMOSPHERIC_VARIABLES = [
    {
        "variable_key": "EF",
        "variable_label": "EF",
        "variable_unit": "1"
    },
    {
        "variable_key": "H",
        "variable_label": "H",
        "variable_unit": "W m⁻²"
    },
    {
        "variable_key": "E",
        "variable_label": "E",
        "variable_unit": "mm day⁻¹"
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """
    Main execution function orchestrating coupling index computation
    for three regions and three atmospheric variables.
    """
    logging.info("=" * 80)
    logging.info("Soil Moisture-Atmosphere Coupling Indices Analysis")
    logging.info("=" * 80)
    
    # Initialize Dask cluster
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=16,
        memory_limit="128GB"
    )
    client = Client(cluster)
    logging.info(f"Dask cluster initialized")
    logging.info(f"Dashboard: {client.dashboard_link}")
    
    try:
        with ProgressBar():
            for region_task in REGIONAL_TASKS:
                region_label = region_task["region_label"]
                latitude_min = region_task["latitude_min"]
                latitude_max = region_task["latitude_max"]
                months = region_task["months"]
                description = region_task["description"]
                
                logging.info(f"▶  Processing {description}")
                
                # Load error adjusted parameters (α, SMM)
                alpha_file = get_alpha_path(region_label)
                dataset_alpha = xr.open_dataset(alpha_file)
                alpha_regional = dataset_alpha["alpha"].sel(lat=slice(latitude_min, latitude_max))
                smm_regional = dataset_alpha["SMM"].sel(lat=slice(latitude_min, latitude_max))
                
                # Process each atmospheric variable
                for atm_var in ATMOSPHERIC_VARIABLES:
                    var_key = atm_var["variable_key"]
                    var_label = atm_var["variable_label"]
                    var_unit = atm_var["variable_unit"]
                    
                    timer_start = perf_counter()
                    
                    # Load soil moisture and atmospheric variable data
                    soil_moisture_data = load_seasonal_data("SoMo", months) \
                        .sel(lat=slice(latitude_min, latitude_max))
                    
                    atmosphere_data = load_seasonal_data(var_key, months) \
                        .sel(lat=slice(latitude_min, latitude_max))
                    
                    # Compute coupling indices
                    dataset_output, encoding = compute_coupling_indices(
                        soil_moisture_data,
                        atmosphere_data,
                        alpha_regional,
                        smm_regional,
                        variable_unit=var_unit,
                        variable_label=var_label
                    )
                    
                    # Add global attributes
                    dataset_output.attrs.update({
                        "Institution": "George Mason University, Virginia, USA",
                        "Contact": "Nazanin Tavakoli (ntavakol@gmu.edu), Paul A Dirmeyer (pdirmeye@gmu.edu)",
                        "Description": (
                            f"Error-corrected soil moisture-{var_label} coupling indices "
                            f"for {description}"
                        ),
                        "Reference": "Tavakoli and Dirmeyer",
                        "Creation date": "2025"
                    })
                    
                    # Save output
                    output_directory = "./output/coupling_indices"
                    os.makedirs(output_directory, exist_ok=True)
                    output_file = os.path.join(
                        output_directory,
                        f"SoMo_{var_label}_{region_label}.nc"
                    )
                    
                    dataset_output.to_netcdf(output_file, encoding=encoding, engine='netcdf4')
                    
                    elapsed = perf_counter() - timer_start
                    logging.info(f"  ↳ {var_label}: {elapsed:.1f}s → {output_file}")
        
        logging.info("=" * 80)
        logging.info("✅ All coupling indices computed successfully")
        logging.info("=" * 80)
    
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
    
    finally:
        client.close()
        cluster.close()
        logging.info("Dask resources cleaned up")


if __name__ == "__main__":
    main()
    