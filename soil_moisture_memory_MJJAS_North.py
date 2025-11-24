#!/usr/bin/env python
# coding: utf-8

"""
Soil Moisture Memory for Northern Region (MJJAS)

This script computes random error adjusted soil moisture memory (SMM) and
measurement error metrics from seasonal soil moisture residuals for the
Northern region (23°N-60°N) during May-September (MJJAS) by:
  1. Computing autocorrelation functions (ACF) for seasonal data
  2. Fitting exponential decay models to ACF values
  3. Deriving memory parameters (SMM) and error correction factors
  4. Computing measurement error variance metrics
  5. Saving results to NetCDF with comprehensive metadata

Workflow:
  - Input: Seasonal soil moisture residual data (NetCDF format)
  - Processing: Annual grouping, ACF computation, exponential fitting
  - Output: SMM, delta_sigma (δ/σ), sigma2 (σ²), delta2 (δ²)
  
Regional Specifics:
  - Region: Northern (23°N-60°N)
  - Season: May-September (MJJAS)
  - Time period: 1979-2023
  - Data source: Climate Change Initiative (CCI) soil moisture residuals

Usage:
    python soil_moisture_memory_MJJAS_North.py
"""

import os
import gc
import logging
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import warnings

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
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def linear_model(x, slope, intercept):
    """
    Simple linear model for exponential decay fitting.
    
    Parameters
    ----------
    x : ndarray
        Lag values
    slope : float
        Slope coefficient
    intercept : float
        Intercept coefficient
    
    Returns
    -------
    ndarray
        Linear predictions
    """
    return slope * x + intercept


def extract_lagged_pairs(time_series, lag):
    """
    Extract valid (non-NaN) lagged pairs from time series.
    
    Parameters
    ----------
    time_series : ndarray
        Time series with potential NaN values
    lag : int
        Lag value for pairing
    
    Returns
    -------
    tuple of ndarray
        (original_values, lagged_values) where both are non-NaN
    """
    time_series = np.asarray(time_series)
    valid_mask = ~np.isnan(time_series)
    shifted_mask = np.roll(valid_mask, -lag)
    shifted_mask[-lag:] = False
    combined_mask = valid_mask & shifted_mask
    return time_series[combined_mask], time_series[np.where(combined_mask)[0] + lag]


def compute_autocorrelation_function(series_list, maximum_lag):
    """
    Compute autocorrelation function (ACF) for a list of time series.
    
    This function computes Pearson correlation coefficients between lagged
    values of the time series, handling missing data appropriately.
    
    Parameters
    ----------
    series_list : list of ndarray
        List of time series (each can contain NaN values)
    maximum_lag : int
        Maximum lag to compute ACF for
    
    Returns
    -------
    list of float
        ACF values for lags 1 to maximum_lag
    
    Notes
    -----
    Returns NaN for lags where insufficient valid pairs exist.
    """
    acf_values = []
    
    for lag in range(1, maximum_lag + 1):
        xs_list = []
        ys_list = []
        
        for time_series in series_list:
            if len(time_series) >= lag + 1:
                x, y = extract_lagged_pairs(time_series, lag)
                xs_list.append(x)
                ys_list.append(y)
        
        if xs_list:
            x_combined = np.concatenate(xs_list)
            y_combined = np.concatenate(ys_list)
            x_mean = np.nanmean(x_combined)
            y_mean = np.nanmean(y_combined)
            covariance = np.sum((x_combined - x_mean) * (y_combined - y_mean))
            variance_x = np.sum((x_combined - x_mean) ** 2)
            variance_y = np.sum((y_combined - y_mean) ** 2)
            
            if variance_x * variance_y > 0:
                acf_values.append(covariance / np.sqrt(variance_x * variance_y))
            else:
                acf_values.append(np.nan)
        else:
            acf_values.append(np.nan)
    
    return acf_values


def estimate_memory_parameter(exponential_fits):
    """
    Estimate soil moisture memory (SMM) from exponential decay fits.
    
    Uses the relationship SMM = (-1 - intercept) / slope, bounded to [0, 153] days.
    
    Parameters
    ----------
    exponential_fits : list of tuple
        List of (lag_indices, log_acf_values, fit_params, param_cov) tuples
        where fit_params = [slope, intercept]
    
    Returns
    -------
    float
        Soil moisture memory in days, or NaN if estimation fails
    """
    if not exponential_fits:
        return np.nan
    
    # Use last (best) fit
    slope, intercept = exponential_fits[-1][2]
    
    if slope == 0:
        return np.nan
    
    # SMM = (-1 - intercept) / slope
    memory_parameter = (-1 - intercept) / slope if intercept > 0 else -1 / slope
    
    # Validate and bound result
    if memory_parameter < 0:
        return np.nan
    
    return min(memory_parameter, 153.0)  # Upper bound: 153 days


# ──────────────────────────────────────────────────────────────────────────────
# Main Processing Function
# ──────────────────────────────────────────────────────────────────────────────
def process_time_series(soil_moisture_data, time_values, region_tag="REGION"):
    """
    Compute SMM and error parameters from soil moisture time series.
    
    This function:
      1. Groups data by year
      2. Computes ACF for each year
      3. Fits exponential decay model to ACF
      4. Derives SMM and error correction factors
    
    Parameters
    ----------
    soil_moisture_data : ndarray
        Soil moisture time series
    time_values : ndarray
        Corresponding time values
    region_tag : str, optional
        Region identifier for logging
    
    Returns
    -------
    tuple of float
        (alpha, smm) where:
        - alpha: error correction factor (unitless)
        - smm: soil moisture memory in days
    """
    # Check for all-NaN data
    if np.isnan(soil_moisture_data).all():
        return np.nan, np.nan
    
    # Convert to xarray for easier time grouping
    data_array = xr.DataArray(
        soil_moisture_data,
        coords={'time': time_values},
        dims='time'
    )
    data_array['time'] = pd.to_datetime(data_array['time'].values)
    
    # Group by year
    groups = data_array.groupby("time.year")
    annual_series_list = [group.values for _, group in groups]
    
    if not annual_series_list:
        return np.nan, np.nan
    
    # Determine maximum lag from longest annual series
    maximum_lag = max(len(ts) for ts in annual_series_list) - 1
    
    if maximum_lag < 1:
        return np.nan, np.nan
    
    # ─────────────────────────────────────────────────────────────────
    # Compute ACF and fit exponential decay
    # ─────────────────────────────────────────────────────────────────
    acf_values = compute_autocorrelation_function(annual_series_list, maximum_lag)
    
    exponential_fits = []
    
    for num_lags in range(2, 8):  # Fit using lags 2-7
        lag_indices = np.arange(1, num_lags + 1)
        acf_subset = np.asarray(acf_values[:num_lags])
        
        # Select valid (positive, non-NaN) ACF values
        valid_indices = np.where((acf_subset > 0) & (~np.isnan(acf_subset)))[0]
        
        if valid_indices.size < 2:
            continue
        
        try:
            # Fit: ln(acf) = slope * lag + intercept
            fit_params, param_cov = curve_fit(
                linear_model,
                lag_indices[valid_indices],
                np.log(acf_subset[valid_indices])
            )
            exponential_fits.append(
                (lag_indices[valid_indices], np.log(acf_subset[valid_indices]), 
                 fit_params, param_cov)
            )
        except Exception as e:
            logging.warning(f"Exponential fit failed for {region_tag}: {e}")
            exponential_fits.append(([], [], [np.nan, np.nan], np.nan))
    
    # ─────────────────────────────────────────────────────────────────
    # Compute alpha and SMM
    # ─────────────────────────────────────────────────────────────────
    # Average intercept across all fits
    intercepts = [fit[2][1] for fit in exponential_fits if not np.isnan(fit[2][1])]
    average_intercept = np.nanmean(intercepts) if intercepts else np.nan
    
    # α = 1 - exp(min(0, intercept))
    alpha = 1 - np.exp(min(0, average_intercept)) if not np.isnan(average_intercept) else np.nan
    
    # SMM = (-1 - intercept) / slope
    smm = estimate_memory_parameter(exponential_fits)
    
    return alpha, smm


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading Functions
# ──────────────────────────────────────────────────────────────────────────────
def load_seasonal_dataset(file_path, latitude_min, latitude_max):
    """
    Load and subset seasonal soil moisture dataset.
    
    Parameters
    ----------
    file_path : str
        Path to NetCDF file
    latitude_min : float
        Minimum latitude for subsetting
    latitude_max : float
        Maximum latitude for subsetting
    
    Returns
    -------
    xr.Dataset
        Subsetted and sorted dataset
    """
    return (
        xr.open_dataset(file_path, engine="h5netcdf")
        .sel(lat=slice(latitude_min, latitude_max))
        .astype("float32")
        .sortby("time")
    )


def load_regional_data(season_files, latitude_min, latitude_max, months_to_select):
    """
    Load and combine multiple seasonal files, then subset by months.
    
    Parameters
    ----------
    season_files : dict
        Mapping of season codes to file paths
    latitude_min : float
        Minimum latitude
    latitude_max : float
        Maximum latitude
    months_to_select : list of int
        Month numbers (1-12) to keep
    
    Returns
    -------
    xr.Dataset
        Combined and month-filtered dataset
    """
    # Determine which seasons to load
    season_codes = sorted({
        {1: "DJF", 2: "DJF", 3: "MAM",
         4: "MAM", 5: "MAM", 6: "JJA",
         7: "JJA", 8: "JJA", 9: "SON",
         10: "SON", 11: "SON", 12: "DJF"}[m]
        for m in months_to_select
    })
    
    # Load all needed seasons
    datasets = [
        load_seasonal_dataset(season_files[season], latitude_min, latitude_max)
        for season in season_codes
    ]
    
    # Combine and sort
    combined = xr.concat(datasets, dim="time").sortby("time")
    
    # Filter to requested months
    combined = combined.sel(time=combined["time.month"].isin(months_to_select))
    
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Analysis and Output Functions
# ──────────────────────────────────────────────────────────────────────────────
def analyze_and_save_results(dataset, output_file_path, region_description=""):
    """
    Compute memory parameters and save to NetCDF.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing 'sm' (soil moisture) variable with dimensions (time, lat, lon)
    output_file_path : str
        Path where to save output NetCDF file
    region_description : str, optional
        Description of region for metadata
    """
    logging.info(f"Processing {region_description}...")
    
    # ─────────────────────────────────────────────────────────────────
    # Compute variances and SMM parameters
    # ─────────────────────────────────────────────────────────────────
    soil_moisture = dataset["sm"].chunk({"lat": 50, "lon": 50, "time": -1})
    variance_soil_moisture = soil_moisture.var(dim="time", ddof=1, skipna=True)
    
    # Compute alpha and SMM using apply_ufunc
    alpha, smm = xr.apply_ufunc(
        process_time_series,
        soil_moisture,
        soil_moisture["time"],
        kwargs={"region_tag": region_description},
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Compute error metrics
    # ─────────────────────────────────────────────────────────────────
    # δ/σ = sqrt(α / (1 + α))
    delta_sigma = np.sqrt(alpha / (1 + alpha))
    
    # δ² = (α / (1 + α)) * σ²
    variance_error = (alpha / (1 + alpha)) * variance_soil_moisture
    
    # ─────────────────────────────────────────────────────────────────
    # Build output dataset
    # ─────────────────────────────────────────────────────────────────
    output_dataset = xr.Dataset({
        'SMM': smm,
        'delta_sigma': delta_sigma,
        'sigma2': variance_soil_moisture,     # σ²
        'delta2': variance_error              # δ²
    })
    
    # ─────────────────────────────────────────────────────────────────
    # Add coordinate attributes
    # ─────────────────────────────────────────────────────────────────
    if "lat" in output_dataset.coords:
        output_dataset["lat"].attrs.update({
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        })
    if "lon" in output_dataset.coords:
        output_dataset["lon"].attrs.update({
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        })
    
    # ─────────────────────────────────────────────────────────────────
    # Add variable attributes
    # ─────────────────────────────────────────────────────────────────
    output_dataset["delta_sigma"].attrs.update({
        "long_name": f"Relative measurement error (δ/σ) for {region_description}",
        "units": "Dimensionless"
    })
    
    output_dataset["sigma2"].attrs.update({
        "long_name": f"Variance of observed soil moisture (σ²) for {region_description}",
        "units": "1"
    })
    
    output_dataset["delta2"].attrs.update({
        "long_name": f"Variance of soil moisture random error (δ²) for {region_description}",
        "units": "1"
    })
    
    # ─────────────────────────────────────────────────────────────────
    # Safe SMM handling (prevent CF time decoding)
    # ─────────────────────────────────────────────────────────────────
    smm_variable = output_dataset["SMM"]
    
    if np.issubdtype(smm_variable.dtype, np.timedelta64):
        smm_variable = (smm_variable / np.timedelta64(1, "D")).astype("float64")
    else:
        smm_variable = smm_variable.astype("float64")
    
    output_dataset["SMM"] = smm_variable
    output_dataset["SMM"].attrs.pop("units", None)
    output_dataset["SMM"].attrs.update({
        "long_name": f"Random error adjusted soil moisture memory (SMM) for {region_description}",
        "unit": "days",  # Non-CF key on purpose
        "valid_min": 0.0,
        "valid_max": 153.0
    })
    
    # ─────────────────────────────────────────────────────────────────
    # Add global attributes
    # ─────────────────────────────────────────────────────────────────
    output_dataset.attrs.update({
        "Institution": "your_institution_here",
        "Contact": "your_name_and_email_here",
        "Description": (
            f"Random error adjusted soil moisture memory (SMM), relative measurement error (δ/σ), "
            f"variance of observed soil moisture random error (δ²), and variance of observed soil moisture (σ²) "
            f"derived from Climate Change Initiative (CCI) soil moisture data. "
            f"Region: {region_description}"
        ),
        "Reference": "your_reference_here",
        "Creation date": "2025"
    })
    
    # ─────────────────────────────────────────────────────────────────
    # Save to NetCDF
    # ─────────────────────────────────────────────────────────────────
    encoding = {var: {'zlib': True, 'complevel': 1} for var in output_dataset.data_vars}
    encoding["SMM"].pop("units", None)  # Extra guard
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    output_dataset.to_netcdf(output_file_path, encoding=encoding, engine='netcdf4')
    
    logging.info(f"✅ Saved to {output_file_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """
    Main execution function orchestrating data loading and processing
    for the Northern region (23°N-60°N) during MJJAS season.
    
    Processes CCI soil moisture residuals to compute:
      - Soil Moisture Memory (SMM): Exponential decay time scale (days)
      - Error correction factor (α): Used for removing measurement error
      - Relative error (δ/σ): Ratio of error std to observed std
      - Error variance (δ²): Variance of measurement error
      - Observed variance (σ²): Variance of observed soil moisture
    """
    logging.info("=" * 80)
    logging.info("Soil Moisture Memory and Error Parameters Analysis")
    logging.info("=" * 80)
    
    # ════════════════════════════════════════════════════════════════════
    # USER CONFIGURATION: Modify these paths and parameters for your data
    # ════════════════════════════════════════════════════════════════════
    
    # Seasonal input files (CCI soil moisture residuals)
    SEASON_INPUT_FILES = {
        'DJF': './data/residuals/sm_DJF.nc',
        'MAM': './data/residuals/sm_MAM.nc',
        'JJA': './data/residuals/sm_JJA.nc',
        'SON': './data/residuals/sm_SON.nc'
    }
    
    # Output configuration
    OUTPUT_DIRECTORY = './output/memory_parameters'
    OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, 'SMM_CCI_0.25.nc')
    
    # Regional parameters - Northern Region (23°N-60°N), MJJAS
    LATITUDE_MIN = 60
    LATITUDE_MAX = 23
    MONTHS_SELECTED = [5, 6, 7, 8, 9]  # May-September (MJJAS)
    REGION_DESCRIPTION = "May-September (MJJAS) over 23°N-60°N Northern region"
    
    # ════════════════════════════════════════════════════════════════════
    # Initialize Dask cluster
    # ════════════════════════════════════════════════════════════════════
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=16,
        memory_limit="128GB"
    )
    client = Client(cluster)
    logging.info(f"Dask cluster initialized")
    logging.info(f"Dashboard: {client.dashboard_link}")
    
    try:
        # ────────────────────────────────────────────────────────────────
        # Load and process data
        # ────────────────────────────────────────────────────────────────
        start_time = perf_counter()
        
        with ProgressBar():
            logging.info(f"Loading regional data: {REGION_DESCRIPTION}")
            dataset = load_regional_data(
                SEASON_INPUT_FILES,
                LATITUDE_MIN,
                LATITUDE_MAX,
                MONTHS_SELECTED
            )
            
            logging.info(f"Analyzing and saving results")
            analyze_and_save_results(
                dataset,
                OUTPUT_FILE,
                REGION_DESCRIPTION
            )
        
        elapsed_time = perf_counter() - start_time
        logging.info("=" * 80)
        logging.info(f"✅ Processing completed in {elapsed_time:.1f} seconds")
        logging.info("=" * 80)
    
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
    
    finally:
        # Cleanup
        client.close()
        cluster.close()
        gc.collect()
        logging.info("Dask resources cleaned up")


if __name__ == "__main__":
    main()