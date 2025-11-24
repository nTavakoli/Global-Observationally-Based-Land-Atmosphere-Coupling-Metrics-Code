#!/usr/bin/env python
# coding: utf-8

"""
Soil Moisture Memory and Error Analysis

This script processes seasonal soil moisture anomalies (from seasonal_anomaly_processor.py) 
to compute:
  1. Autocorrelation functions (ACF) for seasonal segments
  2. Exponential decay model fits for ACF values
  3. Decorrelation parameters (α, SMM)
  4. Measurement error metrics (δ/σ, δ², σ²)
  5. NetCDF output with full metadata for publication


Usage:
    python soil_moisture_memory.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import logging
import warnings
from time import perf_counter
from scipy.optimize import curve_fit
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# ──────────────────────────────────────────────────────────────────────────────
# Configure logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

# Suppress warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Model and Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def straight_line_model(x, m, c):
    """
    Linear model for fitting in log space.
    
    Used to fit exponential decay of autocorrelation in log space.
    
    Parameters
    ----------
    x : ndarray
        Independent variable (lag indices)
    m : float
        Slope coefficient
    c : float
        Intercept coefficient
    
    Returns
    -------
    ndarray
        Linear model predictions: m * x + c
    """
    return m * x + c


def find_valid_pairs(ts, lag):
    """
    Find valid (non-NaN) pairs of data points for a given lag within a time series.
    
    Parameters
    ----------
    ts : array-like
        The time series data
    lag : int
        The lag distance between paired values
    
    Returns
    -------
    tuple of ndarray
        (x_vals, y_vals) where y_vals is shifted by the lag,
        with all NaN values excluded
    """
    ts = np.asarray(ts)
    valid_mask = ~np.isnan(ts)
    shifted_mask = np.roll(valid_mask, -lag)
    shifted_mask[-lag:] = False  # Handle edge cases by setting the last 'lag' elements to False
    combined_mask = valid_mask & shifted_mask
    x_vals = ts[combined_mask]
    y_vals = ts[np.where(combined_mask)[0] + lag]
    return x_vals, y_vals


def ACF(time_series_list, max_lag):
    """
    Calculate the Autocorrelation Function (ACF) for multiple time series.
    
    Computes Pearson correlation coefficients for each lag by pooling valid
    pairs across all input time series segments.
    
    Parameters
    ----------
    time_series_list : list of ndarray
        List containing individual time series segments
    max_lag : int
        Maximum lag distance to compute ACF for
    
    Returns
    -------
    list of float
        ACF values for each lag from 1 to max_lag
    
    Notes
    -----
    Missing values (NaN) are excluded from correlation calculations.
    """
    lag_acfs = []
    
    for lag in range(1, max_lag + 1):
        all_x_vals, all_y_vals = [], []
        
        for ts in time_series_list:
            if len(ts) >= lag + 1:
                x_vals, y_vals = find_valid_pairs(ts, lag)
                all_x_vals.append(x_vals)
                all_y_vals.append(y_vals)
        
        if all_x_vals:
            x_vals = np.concatenate(all_x_vals)
            y_vals = np.concatenate(all_y_vals)
            mean_x, mean_y = np.nanmean(x_vals), np.nanmean(y_vals)
            cov = np.sum((x_vals - mean_x) * (y_vals - mean_y))
            var_x = np.sum((x_vals - mean_x) ** 2)
            var_y = np.sum((y_vals - mean_y) ** 2)
            denom = np.sqrt(var_x * var_y)
            acf = cov / denom if denom != 0 else np.nan
        else:
            acf = np.nan
        
        lag_acfs.append(acf)
    
    return lag_acfs


def SMM(fit_results):
    """
    Estimate the Soil Moisture Memory (SMM) parameter from exponential model fits.
    
    Computes the characteristic lag time based on the exponential decay model
    fitted to the autocorrelation function.
    
    Parameters
    ----------
    fit_results : list of tuple
        List of (x_data, y_data, param, covariance) tuples from curve fitting
    
    Returns
    -------
    float
        Memory parameter in days, bounded [0, 90], or NaN if estimation fails
    """
    if fit_results:
        x_data, y_data, param, _ = fit_results[-1]
        slope, intercept = param
        if slope != 0:
            reciprocal_slope = (-1 - intercept) / slope if intercept > 0 else -1 / slope
            if reciprocal_slope < 0:
                return np.nan
            if reciprocal_slope > 90:
                reciprocal_slope = 90
            return reciprocal_slope
    return np.nan


def process_time_series(data, time_values, season):
    """
    Process a single grid point time series to compute alpha and SMM.
    
    This function computes the autocorrelation function for annual segments,
    fits exponential decay models, and estimates the decorrelation parameters.
    
    Parameters
    ----------
    data : array-like
        Soil moisture data for a single grid point
    time_values : array-like
        Corresponding time coordinates
    season : str
        Season identifier ('DJF', 'MAM', 'JJA', or 'SON')
    
    Returns
    -------
    tuple of float
        (average_a, SMM) where:
        - average_a: decorrelation parameter alpha
        - SMM: soil moisture memory parameter
    """
    if np.isnan(data).all():
        return np.nan, np.nan

    da = xr.DataArray(data, coords={'time': time_values}, dims='time')
    da['time'] = pd.to_datetime(da['time'].values)

    # Group by year with season-specific adjustments
    if season == 'DJF':
        # Adjust season_year: December belongs to the next year
        year = da['time.year'].values
        month = da['time.month'].values
        season_year = year + (month == 12).astype(int)
        da.coords['season_year'] = ('time', season_year)
        grouped = da.groupby('season_year')
    else:
        # For other seasons, group by calendar year
        grouped = da.groupby('time.year')

    time_series = [group.values for _, group in grouped]
    
    # Determine the maximum lag based on the longest time series
    max_lag = max(len(ts) for ts in time_series) - 1
    if max_lag < 1:
        return np.nan, np.nan

    # Compute ACF
    acfs = ACF(time_series, max_lag)

    # Fit exponential models for lag ranges 2 through 7
    fit_params = []
    for i in range(2, 8):
        indices = np.arange(1, i + 1)
        y_data = np.array(acfs[:i])
        valid_indices = np.where((y_data > 0) & (~np.isnan(y_data)))[0]
        
        if valid_indices.size < 2:
            continue
        
        try:
            param, param_cov = curve_fit(
                straight_line_model,
                indices[valid_indices],
                np.log(y_data[valid_indices])
            )
            fit_params.append((indices[valid_indices], np.log(y_data[valid_indices]), param, param_cov))
        except Exception as e:
            logging.warning(f"Curve fitting failed for season {season}. Error: {e}")
            fit_params.append(([], [], [np.nan, np.nan], np.nan))

    # Calculate average_c, ensuring it's non-positive
    average_c = np.nanmean([param[1] for _, _, param, _ in fit_params if not np.isnan(param[1])])
    average_a = 1 - np.exp(min(0, average_c)) if not np.isnan(average_c) else np.nan

    # Calculate SMM
    smm_value = SMM(fit_params)

    return average_a, smm_value


# ──────────────────────────────────────────────────────────────────────────────
# Core Processing Function
# ──────────────────────────────────────────────────────────────────────────────
def process_dataset(input_file, output_file, season):
    """
    Compute SMM, δ/σ, σ², and δ² for the requested season and save to NetCDF.
    
    This function orchestrates the complete analysis pipeline:
      1. Load input NetCDF file
      2. Compute decorrelation parameters (α, SMM)
      3. Calculate variance and error metrics
      4. Add metadata and attributes
      5. Save results to output NetCDF file with compression
    
    Parameters
    ----------
    input_file : str
        Path to input NetCDF file containing soil moisture variable 'sm'
    output_file : str
        Path where output NetCDF file will be saved
    season : str
        Season identifier ('DJF', 'MAM', 'JJA', or 'SON')
    """
    logging.info(f"Loading input file: {input_file}")
    ds = xr.open_dataset(input_file)
    sm = ds['sm'].chunk({'lat': 50, 'lon': 50, 'time': -1})

    # ─────────────────────────────────────────────────────────────────────
    # Compute variance of the soil moisture time series (unbiased, ignores NaNs)
    # ─────────────────────────────────────────────────────────────────────
    logging.info("Computing soil moisture variance...")
    var_soil_moisture = sm.var(dim='time', ddof=1, skipna=True)

    # ─────────────────────────────────────────────────────────────────────
    # Compute alpha (internal) and SMM via autocorrelation analysis
    # ─────────────────────────────────────────────────────────────────────
    logging.info("Computing decorrelation parameters...")
    alpha, smm = xr.apply_ufunc(
        process_time_series,
        sm,
        sm['time'],
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
        kwargs={'season': season}
    )

    # ─────────────────────────────────────────────────────────────────────
    # Compute derived quantities
    # ─────────────────────────────────────────────────────────────────────
    logging.info("Computing error variance metrics...")
    delta_sigma = np.sqrt(alpha / (1 + alpha))
    var_error = (alpha / (1 + alpha)) * var_soil_moisture

    # ─────────────────────────────────────────────────────────────────────
    # Combine into output dataset
    # ─────────────────────────────────────────────────────────────────────
    ds_out = xr.Dataset({
        'SMM': smm,
        'delta_sigma': delta_sigma,
        'sigma2': var_soil_moisture,
        'delta2': var_error
    })

    # ─────────────────────────────────────────────────────────────────────
    # Add coordinate attributes (lat/lon)
    # ─────────────────────────────────────────────────────────────────────
    if "lat" in ds_out.coords:
        ds_out["lat"].attrs.update({
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        })
    if "lon" in ds_out.coords:
        ds_out["lon"].attrs.update({
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        })

    # ─────────────────────────────────────────────────────────────────────
    # Add variable attributes
    # ─────────────────────────────────────────────────────────────────────
    ds_out["delta_sigma"].attrs.update({
        "long_name": f"Relative measurement error= SD(SM random error)/SD(SM) for {season}",
        "units": "Dimensionless"
    })
    ds_out["sigma2"].attrs.update({
        "long_name": f"Variance of observed SM for {season}",
        "units": "1"
    })
    ds_out["delta2"].attrs.update({
        "long_name": f"Variance of observed SM random error for {season}",
        "units": "1"
    })
    # NOTE: use non-CF 'unit' (singular) so xarray won't decode to timedelta on read
    ds_out["SMM"].attrs.update({
        "long_name": f"Random error adjusted soil moisture memory for {season}",
        "unit": "days",          # non-CF key on purpose
        "valid_min": 0.0,
        "valid_max": 90.0
    })

    # ─────────────────────────────────────────────────────────────────────
    # Safe SMM pattern (preserve numbers; prevent CF time decoding)
    # ─────────────────────────────────────────────────────────────────────
    SMM = ds_out["SMM"]
    if np.issubdtype(SMM.dtype, np.timedelta64):
        # Convert timedelta to float days without changing the magnitude
        SMM = (SMM / np.timedelta64(1, "D")).astype("float64")
    else:
        SMM = SMM.astype("float64")  # no-op if already float64
    ds_out["SMM"] = SMM

    # Make sure no CF-style 'units' remains that could trigger decoding
    ds_out["SMM"].attrs.pop("units", None)
    ds_out["SMM"].attrs["unit"] = "days"  # keep a human-readable unit

    # ─────────────────────────────────────────────────────────────────────
    # Add file-level (global) attributes with season text
    # ─────────────────────────────────────────────────────────────────────
    season_text = {
        "JJA": "June- August (JJA)",
        "MAM": "March-May (MAM)",
        "DJF": "December-February (DJF)",
        "SON": "September-November (SON)",
    }.get(season, season)

    ds_out.attrs.update({
        "Institution": "George Mason University, Virginia, USA",
        "Contact": "Nazanin Tavakoli (ntavakol@gmu.edu), Paul A Dirmeyer (pdirmeye@gmu.edu)",
        "Description": (
            "Random error adjusted soil moisture memory (SMM), the relative measurement error (δ/σ), "
            "variance of the observed SM random error (δ**2), and variance of the observed SM (σ**2) " 
            "derived from the Climate Change Initiative (CCI) "
            f"for {season_text} from 1979 to 2023."
        ),
        "Reference": "Tavakoli and Dirmeyer",
        "Creation date": "2025"
    })

    # ─────────────────────────────────────────────────────────────────────
    # Configure compression and encoding
    # ─────────────────────────────────────────────────────────────────────
    encoding = {v: {'zlib': True, 'complevel': 1} for v in ds_out.data_vars}
    encoding["SMM"].pop("units", None)  # extra guard

    # ─────────────────────────────────────────────────────────────────────
    # Write to disk
    # ─────────────────────────────────────────────────────────────────────
    logging.info(f"Saving output to: {output_file}")
    ds_out.to_netcdf(output_file, encoding=encoding)
    logging.info(f"Successfully saved results for season {season}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def main(season):
    """
    Execute the complete processing pipeline for a single season.
    
    Parameters
    ----------
    season : str
        Season identifier ('DJF', 'MAM', 'JJA', or 'SON')
    """
    # Configure Dask cluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=16, memory_limit='128GB')
    client = Client(cluster)

    logging.info("=" * 70)
    logging.info(f"Processing season: {season}")
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    logging.info("=" * 70)

    # ════════════════════════════════════════════════════════════════════
    # USER CONFIGURATION: Modify these paths for your data
    # ════════════════════════════════════════════════════════════════════
    input_file = f"your_input_path_here/sm_{season}.nc"

    # Season-specific output paths
    output_paths = {
        "JJA": "your_output_path_here/JJA/SMM_output.nc",
        "MAM": "your_output_path_here/MAM/SMM_output.nc",
        "SON": "your_output_path_here/SON/SMM_output.nc",
        "DJF": "your_output_path_here/DJF/SMM_output.nc",
    }
    output_file = output_paths.get(season, f"your_output_path_here/{season}/SMM_output.nc")

    # ════════════════════════════════════════════════════════════════════
    # Execute processing
    # ════════════════════════════════════════════════════════════════════
    start_time = perf_counter()
    try:
        process_dataset(input_file, output_file, season)
        elapsed_time = perf_counter() - start_time
        logging.info("=" * 70)
        logging.info(f"Processing {season} completed in {elapsed_time:.2f} seconds")
        logging.info("=" * 70)
    except Exception as error:
        logging.error(f"Processing failed for season {season}: {error}", exc_info=True)
        raise
    finally:
        client.close()
        cluster.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    Process all four seasons sequentially.
    """
    for s in ['DJF', 'MAM', 'JJA', 'SON']:
        main(s)
        