#!/usr/bin/env python
# coding: utf-8


"""
Seasonal Anomaly Processor

This script processes geospatial NetCDF data by:
  1. Loading and combining multiple input files
  2. Applying time-based filtering
  3. Removing seasonal cycles via harmonic analysis
  4. Splitting results by season and saving to NetCDF format

Usage:
    python seasonal_anomaly_processor.py
"""

import logging
import warnings
import os
import xarray as xr
import numpy as np
from time import perf_counter
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import gc
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings("ignore")


def remove_seasonal_harmonics(time_series, harmonics=[1, 2, 3]):
    """
    Remove seasonal harmonics from a 1D time series via least-squares fitting.

    This function fits multiple harmonic components (annual cycle and its higher
    harmonics) to the time series and returns the residual anomalies.

    Parameters
    ----------
    time_series : ndarray
        1D array containing the input time series data.
    harmonics : list of int, optional
        Harmonic indices to fit. Default is [1, 2, 3] for annual, semi-annual,
        and tertiary annual cycles.

    Returns
    -------
    anomalies : ndarray
        Residual time series after subtracting the fitted seasonal cycle.
        Contains NaN where input was invalid.

    Notes
    -----
    The fitting assumes a 365-day year without leap-year adjustments.
    Missing values (NaN) are excluded from the fit.
    """
    try:
        valid_mask = np.isfinite(time_series)
        if np.sum(valid_mask) < 2:
            return np.full_like(time_series, np.nan)

        valid_data = time_series[valid_mask]
        time_indices = np.arange(1, len(time_series) + 1)[valid_mask]

        # Center the data
        data_mean = np.mean(valid_data)
        data_centered = valid_data - data_mean

        # Angular frequency for annual cycle (365-day year)
        annual_frequency = 2 * np.pi / 365

        # Build design matrix with sine and cosine terms for each harmonic
        design_matrix = np.column_stack([
            func(annual_frequency * harmonic_idx * time_indices)
            for harmonic_idx in harmonics
            for func in (np.sin, np.cos)
        ])

        # Least-squares solution for harmonic coefficients
        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix, data_centered, rcond=None
        )
        seasonal_fit = design_matrix @ coefficients

        # Compute anomalies as residuals
        anomalies_valid = data_centered - seasonal_fit

        # Reconstruct full array with NaN in original missing positions
        anomalies = np.full_like(time_series, np.nan)
        anomalies[valid_mask] = anomalies_valid

        return anomalies

    except Exception as e:
        logging.error(f"Error in remove_seasonal_harmonics: {e}", exc_info=True)
        return np.full_like(time_series, np.nan)


def extract_season(data_array, season_code):
    """
    Extract data for a specific season from a time series.

    Parameters
    ----------
    data_array : xarray.DataArray
        Input data with a time dimension.
    season_code : str
        Seasonal code. Must be one of:
        - 'DJF': December, January, February (boreal winter)
        - 'MAM': March, April, May (boreal spring)
        - 'JJA': June, July, August (boreal summer)
        - 'SON': September, October, November (boreal autumn)

    Returns
    -------
    seasonal_data : xarray.DataArray
        Data subset containing only months relevant to the specified season.

    Raises
    ------
    ValueError
        If season_code is not a recognized seasonal identifier.
    """
    season_months = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    if season_code not in season_months:
        raise ValueError(f"Season code '{season_code}' not recognized. "
                        f"Must be one of {list(season_months.keys())}")

    month_values = season_months[season_code]
    seasonal_data = data_array.sel(
        time=data_array['time'].dt.month.isin(month_values)
    )
    logging.info(f"Extracted {season_code} data.")

    return seasonal_data


def process_geospatial_variable(
    variable_name,
    output_directory,
    file_pattern=None,
    file_paths=None,
    start_date=None,
    end_date=None,
    time_encoding=None,
    compression_settings=None,
    seasons=None,
):
    """
    Process geospatial variable: load, detrend seasonality, split by season.

    This is the main processing routine that orchestrates:
      1. Loading NetCDF files from disk
      2. Temporal filtering
      3. Removal of NaN-only grid cells
      4. Deseasonalization via harmonic fitting
      5. Seasonal decomposition and NetCDF output

    Parameters
    ----------
    variable_name : str
        Name of the variable to process (must exist in input files).
    output_directory : str
        Path where output NetCDF files will be saved.
    file_pattern : str, optional
        Glob pattern matching input NetCDF files.
        Example: '/data/model_output/*/var_*.nc'
    file_paths : list of str, optional
        Explicit list of input file paths.
    start_date : str, optional
        Start date for temporal subsetting (ISO format, e.g., '2000-01-01').
    end_date : str, optional
        End date for temporal subsetting (ISO format).
    time_encoding : dict, optional
        NetCDF time encoding specification.
        Default: {'units': 'days since 1970-01-01', 'calendar': 'gregorian'}
    compression_settings : dict, optional
        NetCDF compression parameters. Default: zlib compression, level 4.
    seasons : list of str, optional
        Seasons to extract. Default: ['DJF', 'MAM', 'JJA', 'SON']

    Raises
    ------
    ValueError
        If neither file_pattern nor file_paths is provided, or if no files match.
    """
    if time_encoding is None:
        time_encoding = {'units': 'days since 1970-01-01', 'calendar': 'gregorian'}
    if compression_settings is None:
        compression_settings = dict(zlib=True, complevel=4)
    if seasons is None:
        seasons = ['DJF', 'MAM', 'JJA', 'SON']

    logging.info(f"Starting processing for variable: {variable_name}")
    os.makedirs(output_directory, exist_ok=True)

    # ====================================================================
    # Step 1: Load Input Data
    # ====================================================================
    try:
        if file_pattern:
            file_list = sorted(glob.glob(file_pattern))
            logging.info(f"Files matching pattern for '{variable_name}': {file_list}")

            if not file_list:
                raise ValueError(f"No files found matching: {file_pattern}")

            logging.info(f"Opening and combining {len(file_list)} file(s)...")
            dataset = xr.open_mfdataset(
                file_list,
                combine='by_coords',
                preprocess=lambda x: x.sortby('time'),
                chunks=None
            )

        elif file_paths:
            logging.info(f"Opening and combining {len(file_paths)} file(s)...")
            dataset = xr.open_mfdataset(
                file_paths,
                combine='by_coords',
                preprocess=lambda x: x.sortby('time'),
                chunks=None
            )

        else:
            raise ValueError("Specify either 'file_pattern' or 'file_paths'.")

        logging.info("Data loaded and combined successfully.")

    except Exception as e:
        logging.error(f"Failed to load files: {e}", exc_info=True)
        raise

    logging.info(f"Time range before filtering: {dataset.time.values[0]} to {dataset.time.values[-1]}")

    # ====================================================================
    # Step 2: Temporal Subsetting
    # ====================================================================
    if start_date and end_date:
        dataset = dataset.sel(time=slice(start_date, end_date))
        logging.info(f"Time range after filtering: {dataset.time.values[0]} to {dataset.time.values[-1]}")

    # ====================================================================
    # Step 3: Configure Dask Chunking
    # ====================================================================
    # Chunk time as single block, spatial dimensions into manageable chunks
    dataset = dataset.chunk({'time': -1, 'lat': 100, 'lon': 100})
    logging.info(f"Dataset dimensions: {dict(dataset.dims)}")

    # ====================================================================
    # Step 4: Remove All-NaN Grid Points
    # ====================================================================
    has_valid_data = dataset[variable_name].notnull().any(dim='time')
    dataset = dataset.where(has_valid_data)
    logging.info("Removed grid cells with no valid data across all times.")

    # ====================================================================
    # Step 5: Deseasonalize via Harmonic Fitting
    # ====================================================================
    logging.info("Removing seasonal cycle via harmonic analysis...")
    anomalies = xr.apply_ufunc(
        remove_seasonal_harmonics,
        dataset[variable_name],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )
    logging.info("Deseasonalization complete.")

    gc.collect()

    # ====================================================================
    # Step 6: Season-Based Splitting and Output
    # ====================================================================
    for season_code in seasons:
        logging.info(f"Processing season: {season_code}")
        season_data = extract_season(anomalies, season_code)

        if season_data.time.size == 0:
            logging.warning(f"No data available for {season_code}, skipping.")
            continue

        output_path = os.path.join(
            output_directory,
            f"{variable_name}_{season_code}.nc"
        )

        # Trigger computation
        with ProgressBar():
            season_data = season_data.compute()

        # Configure NetCDF encoding
        variable_encoding = {season_data.name: compression_settings}
        encoding = {'time': time_encoding}
        encoding.update(variable_encoding)

        try:
            season_data.to_netcdf(
                output_path,
                encoding=encoding,
                engine='netcdf4',
                format='NETCDF4',
            )
            logging.info(f"Saved seasonal anomalies to: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save {season_code} data: {e}", exc_info=True)

        del season_data
        gc.collect()

    # Final cleanup
    del dataset, anomalies
    gc.collect()
    logging.info(f"Processing complete for variable: {variable_name}")


# ========================================================================
# Main Execution
# ========================================================================
if __name__ == "__main__":
    try:
        start_time = perf_counter()
        logging.info("=" * 70)
        logging.info("Initiating geospatial data processing pipeline")
        logging.info("=" * 70)

        # Configure distributed computing cluster
        cluster_settings = {
            "n_workers": 2,
            "threads_per_worker": 16,
            "memory_limit": "128GB",
        }

        with LocalCluster(**cluster_settings) as cluster, Client(cluster) as client:
            logging.info(f"Dask cluster dashboard available at: {client.dashboard_link}")

            # ================================================================
            # USER CONFIGURATION: Modify these parameters for your data
            # ================================================================
            variable_name = "your_variable_name_here"  # e.g., "soil_moisture", "evapotranspiration"
            output_directory = "your_output_directory_here"  # e.g., "./output/anomalies"
            file_pattern = "your_file_pattern_here"  # e.g., "./data/input/*/variable_*.nc"
            start_date = "your_start_date_here"  # Format: "YYYY-MM-DD", e.g., "1980-01-01"
            end_date = "your_end_date_here"  # Format: "YYYY-MM-DD", e.g., "2023-12-31"

            # ================================================================
            # Execute processing
            # ================================================================
            process_geospatial_variable(
                variable_name=variable_name,
                output_directory=output_directory,
                file_pattern=file_pattern,
                start_date=start_date,
                end_date=end_date,
            )

            elapsed_time = perf_counter() - start_time
            logging.info("=" * 70)
            logging.info(f"All processing completed successfully in {elapsed_time:.2f} seconds")
            logging.info("=" * 70)

    except Exception as error:
        logging.error("Pipeline execution failed", exc_info=True)
        raise

