#!/usr/bin/env python
# coding: utf-8

"""
Soil Moisture-Evaporative Fraction Regime Analysis

This script identifies soil moisture regimes and critical breakpoints by:
  1. Fitting multiple piecewise models (dry, transition, wet combinations)
  2. Using Bayesian Information Criterion (BIC) for model selection
  3. Computing regime-specific parameters (wilting point, critical soil moisture)
  4. Applying hierarchical criteria to select optimal breakpoints
  5. Saving results with comprehensive metadata to NetCDF format

Workflow:
  - Input: Time series of soil moisture and evaporative fraction data
  - Models: 001 (constant), 010 (linear), 110 (piecewise dry-transition),
            011 (piecewise transition-wet), 111 (full piecewise)
  - Output: Model selection, regime parameters, and uncertainty metrics

Usage:
    python soil_moisture_regime.py
"""

import os
import glob
import logging
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import optimize
from sklearn.linear_model import LinearRegression
from dask.distributed import Client, LocalCluster
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_rmse(y_true, y_predicted):
    """
    Compute root mean square error between observations and predictions.
    
    Parameters
    ----------
    y_true : ndarray
        Observed values
    y_predicted : ndarray
        Predicted values
    
    Returns
    -------
    float
        Root mean square error
    """
    return np.sqrt(np.mean((y_true - y_predicted) ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────────────────────────────────────
def model_110(theta, alpha_1, theta_w):
    """
    Model 110: Dry and Transition regimes (piecewise linear).
    
    EF = 0                      if theta < theta_w
         alpha_1 * (theta - theta_w)  if theta >= theta_w
    
    Parameters
    ----------
    theta : ndarray
        Soil moisture values
    alpha_1 : float
        Slope coefficient
    theta_w : float
        Wilting point (transition threshold)
    
    Returns
    -------
    ndarray
        Evaporative fraction predictions
    """
    y = np.where(
        theta < theta_w,
        0,
        alpha_1 * (theta - theta_w)
    )
    return y


def model_011(theta, theta_w, delta, alpha_1):
    """
    Model 011: Transition and Wet regimes (piecewise linear with saturation).
    
    EF = alpha_1 * (theta - theta_w)          if theta_w <= theta < theta_star
         alpha_1 * (theta_star - theta_w)     if theta >= theta_star
    
    Parameters
    ----------
    theta : ndarray
        Soil moisture values
    theta_w : float
        Transition start (wilting point)
    delta : float
        Width of transition zone
    alpha_1 : float
        Slope coefficient
    
    Returns
    -------
    ndarray
        Evaporative fraction predictions
    """
    theta_star = theta_w + delta
    c = alpha_1 * (theta_star - theta_w)
    y = np.where(theta < theta_star, alpha_1 * (theta - theta_w), c)
    return y


def model_111(theta, theta_w, delta, alpha_1):
    """
    Model 111: All three regimes (piecewise linear with saturation).
    
    EF = 0                                 if theta < theta_w
         alpha_1 * (theta - theta_w)      if theta_w <= theta < theta_star
         alpha_1 * (theta_star - theta_w) if theta >= theta_star
    
    Parameters
    ----------
    theta : ndarray
        Soil moisture values
    theta_w : float
        Dry-transition threshold (wilting point)
    delta : float
        Width of transition zone
    alpha_1 : float
        Slope coefficient
    
    Returns
    -------
    ndarray
        Evaporative fraction predictions
    """
    theta_star = theta_w + delta
    c = alpha_1 * (theta_star - theta_w)

    y = np.piecewise(
        theta,
        [theta < theta_w,
         (theta_w <= theta) & (theta < theta_star),
         theta >= theta_star],
        [
            lambda t: 0,
            lambda t: alpha_1 * (t - theta_w),
            lambda t: c,
        ],
    )
    return y


# ──────────────────────────────────────────────────────────────────────────────
# Main Grid Point Processing Function
# ──────────────────────────────────────────────────────────────────────────────
def process_grid_point(soil_moisture_data, evaporative_fraction_data, minimum_sample_size):
    """
    Fit multiple soil moisture regimes and select optimal breakpoints for a single grid point.
    
    This function fits five different models (001, 010, 110, 011, 111) to the SM-EF
    relationship, computes BIC for model selection, and applies hierarchical criteria
    to select wilting point (wp) and critical soil moisture (csm) breakpoints.
    
    Parameters
    ----------
    soil_moisture_data : ndarray
        Soil moisture time series
    evaporative_fraction_data : ndarray
        Evaporative fraction time series
    minimum_sample_size : int
        Minimum number of valid samples required for analysis
    
    Returns
    -------
    tuple of float (31 elements)
        Contains: sample size, data quality metrics, model parameters, BIC values,
        RMSE values, and selection flags for all models
    
    Notes
    -----
    Returns 31 values corresponding to different model parameters and quality metrics.
    Returns NaN for all outputs if insufficient data.
    """
    # Remove missing values
    valid_mask = np.isfinite(soil_moisture_data) & np.isfinite(evaporative_fraction_data)
    soil_moisture_data = soil_moisture_data[valid_mask]
    evaporative_fraction_data = evaporative_fraction_data[valid_mask]

    # Check minimum sample size
    sample_count = len(soil_moisture_data)
    if sample_count == 0 or sample_count < minimum_sample_size:
        return (np.nan,) * 31

    # ─────────────────────────────────────────────────────────────────────
    # Data Quality Check: Remove dominant maximum values
    # ─────────────────────────────────────────────────────────────────────
    max_sm_value = np.max(soil_moisture_data)
    max_sm_count = np.sum(soil_moisture_data == max_sm_value)
    fraction_max_sm = (max_sm_count / sample_count) * 100

    if fraction_max_sm >= 50:
        # Refit without maximum values
        mask_without_max = (soil_moisture_data < max_sm_value)
        soil_moisture_data = soil_moisture_data[mask_without_max]
        evaporative_fraction_data = evaporative_fraction_data[mask_without_max]
        sample_count = len(soil_moisture_data)
        
        if sample_count == 0 or sample_count < minimum_sample_size:
            return (np.nan,) * 31

    # ─────────────────────────────────────────────────────────────────────
    # Initialize output placeholders
    # ─────────────────────────────────────────────────────────────────────
    BIC_001 = np.nan
    BIC_010 = np.nan
    BIC_110 = np.nan
    BIC_011 = np.nan
    BIC_111 = np.nan
    
    wp_010 = np.nan
    wp_110 = np.nan
    wp_011 = np.nan
    wp_111 = np.nan
    
    csm_011 = np.nan
    csm_111 = np.nan
    
    alpha_010 = np.nan
    alpha_1_opt_1 = np.nan
    alpha_1_opt_2 = np.nan
    alpha_111 = np.nan
    
    y_mean = np.nan
    c_111 = np.nan
    y_100_bic = np.nan
    
    RMSE_001 = np.nan
    RMSE_010 = np.nan
    RMSE_110 = np.nan
    RMSE_011 = np.nan
    RMSE_111 = np.nan
    RMSE_100 = np.nan

    best_model_bic = np.nan

    # ─────────────────────────────────────────────────────────────────────
    # Model 001: Constant fit (horizontal line)
    # ─────────────────────────────────────────────────────────────────────
    try:
        y_mean = np.mean(evaporative_fraction_data)
        residual_sum_of_squares = np.sum((evaporative_fraction_data - y_mean) ** 2)
        num_parameters = 1
        RMSE_001 = compute_rmse(evaporative_fraction_data, np.full_like(evaporative_fraction_data, y_mean))
        
        if residual_sum_of_squares > 0 and not np.isnan(residual_sum_of_squares):
            BIC_001 = sample_count * np.log(residual_sum_of_squares / sample_count) + num_parameters * np.log(sample_count)
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Model 010: Linear regression (unconstrained)
    # ─────────────────────────────────────────────────────────────────────
    try:
        linear_model = LinearRegression()
        x_data = soil_moisture_data.reshape((-1, 1))
        y_data = evaporative_fraction_data.reshape((-1, 1))
        linear_model.fit(x_data, y_data)
        
        intercept = linear_model.intercept_[0]
        slope = linear_model.coef_[0][0]
        alpha_010 = slope

        if slope > 0:
            wp_010 = -intercept / slope
            if 0.01 <= wp_010 <= 0.3:
                y_predictions = linear_model.predict(x_data)
                residual_sum_of_squares = np.sum((y_data - y_predictions) ** 2)
                RMSE_010 = compute_rmse(y_data, y_predictions)
                num_parameters = 2
                
                if residual_sum_of_squares > 0 and not np.isnan(residual_sum_of_squares):
                    BIC_010 = sample_count * np.log(residual_sum_of_squares / sample_count) + num_parameters * np.log(sample_count)
            else:
                wp_010 = np.nan
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Model 110: Dry-Transition (piecewise)
    # ─────────────────────────────────────────────────────────────────────
    try:
        initial_parameters = [
            1.0,                      # alpha_1
            np.min(soil_moisture_data)  # theta_w
        ]
        fitted_params_110, _ = optimize.curve_fit(
            model_110,
            soil_moisture_data,
            evaporative_fraction_data,
            p0=initial_parameters,
            bounds=([0, np.min(soil_moisture_data)], [np.inf, np.max(soil_moisture_data)]),
        )
        
        alpha_1_opt_1, wp_110 = fitted_params_110
        predictions_110 = model_110(soil_moisture_data, *fitted_params_110)
        residual_sum_of_squares_110 = np.sum((evaporative_fraction_data - predictions_110) ** 2)
        RMSE_110 = compute_rmse(evaporative_fraction_data, predictions_110)
        num_parameters = 2
        
        if residual_sum_of_squares_110 > 0 and not np.isnan(residual_sum_of_squares_110):
            if 0.01 <= wp_110 <= 0.3:
                dry_points = np.sum(soil_moisture_data <= wp_110)
                wet_points = np.sum(soil_moisture_data > wp_110)
                
                if dry_points >= 10 and wet_points >= 10:
                    BIC_110 = sample_count * np.log(residual_sum_of_squares_110 / sample_count) + num_parameters * np.log(sample_count)
                else:
                    BIC_110 = np.nan
            else:
                wp_110 = np.nan
                BIC_110 = np.nan
        else:
            BIC_110 = np.nan
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Model 011: Transition-Wet (piecewise)
    # ─────────────────────────────────────────────────────────────────────
    try:
        initial_parameters = [
            np.min(soil_moisture_data),                                  # theta_w
            (np.max(soil_moisture_data) - np.min(soil_moisture_data)) / 2,  # delta
            1,                                                            # alpha_1
        ]
        fitted_params_011, _ = optimize.curve_fit(
            model_011,
            soil_moisture_data,
            evaporative_fraction_data,
            p0=initial_parameters,
            bounds=(
                [np.min(soil_moisture_data), 1e-6, 0],
                [np.max(soil_moisture_data), np.max(soil_moisture_data) - np.min(soil_moisture_data), np.inf],
            ),
        )
        
        wp_011, delta_011, alpha_1_opt_2 = fitted_params_011
        predictions_011 = model_011(soil_moisture_data, *fitted_params_011)
        RMSE_011 = compute_rmse(evaporative_fraction_data, predictions_011)
        theta_star_011 = wp_011 + delta_011
        csm_011 = theta_star_011
        residual_sum_of_squares_011 = np.sum((evaporative_fraction_data - predictions_011) ** 2)
        num_parameters = 3
        
        if residual_sum_of_squares_011 > 0 and not np.isnan(residual_sum_of_squares_011):
            if 0.01 <= wp_011 <= 0.3:
                wet_points = np.sum(soil_moisture_data > csm_011)
                transition_points_011 = np.sum((soil_moisture_data >= wp_011) & (soil_moisture_data <= csm_011))
                
                if wet_points >= 10 and transition_points_011 >= 10:
                    BIC_011 = sample_count * np.log(residual_sum_of_squares_011 / sample_count) + num_parameters * np.log(sample_count)
                else:
                    BIC_011 = np.nan
            else:
                wp_011 = np.nan
                BIC_011 = np.nan
        else:
            BIC_011 = np.nan
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Model 111: All Regimes (piecewise)
    # ─────────────────────────────────────────────────────────────────────
    try:
        initial_parameters = [
            np.min(soil_moisture_data),                                  # theta_w
            (np.max(soil_moisture_data) - np.min(soil_moisture_data)) / 2,  # delta
            1,                                                            # alpha_1
        ]
        fitted_params_111, _ = optimize.curve_fit(
            model_111,
            soil_moisture_data,
            evaporative_fraction_data,
            p0=initial_parameters,
            bounds=(
                [np.min(soil_moisture_data), 1e-6, 0],
                [np.max(soil_moisture_data), np.max(soil_moisture_data) - np.min(soil_moisture_data), np.inf],
            ),
        )
        
        wp_111, delta_111, alpha_1_111 = fitted_params_111
        alpha_111 = alpha_1_111
        predictions_111 = model_111(soil_moisture_data, *fitted_params_111)
        residual_sum_of_squares_111 = np.sum((evaporative_fraction_data - predictions_111) ** 2)
        RMSE_111 = compute_rmse(evaporative_fraction_data, predictions_111)
        num_parameters = 3
        
        if residual_sum_of_squares_111 > 0 and not np.isnan(residual_sum_of_squares_111):
            if 0.01 <= wp_111 <= 0.3:
                wet_points = np.sum(soil_moisture_data > (wp_111 + delta_111))
                dry_points = np.sum(soil_moisture_data < wp_111)
                transition_points = np.sum((soil_moisture_data >= wp_111) & (soil_moisture_data <= (wp_111 + delta_111)))
                
                if (wet_points >= 10) and (dry_points >= 10) and (transition_points >= 10):
                    BIC_111 = sample_count * np.log(residual_sum_of_squares_111 / sample_count) + num_parameters * np.log(sample_count)
                else:
                    BIC_111 = np.nan
            else:
                wp_111 = np.nan
                BIC_111 = np.nan
        else:
            BIC_111 = np.nan

        c_111 = alpha_1_111 * delta_111
        csm_111 = wp_111 + delta_111

    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Select best model by BIC
    # ─────────────────────────────────────────────────────────────────────
    bic_values = [BIC_001, BIC_011, BIC_111, BIC_010, BIC_110]
    model_codes = [1, 2, 3, 4, 5]  # 1=001, 2=011, 3=111, 4=010, 5=110

    valid_bic = [v for v in bic_values if np.isfinite(v)]
    if valid_bic:
        min_bic = min(valid_bic)
        best_model_bic = model_codes[bic_values.index(min_bic)]

    # ─────────────────────────────────────────────────────────────────────
    # Special category "100" for dry-only regime
    # ─────────────────────────────────────────────────────────────────────
    if best_model_bic == 1:
        condition_mask = (evaporative_fraction_data <= 0.2) & (soil_moisture_data <= 0.1)
        fraction_condition = (np.sum(condition_mask) / sample_count) * 100
        
        if fraction_condition >= 50:
            best_model_bic = 6  # "100" category
            y_100_bic = y_mean
            RMSE_100 = RMSE_001

    # ─────────────────────────────────────────────────────────────────────
    # Hierarchical Selection for wp_selected and csm_selected
    # ─────────────────────────────────────────────────────────────────────
    wp_selected = np.nan
    csm_selected = np.nan
    wp_selection_flag = np.nan
    csm_selection_flag = np.nan

    # BIC difference thresholds
    CSM_THRESHOLD = 10.0
    WP_THRESHOLD = 50.0

    def is_best_model(model_code):
        """Check if model is the best by BIC."""
        return best_model_bic == model_code

    def get_second_best_and_diff(target_model):
        """Get second-best model and BIC difference if target is second."""
        bic_values = []
        model_codes_local = []
        
        if np.isfinite(BIC_001):
            bic_values.append(BIC_001)
            model_codes_local.append(1)
        if np.isfinite(BIC_010):
            bic_values.append(BIC_010)
            model_codes_local.append(4)
        if np.isfinite(BIC_110):
            bic_values.append(BIC_110)
            model_codes_local.append(5)
        if np.isfinite(BIC_011):
            bic_values.append(BIC_011)
            model_codes_local.append(2)
        if np.isfinite(BIC_111):
            bic_values.append(BIC_111)
            model_codes_local.append(3)
            
        if len(bic_values) < 2:
            return False, np.inf
            
        sorted_indices = np.argsort(bic_values)
        best_model_local = model_codes_local[sorted_indices[0]]
        second_best_model = model_codes_local[sorted_indices[1]]
        
        if second_best_model == target_model and best_model_local != target_model:
            bic_diff = abs(bic_values[sorted_indices[0]] - bic_values[sorted_indices[1]])
            return True, bic_diff
        return False, np.inf

    # Apply hierarchical selection logic
    # Step 1: Best == 111
    if np.isfinite(best_model_bic) and best_model_bic == 3:
        if np.isfinite(wp_111):
            wp_selected = wp_111
            wp_selection_flag = 1
        if np.isfinite(csm_111):
            csm_selected = csm_111
            csm_selection_flag = 1
    
    # Step 2: Second == 111
    if not np.isfinite(wp_selected) or not np.isfinite(csm_selected):
        is_second_111, bic_diff_111 = get_second_best_and_diff(3)
        if is_second_111:
            if not np.isfinite(wp_selected) and bic_diff_111 <= WP_THRESHOLD and np.isfinite(wp_111):
                wp_selected = wp_111
                wp_selection_flag = 2
            if not np.isfinite(csm_selected) and bic_diff_111 <= CSM_THRESHOLD and np.isfinite(csm_111):
                csm_selected = csm_111
                csm_selection_flag = 2
    
    # Step 3: Best == 011
    if not np.isfinite(wp_selected) or not np.isfinite(csm_selected):
        if np.isfinite(best_model_bic) and best_model_bic == 2:
            if not np.isfinite(wp_selected) and np.isfinite(wp_011):
                wp_selected = wp_011
                wp_selection_flag = 3
            if not np.isfinite(csm_selected) and np.isfinite(csm_011):
                csm_selected = csm_011
                csm_selection_flag = 3
    
    # Step 4: Second == 011
    if not np.isfinite(wp_selected) or not np.isfinite(csm_selected):
        is_second_011, bic_diff_011 = get_second_best_and_diff(2)
        if is_second_011:
            if not np.isfinite(wp_selected) and bic_diff_011 <= WP_THRESHOLD and np.isfinite(wp_011):
                wp_selected = wp_011
                wp_selection_flag = 4
            if not np.isfinite(csm_selected) and bic_diff_011 <= CSM_THRESHOLD and np.isfinite(csm_011):
                csm_selected = csm_011
                csm_selection_flag = 4
    
    # Step 5: Best == 110 (wp only)
    if not np.isfinite(wp_selected):
        if np.isfinite(best_model_bic) and best_model_bic == 5:
            if np.isfinite(wp_110):
                wp_selected = wp_110
                wp_selection_flag = 5
    
    # Step 6: Second == 110 (wp only)
    if not np.isfinite(wp_selected):
        is_second_110, bic_diff_110 = get_second_best_and_diff(5)
        if is_second_110 and bic_diff_110 <= WP_THRESHOLD and np.isfinite(wp_110):
            wp_selected = wp_110
            wp_selection_flag = 6
    
    # Step 7: Best == 010 (wp only)
    if not np.isfinite(wp_selected):
        if np.isfinite(best_model_bic) and best_model_bic == 4:
            if np.isfinite(wp_010):
                wp_selected = wp_010
                wp_selection_flag = 7
    
    # Step 8: Second == 010 (wp only)
    if not np.isfinite(wp_selected):
        is_second_010, bic_diff_010 = get_second_best_and_diff(4)
        if is_second_010 and bic_diff_010 <= WP_THRESHOLD and np.isfinite(wp_010):
            wp_selected = wp_010
            wp_selection_flag = 8

    # ─────────────────────────────────────────────────────────────────────
    # Return all 31 outputs
    # ─────────────────────────────────────────────────────────────────────
    return (
        sample_count,              # 0
        fraction_max_sm,           # 1
        best_model_bic,            # 2
        wp_selected,               # 3
        csm_selected,              # 4
        wp_010,                    # 5
        wp_110,                    # 6
        wp_011,                    # 7
        wp_111,                    # 8
        csm_011,                   # 9
        csm_111,                   # 10
        alpha_010,                 # 11
        alpha_1_opt_1,             # 12
        alpha_1_opt_2,             # 13
        alpha_111,                 # 14
        y_mean,                    # 15
        c_111,                     # 16
        y_100_bic,                 # 17
        BIC_001,                   # 18
        BIC_010,                   # 19
        BIC_110,                   # 20
        BIC_011,                   # 21
        BIC_111,                   # 22
        RMSE_001,                  # 23
        RMSE_010,                  # 24
        RMSE_110,                  # 25
        RMSE_011,                  # 26
        RMSE_111,                  # 27
        RMSE_100,                  # 28
        wp_selection_flag,         # 29
        csm_selection_flag,        # 30
    )


# ──────────────────────────────────────────────────────────────────────────────
# Regional Processing Function
# ──────────────────────────────────────────────────────────────────────────────
def process_region(soil_moisture_data, evaporative_fraction_data, minimum_sample_size):
    """
    Apply grid point processing to all spatial locations in a region.
    
    This function applies process_grid_point to each lat/lon grid cell using
    xarray's apply_ufunc for efficient parallelization with Dask.
    
    Parameters
    ----------
    soil_moisture_data : xr.DataArray
        Soil moisture time series with dimensions (time, lat, lon)
    evaporative_fraction_data : xr.DataArray
        Evaporative fraction time series with dimensions (time, lat, lon)
    minimum_sample_size : int
        Minimum number of valid samples per grid point
    
    Returns
    -------
    xr.Dataset
        Dataset containing 31 variables with regime parameters and model outputs
    """
    # Configure chunking for Dask
    soil_moisture_data = soil_moisture_data.chunk({'time': -1, 'lat': 100, 'lon': 100})
    evaporative_fraction_data = evaporative_fraction_data.chunk({'time': -1, 'lat': 100, 'lon': 100})

    latitudes = soil_moisture_data['lat']
    longitudes = soil_moisture_data['lon']
    logging.info(f"Processing region with dimensions — lat: {len(latitudes)}, lon: {len(longitudes)}")

    # ─────────────────────────────────────────────────────────────────────
    # Apply processing to all grid points
    # ─────────────────────────────────────────────────────────────────────
    logging.info("Starting computation using xr.apply_ufunc with Dask parallelization")
    processing_results = xr.apply_ufunc(
        process_grid_point,
        soil_moisture_data,
        evaporative_fraction_data,
        kwargs={'minimum_sample_size': minimum_sample_size},
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[]]*31,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]*31,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Assemble results into xarray Dataset
    # ─────────────────────────────────────────────────────────────────────
    output_dataset = xr.Dataset(
        {
            'N': (['lat', 'lon'], processing_results[0].data),
            'fraction_max_SM': (['lat', 'lon'], processing_results[1].data),
            'Best_Model_BIC': (['lat', 'lon'], processing_results[2].data),
            'wp_selected': (['lat', 'lon'], processing_results[3].data),
            'csm_selected': (['lat', 'lon'], processing_results[4].data),
            'wp_010': (['lat', 'lon'], processing_results[5].data),
            'wp_110': (['lat', 'lon'], processing_results[6].data),
            'wp_011': (['lat', 'lon'], processing_results[7].data),
            'wp_111': (['lat', 'lon'], processing_results[8].data),
            'csm_011': (['lat', 'lon'], processing_results[9].data),
            'csm_111': (['lat', 'lon'], processing_results[10].data),
            'alpha_010': (['lat', 'lon'], processing_results[11].data),
            'alpha_110': (['lat', 'lon'], processing_results[12].data),
            'alpha_011': (['lat', 'lon'], processing_results[13].data),
            'alpha_111': (['lat', 'lon'], processing_results[14].data),
            'EFmax_001': (['lat', 'lon'], processing_results[15].data),
            'EFmax_111': (['lat', 'lon'], processing_results[16].data),
            'EFmax_100': (['lat', 'lon'], processing_results[17].data),
            'BIC_001_or_100': (['lat', 'lon'], processing_results[18].data),
            'BIC_010': (['lat', 'lon'], processing_results[19].data),
            'BIC_110': (['lat', 'lon'], processing_results[20].data),
            'BIC_011': (['lat', 'lon'], processing_results[21].data),
            'BIC_111': (['lat', 'lon'], processing_results[22].data),
            'RMSE_001': (['lat', 'lon'], processing_results[23].data),
            'RMSE_010': (['lat', 'lon'], processing_results[24].data),
            'RMSE_110': (['lat', 'lon'], processing_results[25].data),
            'RMSE_011': (['lat', 'lon'], processing_results[26].data),
            'RMSE_111': (['lat', 'lon'], processing_results[27].data),
            'RMSE_100': (['lat', 'lon'], processing_results[28].data),
            'wp_selection_flag': (['lat', 'lon'], processing_results[29].data),
            'csm_selection_flag': (['lat', 'lon'], processing_results[30].data),
        },
        coords={'lat': latitudes, 'lon': longitudes}
    )

    # ─────────────────────────────────────────────────────────────────────
    # Add variable attributes
    # ─────────────────────────────────────────────────────────────────────
    output_dataset['N'].attrs = {
        'long_name': 'Number of valid observations after removing missing and dominant maximum values',
        'units': '1'
    }
    output_dataset['fraction_max_SM'].attrs = {
        'long_name': 'Percentage of maximum soil moisture values in dataset',
        'units': '%'
    }
    output_dataset['Best_Model_BIC'].attrs = {
        'long_name': 'Best model selected by Bayesian Information Criterion',
        'units': '1',
        'flag_values': '1, 2, 3, 4, 5, 6',
        'flag_meanings': '001 constant, 011 transition-wet, 111 full-piecewise, 010 linear, 110 dry-transition, 100 dry-only'
    }
    output_dataset['wp_selected'].attrs = {
        'long_name': 'Selected wilting point (dry-transition threshold)',
        'units': 'cm3/cm3'
    }
    output_dataset['csm_selected'].attrs = {
        'long_name': 'Selected critical soil moisture (transition-wet threshold)',
        'units': 'cm3/cm3'
    }

    # Add attributes for all wilting points
    for model_code in ['010', '110', '011', '111']:
        output_dataset[f'wp_{model_code}'].attrs = {
            'long_name': f'Wilting point from Model {model_code}',
            'units': 'cm3/cm3'
        }

    # Add attributes for critical soil moisture
    for model_code in ['011', '111']:
        output_dataset[f'csm_{model_code}'].attrs = {
            'long_name': f'Critical soil moisture from Model {model_code}',
            'units': 'cm3/cm3'
        }

    # Add attributes for slopes
    for model_code in ['010', '110', '011', '111']:
        output_dataset[f'alpha_{model_code}'].attrs = {
            'long_name': f'Slope coefficient (alpha) from Model {model_code}',
            'units': 'dimensionless'
        }

    # Add attributes for maximum EF values
    output_dataset['EFmax_001'].attrs = {'long_name': 'Constant EF (Model 001)', 'units': 'dimensionless'}
    output_dataset['EFmax_111'].attrs = {'long_name': 'Maximum EF (Model 111)', 'units': 'dimensionless'}
    output_dataset['EFmax_100'].attrs = {'long_name': 'Constant EF (Model 100)', 'units': 'dimensionless'}

    # Add attributes for BIC values
    for model_code in ['001_or_100', '010', '110', '011', '111']:
        output_dataset[f'BIC_{model_code}'].attrs = {
            'long_name': f'Bayesian Information Criterion for Model {model_code}',
            'units': 'dimensionless'
        }

    # Add attributes for RMSE values
    for model_code in ['001', '010', '110', '011', '111', '100']:
        output_dataset[f'RMSE_{model_code}'].attrs = {
            'long_name': f'Root mean square error for Model {model_code}',
            'units': 'dimensionless'
        }

    output_dataset['wp_selection_flag'].attrs = {
        'long_name': 'Selection step for wp_selected',
        'units': '1',
        'flag_meanings': '1: Best=111, 2: 2ndBest=111&|ΔBIC|≤50, 3: Best=011, 4: 2ndBest=011&|ΔBIC|≤50, 5: Best=110, 6: 2ndBest=110&|ΔBIC|≤50, 7: Best=010, 8: 2ndBest=010&|ΔBIC|≤50'
    }
    output_dataset['csm_selection_flag'].attrs = {
        'long_name': 'Selection step for csm_selected',
        'units': '1',
        'flag_meanings': '1: Best=111, 2: 2ndBest=111&|ΔBIC|≤10, 3: Best=011, 4: 2ndBest=011&|ΔBIC|≤10'
    }

    # Add coordinate attributes
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

    return output_dataset


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """
    Main execution function for soil moisture regime analysis.
    
    Orchestrates loading data from multiple regions, applying regime fitting,
    combining results, and saving to NetCDF output.
    """
    logging.info("=" * 80)
    logging.info("Initiating Soil Moisture-Evaporative Fraction Regime Analysis")
    logging.info("=" * 80)

    start_timer = perf_counter()

    # ════════════════════════════════════════════════════════════════════
    # USER CONFIGURATION: Modify these paths for your data
    # ════════════════════════════════════════════════════════════════════
    
    # Soil moisture data file pattern
    soil_moisture_file_pattern = "./data/soil_moisture/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-{year}/*.nc"
    soil_moisture_years = range(1980, 2024)  # Years 1980-2023
    
    # Evaporative fraction data pattern
    evaporative_fraction_file_pattern = "./data/evaporative_fraction/EF_*_GLEAM_v4.1a.nc"
    
    # Output configuration
    output_directory = "./output/regime_analysis"
    output_filename = "Regime_SM_EF_analysis_0.25.nc"
    
    # Temporal range
    start_date = "1980-01-01"
    end_date = "2023-12-31"

    # ════════════════════════════════════════════════════════════════════
    # Initialize Dask cluster
    # ════════════════════════════════════════════════════════════════════
    cluster = None
    client = None
    
    try:
        cluster = LocalCluster(
            n_workers=8,
            threads_per_worker=4,
            memory_limit='32GB',
        )
        client = Client(cluster)
        logging.info("Dask cluster initialized successfully")
        logging.info(f"Dashboard available at: {client.dashboard_link}")

    except Exception as e:
        logging.error(f"Failed to initialize Dask cluster: {e}")
        return

    try:
        # ────────────────────────────────────────────────────────────────
        # Load soil moisture data
        # ────────────────────────────────────────────────────────────────
        logging.info("Collecting soil moisture files")
        sm_file_list = []
        for year in soil_moisture_years:
            pattern = soil_moisture_file_pattern.format(year=year)
            sm_file_list.extend(glob.glob(pattern))

        if not sm_file_list:
            logging.error("No soil moisture files found. Check file pattern.")
            return

        logging.info(f"Loading {len(sm_file_list)} soil moisture files")
        sm_dataset = xr.open_mfdataset(
            sm_file_list,
            combine='by_coords',
            preprocess=lambda x: x.sortby('time'),
            chunks=None,
        )
        logging.info("Soil moisture data loaded successfully")

        # ────────────────────────────────────────────────────────────────
        # Load evaporative fraction data
        # ────────────────────────────────────────────────────────────────
        logging.info("Collecting evaporative fraction files")
        ef_file_list = glob.glob(evaporative_fraction_file_pattern)
        
        if not ef_file_list:
            logging.error("No evaporative fraction files found. Check file pattern.")
            return

        logging.info(f"Loading {len(ef_file_list)} evaporative fraction files")
        ef_dataset = xr.open_mfdataset(
            ef_file_list,
            combine='by_coords',
            preprocess=lambda x: x.sortby('time'),
            chunks=None,
        )
        logging.info("Evaporative fraction data loaded successfully")

        # ────────────────────────────────────────────────────────────────
        # Temporal subsetting
        # ────────────────────────────────────────────────────────────────
        sm_dataset = sm_dataset.sel(time=slice(start_date, end_date))
        ef_dataset = ef_dataset.sel(time=slice(start_date, end_date))
        logging.info(f"Data subset to {start_date} through {end_date}")

        # ────────────────────────────────────────────────────────────────
        # Coordinate alignment
        # ────────────────────────────────────────────────────────────────
        sm_dataset = sm_dataset.assign_coords(
            lat=sm_dataset['lat'].round(4),
            lon=sm_dataset['lon'].round(4)
        )
        ef_dataset = ef_dataset.assign_coords(
            lat=ef_dataset['lat'].round(4),
            lon=ef_dataset['lon'].round(4)
        )
        sm_dataset, ef_dataset = xr.align(sm_dataset, ef_dataset, join='inner')
        logging.info("Datasets aligned")

        # ════════════════════════════════════════════════════════════════
        # Process regions
        # ════════════════════════════════════════════════════════════════
        regional_datasets = []

        # Region 1: Northern (23N-60N, MJJAS months)
        logging.info("Processing Northern region (23°N-60°N, May-September)")
        sm_north = sm_dataset.sel(lat=slice(60, 23))
        ef_north = ef_dataset.sel(lat=slice(60, 23))
        months_north = [5, 6, 7, 8, 9]
        sm_north = sm_north.sel(time=sm_north['time.month'].isin(months_north))
        ef_north = ef_north.sel(time=ef_north['time.month'].isin(months_north))
        ds_north = process_region(sm_north['sm'], ef_north['EF'], minimum_sample_size=670)
        regional_datasets.append(ds_north)

        # Region 2: Tropics (23S-23N, all months)
        logging.info("Processing Tropical region (23°S-23°N, all months)")
        sm_tropics = sm_dataset.sel(lat=slice(23, -23))
        ef_tropics = ef_dataset.sel(lat=slice(23, -23))
        ds_tropics = process_region(sm_tropics['sm'], ef_tropics['EF'], minimum_sample_size=1600)
        regional_datasets.append(ds_tropics)

        # Region 3: Southern (60S-23S, NDJFM months)
        logging.info("Processing Southern region (60°S-23°S, November-March)")
        sm_south = sm_dataset.sel(lat=slice(-23, -60))
        ef_south = ef_dataset.sel(lat=slice(-23, -60))
        months_south = [11, 12, 1, 2, 3]
        sm_south = sm_south.sel(time=sm_south['time.month'].isin(months_south))
        ef_south = ef_south.sel(time=ef_south['time.month'].isin(months_south))
        ds_south = process_region(sm_south['sm'], ef_south['EF'], minimum_sample_size=660)
        regional_datasets.append(ds_south)

        # ────────────────────────────────────────────────────────────────
        # Combine regional datasets
        # ────────────────────────────────────────────────────────────────
        logging.info("Combining regional datasets")
        combined_dataset = xr.combine_by_coords(regional_datasets)
        combined_dataset = combined_dataset.compute()

        # ────────────────────────────────────────────────────────────────
        # Add global metadata
        # ────────────────────────────────────────────────────────────────
        combined_dataset.attrs['model_codes'] = (
            '1: Model_001 (constant), 2: Model_011 (transition-wet), '
            '3: Model_111 (dry-transition-wet), 4: Model_010 (linear), '
            '5: Model_110 (dry-transition), 6: Model_100 (dry-only)'
        )

        combined_dataset.attrs['model_descriptions'] = (
            'Models represent different soil moisture-EF regimes: '
            'Model 100 (dry-only): Data clustered in low EF/low SM; '
            'Model 001 (constant): Horizontal fit; '
            'Model 010 (linear): Linear relationship; '
            'Model 110 (dry-transition): Piecewise with dry and transition zones; '
            'Model 011 (transition-wet): Piecewise with transition and saturation; '
            'Model 111 (full): Piecewise with dry, transition, and wet zones'
        )

        combined_dataset.attrs['selection_criteria'] = (
            'Hierarchical selection for wp_selected and csm_selected based on BIC ranking '
            'and threshold differences (≤50 for wp, ≤10 for csm)'
        )

        combined_dataset.attrs.update({
            "Institution": "your_institution_here",
            "Contact": "your_name_and_email_here",
            "Description": (
                "Soil moisture-evaporative fraction regime identification and critical breakpoints "
                "derived from observational data. "
                "Northern region (23°N-60°N): May-September; "
                "Tropics (23°S-23°N): All months; "
                "Southern region (60°S-23°S): November-March. "
                "Time period: 1980-2023."
            ),
            "Reference": "your_reference_here",
            "Creation date": "2025",
        })

        # ────────────────────────────────────────────────────────────────
        # Save to NetCDF
        # ────────────────────────────────────────────────────────────────
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, output_filename)

        logging.info(f"Saving results to {output_file}")
        encoding = {var: {'zlib': True, 'complevel': 1} for var in combined_dataset.data_vars}
        combined_dataset.to_netcdf(
            output_file,
            encoding=encoding,
            engine='netcdf4',
            format='NETCDF4'
        )
        logging.info("Results saved successfully")

        elapsed_time = perf_counter() - start_timer
        logging.info("=" * 80)
        logging.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
    
    finally:
        # Cleanup Dask resources
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
        logging.info("Dask resources closed")


if __name__ == "__main__":
    main()