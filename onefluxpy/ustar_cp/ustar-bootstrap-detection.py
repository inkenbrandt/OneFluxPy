from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

@dataclass
class ChangePointResults:
    """Results from change point detection analysis"""
    cp2: np.ndarray  # Change points from 2-parameter model
    cp3: np.ndarray  # Change points from 3-parameter model
    stats2: List[List[List['CPDStatistics']]]  # Statistics for 2-param model
    stats3: List[List[List['CPDStatistics']]]  # Statistics for 3-param model

@dataclass
class CPDStatistics:
    """Statistics from change point detection"""
    n: int = 0
    cp: float = np.nan
    fmax: float = np.nan
    p: float = np.nan
    b0: float = np.nan
    b1: float = np.nan
    b2: float = np.nan
    c2: float = np.nan
    cib0: float = np.nan
    cib1: float = np.nan
    cic2: float = np.nan
    mean_time: float = np.nan
    time_start: float = np.nan
    time_end: float = np.nan
    ustar_t_corr: float = np.nan
    ustar_t_pval: float = np.nan
    mean_temp: float = np.nan
    ci_temp: float = np.nan

def bootstrap_ustar_threshold(time: np.ndarray,
                            nee: np.ndarray, 
                            ustar: np.ndarray,
                            temp: np.ndarray,
                            night_mask: np.ndarray,
                            n_boot: int = 1000,
                            plot: bool = False,
                            site_year: str = '') -> ChangePointResults:
    """
    Estimate u* threshold uncertainty using change-point detection with bootstrapping.
    
    Args:
        time: Array of timestamps
        nee: Net Ecosystem Exchange measurements
        ustar: Friction velocity measurements
        temp: Temperature measurements
        night_mask: Boolean mask for nighttime periods
        n_boot: Number of bootstrap iterations
        plot: Whether to create diagnostic plots
        site_year: Site and year identifier for plots
        
    Returns:
        ChangePointResults object containing change points and statistics
    """
    # Initialize parameters based on time resolution
    time_step = np.nanmedian(np.diff(time))
    points_per_day = round(1 / time_step)
    
    if points_per_day == 24:  # Hourly data
        points_per_bin = 3
    else:  # Half-hourly data
        points_per_bin = 5
        
    n_windows = 4
    n_temp_strata = 4
    n_bins = 50
    points_per_window = n_temp_strata * n_bins * points_per_bin
    
    # Remove invalid u* values
    ustar = np.where((ustar < 0) | (ustar > 3), np.nan, ustar)
    
    # Get valid nighttime data
    night_idx = np.where(night_mask)[0]
    valid = ~np.isnan(nee[night_idx] + ustar[night_idx] + temp[night_idx])
    night_idx = night_idx[valid]
    n_night = len(night_idx)
    
    min_points = n_windows * points_per_window
    if n_night < min_points:
        return ChangePointResults(
            cp2=np.full((n_windows, n_temp_strata, n_boot), np.nan),
            cp3=np.full((n_windows, n_temp_strata, n_boot), np.nan),
            stats2=[[[] for _ in range(n_temp_strata)] for _ in range(n_windows)],
            stats3=[[[] for _ in range(n_temp_strata)] for _ in range(n_windows)]
        )
    
    # Initialize outputs
    cp2 = np.full((n_windows, n_temp_strata, n_boot), np.nan)
    cp3 = np.full((n_windows, n_temp_strata, n_boot), np.nan)
    stats2 = [[[] for _ in range(n_temp_strata)] for _ in range(n_windows)]
    stats3 = [[[] for _ in range(n_temp_strata)] for _ in range(n_windows)]
    
    # Run bootstrap iterations
    for i_boot in range(n_boot):
        # Bootstrap sample
        boot_idx = np.random.randint(0, len(time), len(time))
        boot_time = time[boot_idx]
        boot_nee = nee[boot_idx]
        boot_ustar = ustar[boot_idx]
        boot_temp = temp[boot_idx]
        boot_night = night_mask[boot_idx]
        
        # Get valid nighttime data for this sample
        boot_night_idx = np.where(boot_night)[0]
        valid = ~np.isnan(boot_nee[boot_night_idx] + 
                         boot_ustar[boot_night_idx] + 
                         boot_temp[boot_night_idx])
        boot_night_idx = boot_night_idx[valid]
        
        if len(boot_night_idx) < min_points:
            continue
            
        # Process moving windows
        window_size = points_per_window
        increment = window_size // 2
        
        for window in range(n_windows):
            # Get window indices
            start_idx = window * increment
            end_idx = start_idx + window_size
            if end_idx > len(boot_night_idx):
                end_idx = len(boot_night_idx)
                start_idx = end_idx - window_size
                
            window_idx = boot_night_idx[start_idx:end_idx]
            
            # Split by temperature
            temp_bounds = np.percentile(boot_temp[window_idx],
                                      np.linspace(0, 100, n_temp_strata + 1))
            
            for temp_class in range(n_temp_strata):
                # Get temperature class data
                temp_mask = ((boot_temp[window_idx] >= temp_bounds[temp_class]) & 
                           (boot_temp[window_idx] <= temp_bounds[temp_class + 1]))
                temp_idx = window_idx[temp_mask]
                
                if len(temp_idx) < n_bins * points_per_bin:
                    continue
                    
                # Bin the data
                ustar_bins, nee_bins = _bin_data(
                    boot_ustar[temp_idx],
                    boot_nee[temp_idx],
                    points_per_bin
                )
                
                # Find change points
                cp2_result, cp3_result = _find_change_points(
                    ustar_bins, nee_bins,
                    plot=(plot and i_boot == 0), 
                    title=f"{site_year} W{window}T{temp_class}"
                )
                
                # Store results
                cp2[window, temp_class, i_boot] = cp2_result.cp
                cp3[window, temp_class, i_boot] = cp3_result.cp
                
                if i_boot == 0:
                    stats2[window][temp_class].append(_compute_statistics(
                        cp2_result, boot_time[temp_idx], boot_temp[temp_idx],
                        boot_ustar[temp_idx]
                    ))
                    stats3[window][temp_class].append(_compute_statistics(
                        cp3_result, boot_time[temp_idx], boot_temp[temp_idx],
                        boot_ustar[temp_idx]
                    ))
    
    return ChangePointResults(cp2, cp3, stats2, stats3)

def _bin_data(x: np.ndarray, y: np.ndarray, 
             points_per_bin: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bin x,y data using specified points per bin"""
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    n_bins = len(x) // points_per_bin
    x_means = np.array([np.mean(x_sorted[i:i+points_per_bin])
                       for i in range(0, len(x_sorted)-points_per_bin, points_per_bin)])
    y_means = np.array([np.mean(y_sorted[i:i+points_per_bin])
                       for i in range(0, len(y_sorted)-points_per_bin, points_per_bin)])
                       
    return x_means, y_means

def _find_change_points(x: np.ndarray, y: np.ndarray,
                      plot: bool = False,
                      title: str = '') -> Tuple['ChangePointResult', 'ChangePointResult']:
    """Find change points using 2-parameter and 3-parameter models"""
    from change_point_detection import find_change_point
    return find_change_point(x, y, plot, title)

def _compute_statistics(result: 'ChangePointResult',
                      time: np.ndarray,
                      temp: np.ndarray,
                      ustar: np.ndarray) -> CPDStatistics:
    """Compute additional statistics for change point results"""
    stats = result.stats
    
    # Add timing info
    stats.mean_time = np.mean(time)
    stats.time_start = np.min(time)
    stats.time_end = np.max(time)
    
    # Add temperature stats
    stats.mean_temp = np.mean(temp)
    stats.ci_temp = 0.5 * np.diff(np.percentile(temp, [2.5, 97.5]))
    
    # Add u*-temperature correlation
    if len(ustar) > 1:
        correlation = np.corrcoef(ustar, temp)
        stats.ustar_t_corr = correlation[0,1]
        # Approximate p-value for correlation
        t_stat = stats.ustar_t_corr * np.sqrt((len(ustar)-2)/(1-stats.ustar_t_corr**2))
        stats.ustar_t_pval = 2 * (1 - stats.t.cdf(abs(t_stat), len(ustar)-2))
    
    return stats
