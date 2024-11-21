import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd

@dataclass
class Dataset:
    """Simplified dataset class for spike detection"""
    rows: np.ndarray
    flags: np.ndarray
    rows_count: int
    details: 'DatasetDetails' 
    has_timestamp_end: bool
    leap_year: bool

@dataclass
class DatasetDetails:
    """Dataset metadata required for spike detection"""
    timeres: str  # 'HOURLY' or 'HALFHOURLY'
    year: int

def get_var_index(dataset: Dataset, var_name: str) -> int:
    """Get index of a variable in dataset"""
    # Prevent errors for special cases
    if var_name in ['SWC', 'TS']:
        raise ValueError("get_var_index must not be used on SWC and TS!")
        
    try:
        # Note: Simplified example - in practice would need to handle
        # variable naming/lookup based on dataset header structure
        return {'NIGHT': 0, 'DAY': 1, 'TEMP': 2}[var_name]
    except KeyError:
        return -1

def get_flag_index(dataset: Dataset, flag_name: str) -> int:
    """Get index of a flag in dataset"""
    try:
        # Note: Simplified mapping - would need proper flag lookup in practice
        return {'SPIKE_NEE': 0, 'SPIKE_H': 1, 'SPIKE_LE': 2}[flag_name]
    except KeyError:
        return -1

def set_night_and_day(dataset: Dataset) -> bool:
    """
    Set night/day flags based on radiation conditions.
    Night is determined if:
    - RPOT <= 12 or
    - SWIN < 12 or
    - PPFD < 25
    """
    rpot = get_var_index(dataset, "RPOT")
    swin = get_var_index(dataset, "SWIN")
    if swin == -1:
        swin = get_var_index(dataset, "itpSWIN")
    ppfd = get_var_index(dataset, "PPFD") 
    if ppfd == -1:
        ppfd = get_var_index(dataset, "itpPPFD")
    night = get_var_index(dataset, "NIGHT")
    day = get_var_index(dataset, "DAY")
    
    if -1 in [rpot, night, day]:
        return False
        
    # Initialize temporary arrays for night/day calculation
    night_arr = np.full(dataset.rows_count, np.nan)
    day_arr = np.full(dataset.rows_count, np.nan)
    
    # Determine night based on conditions
    for i in range(dataset.rows_count):
        night_arr[i] = np.nan
        day_arr[i] = np.nan
        
        # Check RPOT condition
        if dataset.rows[i][rpot] <= 12.0:
            night_arr[i] = 1
            
        # Check SWIN if available
        if swin != -1 and not np.isnan(dataset.rows[i][swin]):
            if dataset.rows[i][swin] < 12.0:
                night_arr[i] = 1
            else:
                night_arr[i] = np.nan
                
        # Check PPFD if available
        if ppfd != -1 and not np.isnan(dataset.rows[i][ppfd]):
            if dataset.rows[i][ppfd] < 25.0:
                night_arr[i] = 1
            else:
                night_arr[i] = np.nan
                
        # Day is inverse of night
        if np.isnan(night_arr[i]):
            day_arr[i] = 1
            
    # Handle edges by extending +/- 1 row
    dataset.rows[0][night] = np.nan
    if not np.isnan(night_arr[0]) or not np.isnan(night_arr[1]):
        dataset.rows[0][night] = 1
        
    dataset.rows[0][day] = np.nan
    if not np.isnan(day_arr[0]) or not np.isnan(day_arr[1]):
        dataset.rows[0][day] = 1
        
    # Process middle rows
    for i in range(1, dataset.rows_count - 1):
        if (np.isnan(night_arr[i]) and 
            np.isnan(night_arr[i-1]) and 
            np.isnan(night_arr[i+1])):
            dataset.rows[i][night] = np.nan
        else:
            dataset.rows[i][night] = 1
            
        if (np.isnan(day_arr[i]) and
            np.isnan(day_arr[i-1]) and
            np.isnan(day_arr[i+1])):
            dataset.rows[i][day] = np.nan
        else:
            dataset.rows[i][day] = 1
            
    # Handle final row
    dataset.rows[-1][night] = np.nan
    if not np.isnan(dataset.rows[-1][night]) or not np.isnan(dataset.rows[-2][night]):
        dataset.rows[-1][night] = 1
        
    dataset.rows[-1][day] = np.nan  
    if not np.isnan(dataset.rows[-1][day]) or not np.isnan(dataset.rows[-2][day]):
        dataset.rows[-1][day] = 1
        
    return True

def get_median(data: np.ndarray) -> float:
    """Calculate median excluding NaN values"""
    return np.nanmedian(data)

def get_standard_deviation(data: np.ndarray) -> float:
    """Calculate standard deviation excluding NaN values""" 
    return np.nanstd(data)

def set_spikes(dataset: Dataset, var_name: str, zfc: float, 
               flag_name: str, result: int, window: int) -> bool:
    """
    Detect spikes using moving window approach.
    For data before/after gaps where three point differences can't be calculated,
    flag is set to 1 if difference with adjacent point exceeds threshold.
    """
    var_idx = get_var_index(dataset, var_name)
    flag_idx = get_flag_index(dataset, flag_name)
    temp_idx = get_var_index(dataset, "TEMP")
    period_idx = get_var_index(dataset, "NIGHT")
    
    if -1 in [temp_idx, flag_idx, period_idx]:
        return False
        
    if var_idx == -1:
        return True
    
    # Calculate number of windows
    loop = dataset.rows_count // window
    if loop <= 0:
        return False
        
    time_window = dataset.rows_count - ((loop-1) * window)
    
    # Process night and day periods
    for period in range(2):
        # Copy values to temp column 
        for i in range(dataset.rows_count):
            if (np.isnan(dataset.rows[i][period_idx + period]) or 
                np.isnan(dataset.rows[i][var_idx])):
                dataset.rows[i][temp_idx] = np.nan
            else:
                dataset.rows[i][temp_idx] = dataset.rows[i][var_idx]
                
        # Process each window
        time_window = window
        for i in range(loop):
            # Handle last window
            if i == loop - 1:
                time_window = dataset.rows_count - ((loop-1) * window)
                
            # Calculate differences
            diffs = np.full(time_window, np.nan)
            diffs[0] = np.nan
            diffs[-1] = np.nan
            
            for y in range(1, time_window - 1):
                row = y + (i * window)
                vals = [
                    dataset.rows[row-1][temp_idx],
                    dataset.rows[row][temp_idx],
                    dataset.rows[row+1][temp_idx]
                ]
                if any(np.isnan(vals)):
                    continue
                    
                diffs[y] = (vals[1] - vals[0]) - (vals[2] - vals[1])
                
            # Get median of differences
            median = get_median(diffs)
            if np.isnan(median):
                continue
                
            # Calculate median absolute deviation
            mad = np.array([
                abs(x - median) if not np.isnan(x) else np.nan 
                for x in diffs
            ])
            
            med_abs = get_median(mad)
            if np.isnan(med_abs):
                continue
                
            # Set threshold bounds
            max_thresh = median + (zfc * med_abs / 0.6745)
            min_thresh = median - (zfc * med_abs / 0.6745)
            
            # Flag spikes
            for y in range(time_window):
                if not np.isnan(diffs[y]):
                    if diffs[y] > max_thresh or diffs[y] < min_thresh:
                        dataset.flags[y + (i * window)][flag_idx] = result
                        
    return True

def set_spikes_2(dataset: Dataset, var_name: str, flag_name: str, threshold: float):
    """
    Set spike flags for data points before/after gaps.
    Flag is set if difference with adjacent point exceeds threshold.
    """
    var_idx = get_var_index(dataset, var_name)
    flag_idx = get_flag_index(dataset, flag_name)
    
    if flag_idx == -1:
        return
        
    if var_idx == -1:
        return
        
    # Check points before gaps
    for i in range(2, dataset.rows_count):
        if (np.isnan(dataset.rows[i][var_idx]) and
            not np.isnan(dataset.rows[i-1][var_idx]) and  
            not np.isnan(dataset.rows[i-2][var_idx])):
                
            if abs(dataset.rows[i-1][var_idx] - dataset.rows[i-2][var_idx]) > threshold:
                dataset.flags[i-1][flag_idx] = 1
                
    # Check points after gaps
    for i in range(dataset.rows_count - 2):
        if (np.isnan(dataset.rows[i][var_idx]) and
            not np.isnan(dataset.rows[i+1][var_idx]) and
            not np.isnan(dataset.rows[i+2][var_idx])):
                
            if abs(dataset.rows[i+1][var_idx] - dataset.rows[i+2][var_idx]) > threshold:
                dataset.flags[i+1][flag_idx] = 1
                
    # Set invalid flags for missing values
    for i in range(dataset.rows_count):
        if np.isnan(dataset.rows[i][var_idx]):
            dataset.flags[i][flag_idx] = np.nan
