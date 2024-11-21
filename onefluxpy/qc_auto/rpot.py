"""
rpot.py

This module handles calculation of potential solar radiation (RPOT) and
performs radiation quality checks by comparing measured radiation against 
calculated potential values.

Author: Claude
Based on original C code by Alessio Ribeca
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Constants
SWIN_CHECK = 50.0
RPOT_CHECK = 200.0
SWIN_LIMIT = 0.15
INVALID_VALUE = -9999.0

@dataclass
class Dataset:
    """Simplified Dataset class for RPOT functionality"""
    rows: np.ndarray
    flags: np.ndarray
    rows_count: int
    var_names: List[str]
    flag_names: List[str]
    details: 'DatasetDetails'

@dataclass
class DatasetDetails:
    """Dataset metadata required for RPOT calculations"""
    lat: float  # Latitude in degrees
    lon: float  # Longitude in degrees
    timeres: str  # Time resolution ('HOURLY' or 'HALFHOURLY')
    year: int

def get_var_index(dataset: Dataset, var_name: str) -> int:
    """Get index of variable in dataset"""
    try:
        return dataset.var_names.index(var_name)
    except ValueError:
        return -1

def get_flag_index(dataset: Dataset, flag_name: str) -> int:
    """Get index of flag in dataset"""
    try:
        return dataset.flag_names.index(flag_name)
    except ValueError:
        return -1

def solar_declination(doy: int) -> float:
    """
    Calculate solar declination angle for a given day of year.
    
    Args:
        doy: Day of year (1-366)
        
    Returns:
        float: Solar declination in radians
    """
    return 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)

def equation_of_time(doy: int) -> float:
    """
    Calculate equation of time correction for a given day of year.
    
    Args:
        doy: Day of year (1-366)
        
    Returns:
        float: Time correction in hours
    """
    b = 2 * math.pi * (doy - 81) / 364
    return 0.17 * math.sin(4 * math.pi * (doy - 80) / 373) - 0.129 * math.sin(2 * b)

def calculate_rpot(lat: float, lon: float, doy: int, hour: float, solar_noon: int = 0) -> float:
    """
    Calculate potential solar radiation for a given location and time.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        doy: Day of year
        hour: Hour of day (decimal)
        solar_noon: Solar noon offset adjustment
        
    Returns:
        float: Potential solar radiation in W/m2
    """
    # Convert lat to radians
    lat_rad = math.radians(lat)
    
    # Solar calculations
    decl = solar_declination(doy)
    eqtime = equation_of_time(doy)
    
    # Time calculations
    solar_time = hour + eqtime + (lon / 15.0) + solar_noon
    hour_angle = math.radians(15.0 * (solar_time - 12.0))
    
    # Calculate solar elevation angle
    elev = math.asin(math.sin(lat_rad) * math.sin(decl) + 
                     math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    
    if elev <= 0:
        return 0.0
        
    # Calculate potential radiation
    rpot = 1367.0  # Solar constant
    rpot *= math.sin(elev)
    
    # Atmospheric transmission
    rpot *= 0.75  # Bulk atmospheric transmission coefficient
    
    return rpot

def set_rpot(dataset: Dataset) -> bool:
    """
    Calculate and set RPOT values for entire dataset.
    
    Args:
        dataset: Dataset object containing the data
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get RPOT column index
    rpot_idx = get_var_index(dataset, 'RPOT')
    if rpot_idx == -1:
        print("Unable to compute RPOT: column not found")
        return False
        
    # Calculate timestep in hours
    timestep = 0.5 if dataset.details.timeres == 'HALFHOURLY' else 1.0
    
    # Loop through each timestep
    rows_per_day = int(24 / timestep)
    days = dataset.rows_count / rows_per_day
    
    for day in range(int(days)):
        doy = day + 1
        for step in range(rows_per_day):
            hour = step * timestep
            rpot = calculate_rpot(
                dataset.details.lat,
                dataset.details.lon, 
                doy,
                hour
            )
            row = day * rows_per_day + step
            dataset.rows[row, rpot_idx] = rpot
            
    return True

def set_swin_vs_rpot_flag(dataset: Dataset) -> None:
    """
    Check SWin against RPOT and create quality flag.
    
    Flag is set to 1 when:
    - RPOT = 0 and SWin > 50
    - RPOT > 200 and (SWin-RPOT) > 50 and SWin is >15% more than RPOT
    
    Args:
        dataset: Dataset object containing the data
    """
    # Get required indexes
    swin_idx = get_var_index(dataset, 'SW_IN')
    if swin_idx == -1:
        swin_idx = get_var_index(dataset, 'itpSW_IN')
        
    rpot_idx = get_var_index(dataset, 'RPOT')
    flag_idx = get_flag_index(dataset, 'SW_IN_VS_SW_IN_POT')
    
    if -1 in (swin_idx, rpot_idx, flag_idx):
        return
        
    # Check each row
    for i in range(dataset.rows_count):
        swin = dataset.rows[i, swin_idx]
        rpot = dataset.rows[i, rpot_idx]
        
        if not np.isnan(swin):
            diff = swin - rpot
            if diff > 0:
                if math.isclose(rpot, 0.0):
                    if swin > SWIN_CHECK:
                        dataset.flags[i, flag_idx] = 1
                elif (diff > SWIN_CHECK) and (rpot > RPOT_CHECK):
                    if (diff / rpot) > SWIN_LIMIT:
                        dataset.flags[i, flag_idx] = 1
        else:
            dataset.flags[i, flag_idx] = INVALID_VALUE
