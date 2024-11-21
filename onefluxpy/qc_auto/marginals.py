"""
marginals.py

This module handles detection of marginal data defined as single or double isolated points
between invalid data in flux measurements.

Author: Claude
Based on original C code by Alessio Ribeca
"""

from typing import Optional, List
import numpy as np
from dataclasses import dataclass

@dataclass 
class Dataset:
    """
    Simplified Dataset class containing only the required attributes
    for marginal detection functionality
    """
    rows: np.ndarray
    flags: np.ndarray
    rows_count: int
    var_names: List[str]
    flag_names: List[str]

def get_var_index(dataset: Dataset, var_name: str) -> int:
    """Get index of a variable in the dataset."""
    try:
        return dataset.var_names.index(var_name)
    except ValueError:
        return -1

def get_flag_index(dataset: Dataset, flag_name: str) -> int:
    """Get index of a flag in the dataset."""
    try:
        return dataset.flag_names.index(flag_name)
    except ValueError:
        return -1

def set_marginals_on_var(dataset: Dataset, marginals_window: int, 
                        var_index: int, flag_index: int) -> bool:
    """
    Set marginal flags for a specific variable.
    
    Args:
        dataset: Dataset object containing the data
        marginals_window: Window size for detecting marginal sequences
        var_index: Index of the variable to check
        flag_index: Index where flags should be set
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Skip if var_index is invalid
    if var_index == -1:
        return True
        
    # Get TEMP index for temporary calculations
    temp_index = get_var_index(dataset, 'TEMP')
    if temp_index == -1:
        return False
        
    # Copy values to temp column
    dataset.rows[:, temp_index] = dataset.rows[:, var_index]

    # Check for marginals_window or more marginals
    invalid_count = 0
    for i in range(dataset.rows_count):
        if np.isnan(dataset.rows[i, temp_index]):
            invalid_count += 1
        else:
            if invalid_count >= marginals_window:
                start = (i - invalid_count) - 1
                if start >= 0:
                    dataset.flags[start, flag_index] = 1
            invalid_count = 0
            
    # Check end of dataset
    if (invalid_count >= marginals_window) and (invalid_count != dataset.rows_count):
        start = dataset.rows_count - invalid_count - 1
        if start >= 0:
            dataset.flags[start, flag_index] = 1

    # Check for 1 isolated marginal
    for i in range(1, dataset.rows_count - 1):
        if (np.isnan(dataset.rows[i-1, temp_index]) and
            not np.isnan(dataset.rows[i, temp_index]) and
            np.isnan(dataset.rows[i+1, temp_index])):
            dataset.flags[i, flag_index] = 1

    # Check for 2 isolated marginals
    for i in range(1, dataset.rows_count - 2):
        if (np.isnan(dataset.rows[i-1, temp_index]) and
            not np.isnan(dataset.rows[i, temp_index]) and
            not np.isnan(dataset.rows[i+1, temp_index]) and
            np.isnan(dataset.rows[i+2, temp_index])):
            dataset.flags[i, flag_index] = 1
            dataset.flags[i+1, flag_index] = 1

    return True

def set_marginals(dataset: Dataset, marginals_window: int) -> bool:
    """
    Set marginal flags for NEE, LE and H variables.
    
    Args:
        dataset: Dataset object containing the data
        marginals_window: Window size for detecting marginal sequences
    
    Returns:
        bool: True if all operations successful, False otherwise
    """
    # Get variable indexes
    nee = get_var_index(dataset, 'NEE')
    le = get_var_index(dataset, 'LE') 
    h = get_var_index(dataset, 'H')
    
    # Get flag indexes
    nee_flag = get_flag_index(dataset, 'MARGINAL_NEE')
    le_flag = get_flag_index(dataset, 'MARGINAL_LE')
    h_flag = get_flag_index(dataset, 'MARGINAL_H')
    
    if -1 in (nee_flag, le_flag, h_flag):
        return False

    # Compute marginals for each variable
    success = 0
    success += set_marginals_on_var(dataset, marginals_window, nee, nee_flag)
    success += set_marginals_on_var(dataset, marginals_window, le, le_flag)
    success += set_marginals_on_var(dataset, marginals_window, h, h_flag)

    return success == 3
