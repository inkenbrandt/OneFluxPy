#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

# Constants
PROGRAM_VERSION = "v1.01"
INVALID_VALUE = -9999.0

# Default parameters
MARGINALS_WINDOW = 15
MARGINALS_WINDOW_HOURLY = 8
SWIN_CHECK = 50.0
RPOT_CHECK = 200.0 
SWIN_LIMIT = 0.15
RADIATION_CHECK = 11000
RADIATION_CHECK_STDDEV = 5.0
SWINVSPPFD_THRESHOLD = 10.0
USTAR_CHECK = 9000
USTAR_CHECK_STDDEV = 4.0
SPIKES_WINDOW = 15
SPIKES_WINDOW_HOURLY = 8
SPIKE_CHECK_1 = 2.5
SPIKE_CHECK_2 = 4.0
SPIKE_CHECK_3 = 6.0
SPIKE_CHECK_1_RETURN = 1
SPIKE_CHECK_2_RETURN = 2  
SPIKE_CHECK_3_RETURN = 3
SPIKE_THRESHOLD_NEE = 6.0
SPIKE_THRESHOLD_LE = 100.0
SPIKE_THRESHOLD_H = 100.0

@dataclass
class Dataset:
    """Class representing a flux dataset"""
    site: str
    year: int
    timeres: str
    rows: np.ndarray
    flags: np.ndarray
    header: List[str]
    details: dict
    missings: np.ndarray
    meteora: Optional[np.ndarray] = None
    
class CompareType(IntEnum):
    EQUAL = 0
    GREATER = 1
    LESS = 2
    GREATER_EQUAL = 3
    LESS_EQUAL = 4

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='QC Auto - Flux data quality control')
    
    # Input/output paths
    parser.add_argument('--input_path', help='Input file or directory path')
    parser.add_argument('--output_path', help='Output directory path')
    
    # Processing parameters
    parser.add_argument('--marginals_window', type=int, default=MARGINALS_WINDOW,
                      help='Size of window for marginals')
    parser.add_argument('--sw_in_check', type=float, default=SWIN_CHECK,
                      help='Value for SW_IN check')
    parser.add_argument('--sw_in_pot_check', type=float, default=RPOT_CHECK,
                      help='Value for SW_IN_POT check')
    parser.add_argument('--sw_in_limit', type=float, default=SWIN_LIMIT,
                      help='Value for SW_IN limit') 
    parser.add_argument('--radiation_check', type=int, default=RADIATION_CHECK,
                      help='Value for radiation check')
    parser.add_argument('--radiation_check_stddev', type=float, default=RADIATION_CHECK_STDDEV,
                      help='Standard deviation for radiation check')
    parser.add_argument('--sw_in_vs_ppfd_in_threshold', type=float, default=SWINVSPPFD_THRESHOLD,
                      help='Threshold value for SW_IN vs PPFD_IN')
    parser.add_argument('--spikes_window', type=int, default=SPIKES_WINDOW,
                      help='Size of window for spikes')
    parser.add_argument('--qc2_filter', action='store_true',
                      help='Enable QC2 filter')
    parser.add_argument('--no_spike_filter', action='store_true',
                      help='Disable spike filtering for NEE, H and LE')
    parser.add_argument('--doy', type=int, help='Custom day of year for solar noon')
    
    # Output formats
    parser.add_argument('--db', action='store_true', help='Create database output')
    parser.add_argument('--graph', action='store_true', help='Create graph output')
    parser.add_argument('--ustar', action='store_true', help='Create u* output')
    parser.add_argument('--nee', action='store_true', help='Create NEE output')
    parser.add_argument('--energy', action='store_true', help='Create energy output')
    parser.add_argument('--meteo', action='store_true', help='Create meteo output')
    parser.add_argument('--sr', action='store_true', help='Create SR output')
    parser.add_argument('--solar', action='store_true', help='Create solar output')
    parser.add_argument('--all', action='store_true', help='Create all outputs except SR and solar')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.db, args.graph, args.ustar, args.nee, args.energy, 
                args.meteo, args.sr, args.solar, args.all]):
        parser.error("No output format specified")
        
    if args.all:
        args.db = args.graph = args.ustar = args.nee = args.energy = args.meteo = True
        
    if args.solar:
        args.graph = True
        
    return args

def import_dataset(filename: str) -> Optional[Dataset]:
    """Import dataset from file"""
    try:
        # Load data
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        
        # Parse header
        with open(filename) as f:
            header = f.readline().strip().split(',')
            
        # Create Dataset object
        dataset = Dataset(
            site = os.path.basename(filename).split('_')[0],
            year = int(os.path.basename(filename).split('_')[-1].split('.')[0]),
            timeres = 'HALFHOURLY' if data.shape[0] == 17520 else 'HOURLY',
            rows = data,
            flags = np.zeros_like(data),
            header = header,
            details = {}, # Parse from file
            missings = np.count_nonzero(data == INVALID_VALUE, axis=0)
        )
        
        return dataset
        
    except Exception as e:
        print(f"Error importing dataset: {e}")
        return None

def set_marginals(dataset: Dataset, window: int) -> bool:
    """Set marginal flags for NEE, LE and H"""
    for var in ['NEE', 'LE', 'H']:
        try:
            col = dataset.header.index(var)
        except ValueError:
            continue
            
        flag_col = dataset.header.index(f'FLAG_MARGINAL_{var}')
        
        # Find isolated points
        for i in range(1, len(dataset.rows)-1):
            if (np.isnan(dataset.rows[i-1,col]) and
                not np.isnan(dataset.rows[i,col]) and
                np.isnan(dataset.rows[i+1,col])):
                dataset.flags[i,flag_col] = 1
                
        # Find pairs of isolated points
        for i in range(1, len(dataset.rows)-2):
            if (np.isnan(dataset.rows[i-1,col]) and
                not np.isnan(dataset.rows[i,col]) and
                not np.isnan(dataset.rows[i+1,col]) and
                np.isnan(dataset.rows[i+2,col])):
                dataset.flags[i:i+2,flag_col] = 1
                
    return True

def set_spikes(dataset: Dataset, var_col: int, threshold: float, flag_col: int,
               result: int, window: int) -> bool:
    """Detect spikes in variable using moving window"""
    
    # Get day/night periods
    night = dataset.header.index('NIGHT')
    day = dataset.header.index('DAY')
    
    for period in [night, day]:
        # Copy values
        vals = dataset.rows[:,var_col].copy()
        vals[dataset.rows[:,period] != 1] = np.nan
        
        # Process windows
        for i in range(0, len(vals), window):
            window_vals = vals[i:i+window]
            if len(window_vals) < 3:
                continue
                
            # Calculate differences
            diffs = np.diff(window_vals, 2)
            
            # Get median and MAD
            med = np.nanmedian(diffs)
            mad = np.nanmedian(np.abs(diffs - med))
            
            # Find spikes
            thresh = med + threshold * mad/0.6745
            spikes = np.where((diffs > thresh) | (diffs < -thresh))[0]
            
            # Set flags
            dataset.flags[i+spikes+1,flag_col] = result
            
    return True

def main():
    """Main program entry point"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Show banner
    print(f"\nQC Auto {PROGRAM_VERSION}")
    print("Python version by Claude")
    print(f"(built on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')})")
    
    # Get input/output paths
    input_path = args.input_path or os.getcwd()
    output_path = args.output_path or os.getcwd()
    
    # Process files
    files_processed = 0
    files_skipped = 0
    
    for filename in os.listdir(input_path):
        if not filename.endswith('.csv'):
            continue
            
        print(f"Processing {filename}...")
        
        # Import dataset
        dataset = import_dataset(os.path.join(input_path, filename))
        if dataset is None:
            files_skipped += 1
            continue
            
        # Apply QC processing steps
        if not set_marginals(dataset, args.marginals_window):
            files_skipped += 1
            continue
            
        # More processing steps...
            
        # Save outputs
        if args.db:
            save_db_file(dataset, output_path)
        if args.graph:
            save_graph_file(dataset, output_path)
        if args.ustar:
            save_ustar_file(dataset, output_path)
        if args.nee:
            save_nee_file(dataset, output_path)
        if args.energy:
            save_energy_file(dataset, output_path)
        if args.meteo:
            save_meteo_file(dataset, output_path)
        if args.sr:
            save_sr_file(dataset, output_path)
        if args.solar:
            save_solar_file(dataset, output_path)
            
        files_processed += 1
        print("Done\n")
        
    # Show summary
    total = files_processed + files_skipped
    print(f"\n{total} file{'s' if total != 1 else ''} found: "
          f"{files_processed} processed, {files_skipped} skipped.\n")

if __name__ == "__main__":
    main()
