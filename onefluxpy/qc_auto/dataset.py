from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Constants
INVALID_VALUE = -9999.0
BUFFER_SIZE = 1024
TIMESTAMP_HEADER = "TIMESTAMP_START,TIMESTAMP_END"
TIMESTAMP_START_STRING = "TIMESTAMP_START"
TIMESTAMP_END_STRING = "TIMESTAMP_END"
TIMESTAMP_STRING = "TIMESTAMP"

# Variable names - same order as in original enum
VAR_NAMES = [
    'CO2', 'H2O', 'ZL', 'FC', 'FC_SSITC_TEST',  # QcFc
    'H', 'H_SSITC_TEST',  # QcH
    'LE', 'LE_SSITC_TEST',  # QcLE
    'USTAR',
    'TR', 'SB', 'SC', 'SLE',  # SW
    'SH', 'P', 'SW_OUT', 'SW_IN', 'NETRAD', 'SW_DIF',
    'PPFD_IN', 'APAR', 'TA', 'PA',
    'T_CANOPY', 'T_BOLE', 'TS', 'SWC', 'G', 'RH',
    'WD', 'WS', 'TAU', 'LW_IN', 'NEE', 'VPD',
    'itpVPD', 'itpSW_IN', 'itpPPFD_IN', 'itpTA',
    'itpTS', 'itpSWC', 'itpP', 'itpRH',
    'FETCH_FILTER',  # QcFoot
    'SWIN_ORIGINAL', 'PPFD_ORIGINAL', 'FCSTOR', 'FCSTORTT',
    'HEIGHT', 'SW_IN_POT', 'NEE_SPIKE', 'LE_SPIKE',
    'H_SPIKE', 'NIGHT', 'DAY', 'SC_NEGLES', 'TEMP'
]

# Flag variable names - same order as in original enum
VAR_FLAG_NAMES = [
    'NEE', 'USTAR', 'MARGINAL_NEE', 'MARGINAL_LE', 'MARGINAL_H',
    'SW_IN_FROM_PPFD', 'SW_IN_VS_SW_IN_POT', 'PPFD_IN_VS_SW_IN_POT',
    'SW_IN_VS_PPFD', 'SPIKE_NEE', 'SPIKE_LE', 'SPIKE_H',
    'QC2_NEE', 'QC2_LE', 'QC2_H'
]

@dataclass
class TimeZone:
    """Class representing a timezone change"""
    timestamp: datetime
    offset: int

@dataclass 
class DatasetDetails:
    """Class containing dataset metadata"""
    site: str
    year: int
    lat: float
    lon: float
    timeres: str
    time_zones: List[TimeZone]
    time_zones_count: int
    htower: List[float]
    sc_negles: List[Tuple[datetime, int]]
    sc_negles_count: int

@dataclass
class Dataset:
    """Main class for handling flux datasets"""
    details: DatasetDetails
    header: List[str]
    columns_count: int
    rows: np.ndarray  
    rows_count: int
    flags: np.ndarray
    flags_count: int
    missings: np.ndarray
    meteora: Optional[np.ndarray]
    has_timestamp_start: bool
    has_timestamp_end: bool
    leap_year: bool

    def get_var_index(self, var_name: str) -> int:
        """Get index of variable in dataset"""
        # Prevent errors for special cases
        if var_name in ['SWC', 'TS']:
            raise ValueError("get_var_index must not be used on SWC and TS!")
            
        for i, header in enumerate(self.header):
            if header.upper() == var_name.upper():
                return i
        return -1

    def get_flag_index(self, flag_name: str) -> int:
        """Get index of flag in dataset"""
        for i, flag in enumerate(VAR_FLAG_NAMES):
            if flag.upper() == flag_name.upper():
                return i
        return -1

    def get_var_indexes(self, var_name: str) -> Tuple[List[int], int]:
        """Get all indexes for a variable, including profile numbers"""
        indexes = []
        for i, header in enumerate(self.header):
            if header.upper() == var_name.upper():
                indexes.append(i)
        return indexes, len(indexes)
                
def parse_dataset_details(file) -> Optional[DatasetDetails]:
    """Parse dataset details from file header"""
    try:
        # Read header section
        header_lines = []
        while True:
            line = file.readline().strip()
            if not line or line.startswith(TIMESTAMP_STRING):
                break
            header_lines.append(line)

        # Parse required fields
        site = next(line.split(',')[1] for line in header_lines if line.startswith('site,'))
        year = int(next(line.split(',')[1] for line in header_lines if line.startswith('year,')))
        lat = float(next(line.split(',')[1] for line in header_lines if line.startswith('lat,')))
        lon = float(next(line.split(',')[1] for line in header_lines if line.startswith('lon,')))
        timeres = next(line.split(',')[1] for line in header_lines if line.startswith('timeres,'))

        # Parse timezone changes
        time_zones = []
        for line in header_lines:
            if line.startswith('timezone,'):
                parts = line.split(',')
                tz = TimeZone(
                    timestamp=datetime.strptime(parts[1], '%Y-%m-%d %H:%M'),
                    offset=int(parts[2])
                )
                time_zones.append(tz)

        # Parse tower heights
        htower = []
        for line in header_lines:
            if line.startswith('htower,'):
                htower.append(float(line.split(',')[1]))

        # Parse SC negles periods
        sc_negles = []
        for line in header_lines:
            if line.startswith('sc_negl,'):
                parts = line.split(',')
                sc_negles.append((
                    datetime.strptime(parts[1], '%Y-%m-%d %H:%M'),
                    int(parts[2])
                ))

        return DatasetDetails(
            site=site,
            year=year,
            lat=lat,
            lon=lon,
            timeres=timeres,
            time_zones=time_zones,
            time_zones_count=len(time_zones),
            htower=htower,
            sc_negles=sc_negles,
            sc_negles_count=len(sc_negles)
        )

    except Exception as e:
        print(f"Error parsing dataset details: {e}")
        return None

def import_dataset(filename: str) -> Optional[Dataset]:
    """Import dataset from file"""
    try:
        with open(filename) as f:
            # Parse details section
            details = parse_dataset_details(f)
            if not details:
                return None

            # Read data section into pandas
            df = pd.read_csv(f)
            
            # Create Dataset object
            dataset = Dataset(
                details=details,
                header=list(df.columns),
                columns_count=len(df.columns),
                rows=df.values,
                rows_count=len(df),
                flags=np.zeros((len(df), len(VAR_FLAG_NAMES))),
                flags_count=len(VAR_FLAG_NAMES),
                missings=np.count_nonzero(df == INVALID_VALUE, axis=0),
                meteora=None,
                has_timestamp_start=TIMESTAMP_START_STRING in df.columns,
                has_timestamp_end=TIMESTAMP_END_STRING in df.columns,
                leap_year=details.year % 4 == 0
            )

            return dataset

    except Exception as e:
        print(f"Error importing dataset: {e}")
        return None
