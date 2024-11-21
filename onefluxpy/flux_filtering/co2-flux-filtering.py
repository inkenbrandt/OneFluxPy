"""
CO2 flux filtering modules based on Barr et al. (2013)
Implements u*, σw, and clustering-based filtering approaches
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

@dataclass
class FilterResults:
    """Container for filtering results"""
    threshold: float
    confidence_interval: Tuple[float, float]
    retained_data: np.ndarray
    f_max: float
    mode: str
    percent_selected: float

class ChangePointDetector:
    """Base class for change point detection in flux data"""
    
    def __init__(self, n_boot: int = 1000, significance_level: float = 0.05):
        """
        Args:
            n_boot: Number of bootstrap iterations
            significance_level: Statistical significance level for changepoint detection
        """
        self.n_boot = n_boot
        self.significance_level = significance_level
        
        # F-max critical values from Barr et al. Table 2
        self.f_crit = {
            10: 9.147, 15: 7.877, 20: 7.443, 30: 7.031,
            50: 6.876, 70: 6.888, 100: 6.918, 150: 6.981,
            200: 7.062, 300: 7.201, 500: 7.342, 700: 7.563,
            1000: 7.783
        }

    def _get_f_critical(self, n: int) -> float:
        """Get critical F value for sample size n"""
        keys = np.array(list(self.f_crit.keys()))
        idx = np.searchsorted(keys, n)
        if idx == len(keys):
            return self.f_crit[keys[-1]]
        if idx == 0:
            return self.f_crit[keys[0]]
        return self.f_crit[keys[idx-1]] + \
               (n - keys[idx-1]) * \
               (self.f_crit[keys[idx]] - self.f_crit[keys[idx-1]]) / \
               (keys[idx] - keys[idx-1])

    def detect_changepoint(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Detect changepoint using two-phase regression with zero slope constraint
        
        Args:
            x: Independent variable (e.g. u*)
            y: Dependent variable (e.g. NEE)
            
        Returns:
            changepoint: Detected threshold
            slope: Slope below changepoint 
            f_max: Maximum F statistic
            r_squared: R-squared value
        """
        n = len(x)
        if n < 5:
            return np.nan, np.nan, 0, 0
            
        max_r2 = -np.inf
        best_cp = np.nan
        best_slope = np.nan
        f_max = 0
        
        for i in range(2, n-2):
            x1 = x[:i]
            y1 = y[:i]
            x2 = x[i:]
            y2 = y[i:]
            
            # Fit line below changepoint
            slope, intercept, r1, _, _ = stats.linregress(x1, y1)
            
            # Calculate means above changepoint
            mean_above = np.mean(y2)
            
            # Calculate SSE for split model
            pred1 = slope * x1 + intercept
            pred2 = np.full_like(x2, mean_above)
            sse_split = np.sum((y1 - pred1)**2) + np.sum((y2 - pred2)**2)
            
            # Calculate SSE for null model (single mean)
            sse_null = np.sum((y - np.mean(y))**2)
            
            # Calculate F statistic
            if sse_split > 0:
                f_stat = ((sse_null - sse_split) / 2) / (sse_split / (n - 3))
                
                # Calculate R2
                r2 = 1 - sse_split/sse_null
                
                if r2 > max_r2:
                    max_r2 = r2
                    best_cp = x[i]
                    best_slope = slope
                    f_max = f_stat
                    
        return best_cp, best_slope, f_max, max_r2

class TurbulenceFilter(ChangePointDetector):
    """Implementation of u* and σw filtering methods"""
    
    def filter_data(self, 
                   turbulence: np.ndarray,
                   nee: np.ndarray,
                   temp: np.ndarray,
                   bins_per_window: int = 50,
                   min_pts_per_bin: int = 5) -> FilterResults:
        """
        Filter flux data using turbulence threshold
        
        Args:
            turbulence: Array of turbulence measurements (u* or σw)
            nee: Array of CO2 flux measurements
            temp: Array of temperature measurements
            bins_per_window: Number of bins per analysis window
            min_pts_per_bin: Minimum points required per bin
            
        Returns:
            FilterResults object containing threshold and filtering results
        """
        # Remove nans
        mask = ~np.isnan(turbulence) & ~np.isnan(nee) & ~np.isnan(temp)
        turbulence = turbulence[mask]
        nee = nee[mask]
        temp = temp[mask]
        
        # Split into temperature classes
        temp_bounds = np.percentile(temp, np.linspace(0, 100, 8))
        changepoints = []
        slopes = []
        f_maxes = []
        
        # Bootstrap analysis
        for _ in range(self.n_boot):
            # Resample data
            idx = np.random.randint(0, len(turbulence), len(turbulence))
            turb_boot = turbulence[idx]
            nee_boot = nee[idx]
            temp_boot = temp[idx]
            
            # Analyze each temperature class
            for i in range(len(temp_bounds)-1):
                mask = (temp_boot >= temp_bounds[i]) & (temp_boot < temp_bounds[i+1])
                if np.sum(mask) < bins_per_window * min_pts_per_bin:
                    continue
                    
                # Bin the data
                turb_class = turb_boot[mask]
                nee_class = nee_boot[mask]
                
                bins = np.quantile(turb_class, np.linspace(0, 1, bins_per_window+1))
                bin_means_x = []
                bin_means_y = []
                
                for j in range(bins_per_window):
                    bin_mask = (turb_class >= bins[j]) & (turb_class < bins[j+1])
                    if np.sum(bin_mask) >= min_pts_per_bin:
                        bin_means_x.append(np.mean(turb_class[bin_mask]))
                        bin_means_y.append(np.mean(nee_class[bin_mask]))
                
                if len(bin_means_x) < 5:
                    continue
                    
                # Detect changepoint
                cp, slope, f_max, _ = self.detect_changepoint(
                    np.array(bin_means_x),
                    np.array(bin_means_y)
                )
                
                if not np.isnan(cp) and f_max > self._get_f_critical(len(bin_means_x)):
                    changepoints.append(cp)
                    slopes.append(slope)
                    f_maxes.append(f_max)
        
        if len(changepoints) < 0.2 * self.n_boot:
            return FilterResults(
                threshold=np.nan,
                confidence_interval=(np.nan, np.nan),
                retained_data=np.full_like(turbulence, False),
                f_max=0,
                mode='insufficient_data',
                percent_selected=0
            )
            
        # Determine filtering mode
        slopes = np.array(slopes)
        if np.mean(slopes > 0) > 0.5:
            mode = 'deficit'
            valid_mask = slopes > 0
        else:
            mode = 'excess'
            valid_mask = slopes < 0
            
        # Calculate threshold and confidence interval
        valid_cps = np.array(changepoints)[valid_mask]
        threshold = np.median(valid_cps)
        conf_int = np.percentile(valid_cps, [2.5, 97.5])
        
        # Apply threshold
        retained = turbulence >= threshold
        
        return FilterResults(
            threshold=threshold,
            confidence_interval=tuple(conf_int),
            retained_data=retained,
            f_max=np.median(f_maxes),
            mode=mode,
            percent_selected=np.mean(valid_mask)
        )

class ClusterFilter:
    """Implementation of K-means clustering filtering method"""
    
    def __init__(self, n_clusters: int = 7):
        """
        Args:
            n_clusters: Number of clusters for K-means
        """
        self.n_clusters = n_clusters
        self.turbulence_filter = TurbulenceFilter()
        
    def filter_data(self,
                   features: np.ndarray,
                   sigma_w: np.ndarray,
                   nee: np.ndarray,
                   temp: np.ndarray,
                   radiation: Optional[np.ndarray] = None) -> FilterResults:
        """
        Filter flux data using clustering approach
        
        Args:
            features: Array of normalized input features [σw, temp, wind profiles]
            sigma_w: Array of σw measurements at reference height
            nee: Array of CO2 flux measurements  
            temp: Array of temperature measurements
            radiation: Optional array of radiation measurements
            
        Returns:
            FilterResults object containing filtering results
        """
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Evaluate each cluster
        retained_clusters = []
        f_maxes = []
        
        for cluster in range(self.n_clusters):
            mask = clusters == cluster
            if np.sum(mask) < 50:
                continue
                
            # Filter cluster data
            result = self.turbulence_filter.filter_data(
                sigma_w[mask],
                nee[mask], 
                temp[mask]
            )
            
            # Retain cluster if F-max is below threshold
            if result.f_max < 10.664:  # 1% significance level
                retained_clusters.append(cluster)
                f_maxes.append(result.f_max)
                
        # Create mask for retained data
        retained = np.zeros_like(nee, dtype=bool)
        for cluster in retained_clusters:
            retained |= clusters == cluster
            
        return FilterResults(
            threshold=np.nan,  # No single threshold for clustering
            confidence_interval=(np.nan, np.nan),
            retained_data=retained,
            f_max=np.mean(f_maxes) if f_maxes else 0,
            mode='cluster',
            percent_selected=len(retained_clusters) / self.n_clusters
        )

def calculate_omega(sigma_w: np.ndarray,
                   theta: np.ndarray, 
                   theta_e: np.ndarray,
                   h: float,
                   lai: float,
                   u_star: np.ndarray) -> np.ndarray:
    """
    Calculate physical decoupling metric Ω
    
    Args:
        sigma_w: Standard deviation of vertical velocity 
        theta: Mean potential temperature below h
        theta_e: Potential temperature of downward moving air
        h: Measurement height
        lai: Leaf area index
        u_star: Friction velocity
        
    Returns:
        omega: Decoupling metric Ω
    """
    g = 9.81  # Gravity acceleration
    
    # Calculate critical vertical velocity
    w_e_crit = np.sqrt(2 * g * h * (theta_e - theta) / theta - lai * u_star**2)
    
    return sigma_w / np.abs(w_e_crit)

def calculate_brunt_vaisala(theta: np.ndarray,
                          theta_e: np.ndarray,
                          z: float) -> np.ndarray:
    """
    Calculate squared Brunt-Väisälä frequency
    
    Args:
        theta: Mean potential temperature
        theta_e: Potential temperature of air parcel
        z: Height
        
    Returns:
        N2: Squared Brunt-Väisälä frequency 
    """
    g = 9.81
    return g * (theta - theta_e) / (theta_e * z)

# Utility functions for gapfilling
def train_gapfilling_model(features: np.ndarray,
                          nee: np.ndarray,
                          **kwargs) -> 'RandomForestRegressor':
    """
    Train Random Forest model for NEE gapfilling
    
    Args:
        features: Array of input features [day of year, soil temp, VPD, radiation]
        nee: Array of NEE measurements
        **kwargs: Additional arguments passed to RandomForestRegressor
        
    Returns:
        model: Trained RandomForestRegressor
    """
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(**kwargs)
    model.fit(features, nee)
    return model
