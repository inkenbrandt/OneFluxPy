import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple
from datetime import datetime, timedelta
import scipy.stats
from abc import ABC, abstractmethod

@dataclass
class CPDStatistics:
    """Statistics from change point detection analysis"""
    n: int  # Number of points
    cp: float  # Change point value
    fmax: float  # Maximum F statistic
    p: float  # P-value
    b0: float  # Intercept
    b1: float  # Slope below change point
    b2: float  # Slope difference above change point (3-param model only)
    c2: float  # Total slope above change point (3-param model only)
    cib0: float  # Confidence interval for b0
    cib1: float  # Confidence interval for b1  
    cic2: float  # Confidence interval for slope difference
    mean_time: float  # Mean time of window
    time_start: float  # Start time of window
    time_end: float  # End time of window
    ustar_t_corr: float  # Correlation between u* and temperature
    ustar_t_pval: float  # P-value of u*-temperature correlation
    mean_temp: float  # Mean temperature
    ci_temp: float  # Temperature confidence interval

class ChangePointResult(NamedTuple):
    """Results from change point detection"""
    cp: float  # Change point value 
    stats: CPDStatistics  # Associated statistics

def find_change_point(xx: np.ndarray, yy: np.ndarray, 
                     plot: bool = False, plot_title: str = '') -> Tuple[ChangePointResult, ChangePointResult]:
    """
    Find change points using 2-parameter and 3-parameter models
    
    Args:
        xx: Independent variable (e.g. u*)
        yy: Dependent variable (e.g. NEE)
        plot: Whether to create diagnostic plot
        plot_title: Title for plot if created
        
    Returns:
        Tuple of (2-param result, 3-param result)
    """
    # Remove missing values
    mask = ~np.isnan(xx) & ~np.isnan(yy)
    x = xx[mask]
    y = yy[mask]
    n = len(x)
    
    if n < 10:
        return ChangePointResult(np.nan, _empty_stats()), ChangePointResult(np.nan, _empty_stats())

    # Remove outliers
    coef = np.polyfit(x, y, 1)
    y_hat = np.polyval(coef, x)
    resid = y - y_hat
    outliers = abs(resid - np.mean(resid)) > 4 * np.std(resid)
    x = x[~outliers]
    y = y[~outliers]
    n = len(x)

    if n < 10:
        return ChangePointResult(np.nan, _empty_stats()), ChangePointResult(np.nan, _empty_stats())

    # Compute null models for significance testing
    y_hat2 = np.mean(y)
    sse_red2 = np.sum((y - y_hat2)**2)
    
    coef = np.polyfit(x, y, 1) 
    y_hat3 = np.polyval(coef, x)
    sse_red3 = np.sum((y - y_hat3)**2)
    
    # Find change points by maximizing F statistic
    n_end_pts = max(3, int(0.05 * n))
    f2_scores = np.zeros(n-1)
    f3_scores = np.zeros(n-1)
    
    for i in range(n-1):
        # 2-parameter model
        above = slice(i+1, n)
        x1 = x.copy()
        x1[above] = x[i]
        X = np.column_stack([np.ones(n), x1])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ coef
        sse_full2 = np.sum((y - y_hat)**2)
        f2_scores[i] = ((sse_red2 - sse_full2) / 
                       (sse_full2 / (n - 2)))

        # 3-parameter model  
        z_above = np.zeros(n)
        z_above[above] = 1
        x1 = x.copy()
        x2 = (x - x[i]) * z_above
        X = np.column_stack([np.ones(n), x1, x2])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ coef
        sse_full3 = np.sum((y - y_hat)**2)
        f3_scores[i] = ((sse_red3 - sse_full3) /
                       (sse_full3 / (n - 3)))

    # Find maxima and compute statistics
    i_max2 = np.argmax(f2_scores)
    cp2 = x[i_max2]
    f_max2 = f2_scores[i_max2]
    
    i_max3 = np.argmax(f3_scores) 
    cp3 = x[i_max3]
    f_max3 = f3_scores[i_max3]

    # Compute model fits at change points
    stats2 = _compute_2param_stats(x, y, i_max2, f_max2, n_end_pts)
    stats3 = _compute_3param_stats(x, y, i_max3, f_max3, n_end_pts)

    # Check significance
    if stats2.p > 0.05:
        cp2 = np.nan
    if stats3.p > 0.05:
        cp3 = np.nan
        
    return (ChangePointResult(cp2, stats2),
            ChangePointResult(cp3, stats3))

def _empty_stats() -> CPDStatistics:
    """Create empty statistics object"""
    return CPDStatistics(
        n=np.nan, cp=np.nan, fmax=np.nan, p=np.nan,
        b0=np.nan, b1=np.nan, b2=np.nan, c2=np.nan,
        cib0=np.nan, cib1=np.nan, cic2=np.nan,
        mean_time=np.nan, time_start=np.nan, time_end=np.nan,
        ustar_t_corr=np.nan, ustar_t_pval=np.nan,
        mean_temp=np.nan, ci_temp=np.nan
    )

def _compute_2param_stats(x: np.ndarray, y: np.ndarray, 
                         i_cp: int, f_max: float,
                         n_end_pts: int) -> CPDStatistics:
    """Compute statistics for 2-parameter model"""
    n = len(x)
    if not (n_end_pts < i_cp < n - n_end_pts):
        return _empty_stats()
        
    # Fit model
    above = slice(i_cp+1, n)
    x1 = x.copy()
    x1[above] = x[i_cp]
    X = np.column_stack([np.ones(n), x1])
    
    # Get coefficients and confidence intervals
    coef, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
    mse = resid[0] / (n - 2)
    conf_int = scipy.stats.t.ppf(0.975, n-2) * np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
    
    # Approximate p-value
    p = 1 - scipy.stats.f.cdf(f_max, 1, n-2)
    
    return CPDStatistics(
        n=n, cp=x[i_cp], fmax=f_max, p=p,
        b0=coef[0], b1=coef[1], b2=np.nan, c2=np.nan,
        cib0=conf_int[0], cib1=conf_int[1], cic2=np.nan,
        mean_time=np.nan, time_start=np.nan, time_end=np.nan,
        ustar_t_corr=np.nan, ustar_t_pval=np.nan,
        mean_temp=np.nan, ci_temp=np.nan
    )

def _compute_3param_stats(x: np.ndarray, y: np.ndarray,
                         i_cp: int, f_max: float, 
                         n_end_pts: int) -> CPDStatistics:
    """Compute statistics for 3-parameter model"""
    n = len(x)
    if not (n_end_pts < i_cp < n - n_end_pts):
        return _empty_stats()
        
    # Fit model
    above = slice(i_cp+1, n)
    z_above = np.zeros(n)
    z_above[above] = 1
    x1 = x.copy()
    x2 = (x - x[i_cp]) * z_above
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Get coefficients and confidence intervals
    coef, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
    mse = resid[0] / (n - 3)
    conf_int = scipy.stats.t.ppf(0.975, n-3) * np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
    
    # Approximate p-value
    p = 1 - scipy.stats.f.cdf(f_max, 2, n-3)
    
    return CPDStatistics(
        n=n, cp=x[i_cp], fmax=f_max, p=p,
        b0=coef[0], b1=coef[1], b2=coef[2],
        c2=coef[1] + coef[2],
        cib0=conf_int[0], cib1=conf_int[1], cic2=conf_int[2],
        mean_time=np.nan, time_start=np.nan, time_end=np.nan,
        ustar_t_corr=np.nan, ustar_t_pval=np.nan,
        mean_temp=np.nan, ci_temp=np.nan
    )

class UStarThresholdFinder(ABC):
    """Base class for u* threshold evaluation strategies"""
    
    def __init__(self, n_temp_strata: int = 4, n_bins: int = 50,
                 points_per_bin: int = 5):
        self.n_temp_strata = n_temp_strata
        self.n_bins = n_bins
        self.points_per_bin = points_per_bin

    @abstractmethod
    def evaluate(self, time: np.ndarray, nee: np.ndarray, 
                ustar: np.ndarray, temp: np.ndarray,
                night_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate u* thresholds"""
        pass
        
    def _bin_data(self, x: np.ndarray, y: np.ndarray, 
                 n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Bin x,y data using specified points per bin"""
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        n_bins = len(x) // n_points
        x_means = np.array([np.mean(x_sorted[i:i+n_points])
                           for i in range(0, len(x_sorted)-n_points, n_points)])
        y_means = np.array([np.mean(y_sorted[i:i+n_points])
                           for i in range(0, len(y_sorted)-n_points, n_points)])
                           
        return x_means, y_means

class SeasonalUStarThresholdFinder(UStarThresholdFinder):
    """Evaluate u* thresholds using 4 seasonal windows"""
    
    def evaluate(self, time: np.ndarray, nee: np.ndarray,
                ustar: np.ndarray, temp: np.ndarray, 
                night_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate seasonal u* thresholds"""
        
        # Get night time data
        night_idx = np.where(night_mask)[0]
        valid = ~np.isnan(nee[night_idx] + ustar[night_idx] + temp[night_idx])
        night_idx = night_idx[valid]
        
        if len(night_idx) < self.n_temp_strata * self.n_bins * self.points_per_bin:
            return np.full((4, 8), np.nan), np.full((4, 8), np.nan)
            
        # Split into seasons
        year = np.median(time).year
        season_bounds = [
            datetime(year-1, 12, 1),
            datetime(year, 3, 1),
            datetime(year, 6, 1), 
            datetime(year, 9, 1),
            datetime(year, 12, 1)
        ]
        
        cp2 = np.full((4, 8), np.nan)
        cp3 = np.full((4, 8), np.nan)
        
        for season in range(4):
            # Get seasonal data
            season_mask = ((time >= season_bounds[season]) &
                         (time < season_bounds[season+1]))
            season_idx = night_idx[season_mask[night_idx]]
            
            if len(season_idx) < 100:
                continue
                
            # Split by temperature
            temp_bounds = np.percentile(temp[season_idx],
                                      np.linspace(0, 100, self.n_temp_strata+1))
            
            for t in range(self.n_temp_strata):
                # Get temperature strata
                t_mask = ((temp[season_idx] >= temp_bounds[t]) &
                         (temp[season_idx] <= temp_bounds[t+1]))
                t_idx = season_idx[t_mask]
                
                if len(t_idx) < self.n_bins * self.points_per_bin:
                    continue
                    
                # Bin data
                ustar_bins, nee_bins = self._bin_data(
                    ustar[t_idx], nee[t_idx],
                    self.points_per_bin
                )
                
                # Find change points
                result2, result3 = find_change_point(ustar_bins, nee_bins)
                cp2[season,t] = result2.cp
                cp3[season,t] = result3.cp
                
        return cp2, cp3

class MovingWindowUStarThresholdFinder(UStarThresholdFinder):
    """