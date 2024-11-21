import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from typing import Optional, Tuple, NamedTuple

@dataclass
class ChangePointStats:
    """Statistics from change point detection analysis"""
    n: int = 0              # Number of points
    cp: float = np.nan      # Change point value
    fmax: float = np.nan    # Maximum F statistic
    p: float = np.nan       # P-value 
    b0: float = np.nan      # Intercept
    b1: float = np.nan      # Slope below change point
    b2: float = np.nan      # Slope difference above change point (3-param model only)
    c2: float = np.nan      # Total slope above change point (3-param model only)
    cib0: float = np.nan    # Confidence interval for b0
    cib1: float = np.nan    # Confidence interval for b1
    cic2: float = np.nan    # Confidence interval for slope difference

class ChangePointResult(NamedTuple):
    """Results from change point detection"""
    cp: float           # Change point value
    stats: ChangePointStats  # Associated statistics

def find_change_point(xx: np.ndarray, yy: np.ndarray, 
                     plot: bool = False, plot_title: str = '') -> Tuple[ChangePointResult, ChangePointResult]:
    """
    Find change points using 2-parameter and 3-parameter models following Lund and Reeves (2002)
    modified by Alan Barr for u* threshold evaluation.
    
    Args:
        xx: Independent variable (e.g. u*)
        yy: Dependent variable (e.g. NEE) 
        plot: Whether to create diagnostic plot
        plot_title: Title for plot if created
        
    Returns:
        Tuple of (2-param result, 3-param result)
        For each model:
        - If significant (p<0.05), returns change point value and statistics
        - If not significant, returns np.nan for change point and statistics
    """
    # Convert inputs to vectors and remove missing values
    x = np.asarray(xx).reshape(-1)
    y = np.asarray(yy).reshape(-1)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 10:
        return (ChangePointResult(np.nan, ChangePointStats()), 
                ChangePointResult(np.nan, ChangePointStats()))

    # Remove extreme linear regression outliers
    coef = np.polyfit(x, y, 1)
    y_hat = np.polyval(coef, x)
    resid = y - y_hat
    outliers = np.abs(resid - np.mean(resid)) > 4 * np.std(resid)
    x = x[~outliers]
    y = y[~outliers]
    n = len(x)

    if n < 10:
        return (ChangePointResult(np.nan, ChangePointStats()),
                ChangePointResult(np.nan, ChangePointStats()))

    # Compute statistics of reduced (null hypothesis) models
    y_hat2 = np.mean(y) * np.ones_like(y)
    sse_red2 = np.sum((y - y_hat2)**2)
    
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat3 = X @ coef
    sse_red3 = np.sum((y - y_hat3)**2)
    
    n_red2 = 1  # Parameters in reduced 2-param model
    n_full2 = 2  # Parameters in full 2-param model
    n_red3 = 2  # Parameters in reduced 3-param model 
    n_full3 = 3  # Parameters in full 3-param model

    # Compute F scores for each potential change point
    n_end_pts = max(3, int(0.05 * n))
    f2_scores = np.zeros(n-1)
    f3_scores = np.zeros(n-1)
    
    for i in range(n-1):
        # 2-parameter model (zero slope above change point)
        above = slice(i+1, n)
        x1 = x.copy()
        x1[above] = x[i]
        X = np.column_stack([np.ones_like(x), x1])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ coef
        sse_full2 = np.sum((y - y_hat)**2)
        f2_scores[i] = ((sse_red2 - sse_full2) / 
                       (sse_full2 / (n - n_full2)))

        # 3-parameter model (different non-zero slopes)
        z_above = np.zeros(n)
        z_above[above] = 1
        x1 = x.copy()
        x2 = (x - x[i]) * z_above
        X = np.column_stack([np.ones_like(x), x1, x2])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ coef
        sse_full3 = np.sum((y - y_hat)**2)
        f3_scores[i] = ((sse_red3 - sse_full3) /
                       (sse_full3 / (n - n_full3)))

    # Find maxima and compute statistics
    i_max2 = np.argmax(f2_scores)
    cp2 = x[i_max2]
    f_max2 = f2_scores[i_max2]
    
    i_max3 = np.argmax(f3_scores)
    cp3 = x[i_max3]
    f_max3 = f3_scores[i_max3]

    # Compute final model fits and statistics
    stats2 = ChangePointStats(n=n)
    stats3 = ChangePointStats(n=n)

    if n_end_pts < i_max2 < (n - n_end_pts):
        # 2-parameter model statistics
        above = slice(i_max2+1, n)
        x1 = x.copy()
        x1[above] = x[i_max2]
        X = np.column_stack([np.ones_like(x), x1])
        
        coef, resid, rank, s = np.linalg.lstsq(X, y, rcond=None)
        mse = resid[0] / (n - n_full2)
        ci = (stats.t.ppf(0.975, n-n_full2) * 
              np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X))))

        p2 = 1 - stats.f.cdf(f_max2, 1, n-n_full2)
        
        stats2.cp = cp2
        stats2.fmax = f_max2 
        stats2.p = p2
        stats2.b0 = coef[0]
        stats2.b1 = coef[1]
        stats2.cib0 = ci[0]
        stats2.cib1 = ci[1]

    if n_end_pts < i_max3 < (n - n_end_pts):
        # 3-parameter model statistics
        above = slice(i_max3+1, n)
        z_above = np.zeros(n)
        z_above[above] = 1
        x1 = x.copy()
        x2 = (x - x[i_max3]) * z_above
        X = np.column_stack([np.ones_like(x), x1, x2])
        
        coef, resid, rank, s = np.linalg.lstsq(X, y, rcond=None)
        mse = resid[0] / (n - n_full3)
        ci = (stats.t.ppf(0.975, n-n_full3) * 
              np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X))))

        p3 = 1 - stats.f.cdf(f_max3, 2, n-n_full3)
        
        stats3.cp = cp3
        stats3.fmax = f_max3
        stats3.p = p3
        stats3.b0 = coef[0]
        stats3.b1 = coef[1] 
        stats3.b2 = coef[2]
        stats3.c2 = coef[1] + coef[2]
        stats3.cib0 = ci[0]
        stats3.cib1 = ci[1]
        stats3.cic2 = ci[2]

    # Create final results, setting change points to NaN if not significant
    p_threshold = 0.05
    cp2_final = cp2 if stats2.p <= p_threshold else np.nan
    cp3_final = cp3 if stats3.p <= p_threshold else np.nan

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'ko', markerfacecolor='k', label='Data')
        
        # Plot 2-param model fit
        x_sort = np.sort(x)
        if not np.isnan(cp2_final):
            y2 = stats2.b0 + stats2.b1 * np.minimum(x_sort, cp2)
            plt.plot(x_sort, y2, 'r-', linewidth=2, label='2-param model')
            plt.axvline(cp2, color='r', linewidth=2)
            
        # Plot 3-param model fit
        if not np.isnan(cp3_final):
            mask = x_sort <= cp3
            y3 = np.zeros_like(x_sort)
            y3[mask] = stats3.b0 + stats3.b1 * x_sort[mask]
            y3[~mask] = (stats3.b0 + stats3.b1 * x_sort[~mask] + 
                        stats3.b2 * (x_sort[~mask] - cp3))
            plt.plot(x_sort, y3, 'g-', linewidth=2, label='3-param model')
            plt.axvline(cp3, color='g', linewidth=2)
            
        plt.grid(True)
        plt.title(f"{plot_title} CP2={cp2_final:.3f}")
        plt.legend()
        plt.show()

    return (ChangePointResult(cp2_final, stats2),
            ChangePointResult(cp3_final, stats3))
