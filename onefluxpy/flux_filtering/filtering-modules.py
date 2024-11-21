# filtering.py
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats

class FluxFilter:
    """Base class for CO2 flux filtering methods"""
    
    def __init__(self, n_seasons=4, n_temp_classes=7):
        self.n_seasons = n_seasons
        self.n_temp_classes = n_temp_classes
        
    def _calculate_changepoint(self, turbulence, flux, n_bins=50):
        """Implements changepoint detection using two-phase regression
        
        Args:
            turbulence: Array of turbulence measurements (u* or σw)
            flux: Array of CO2 flux measurements
            n_bins: Number of bins for binning the data
            
        Returns:
            changepoint: The calculated threshold
            f_max: Statistical significance measure
        """
        if len(turbulence) < n_bins:
            return None, 0
            
        # Bin the data
        bins = np.quantile(turbulence, np.linspace(0, 1, n_bins+1))
        bin_means_x = []
        bin_means_y = []
        
        for i in range(n_bins):
            mask = (turbulence >= bins[i]) & (turbulence < bins[i+1])
            if np.sum(mask) > 0:
                bin_means_x.append(np.mean(turbulence[mask]))
                bin_means_y.append(np.mean(flux[mask]))
                
        bin_means_x = np.array(bin_means_x)
        bin_means_y = np.array(bin_means_y)
        
        if len(bin_means_x) < 5:
            return None, 0
            
        # Find best split point
        max_r2 = -np.inf
        best_split = None
        f_max = 0
        
        for i in range(2, len(bin_means_x)-2):
            x1 = bin_means_x[:i]
            y1 = bin_means_y[:i]
            x2 = bin_means_x[i:]
            y2 = bin_means_y[i:]
            
            # Fit two lines
            slope1, intercept1, r1, _, _ = stats.linregress(x1, y1)
            slope2, intercept2, r2, _, _ = stats.linregress(x2, y2)
            
            # Calculate R2
            r2_total = (len(x1) * r1**2 + len(x2) * r2**2) / len(bin_means_x)
            
            if r2_total > max_r2:
                max_r2 = r2_total
                best_split = bin_means_x[i]
                
                # Calculate F statistic
                residuals1 = y1 - (slope1 * x1 + intercept1)
                residuals2 = y2 - (slope2 * x2 + intercept2)
                rss_split = np.sum(residuals1**2) + np.sum(residuals2**2)
                
                # Fit single line to all data
                slope, intercept, _, _, _ = stats.linregress(bin_means_x, bin_means_y)
                residuals = bin_means_y - (slope * bin_means_x + intercept)
                rss_single = np.sum(residuals**2)
                
                # Calculate F statistic
                if rss_split > 0:
                    f_max = ((rss_single - rss_split) / 2) / (rss_split / (len(bin_means_x) - 4))
                
        return best_split, f_max

class TurbulenceFilter(FluxFilter):
    """Implementation of u* and σw filtering methods"""
    
    def calculate_threshold(self, turbulence, flux, temp):
        """Calculate filtering threshold using CPD across seasons and temperature classes
        
        Args:
            turbulence: Array of turbulence measurements (u* or σw)
            flux: Array of CO2 flux measurements 
            temp: Array of temperature measurements
            
        Returns:
            threshold: Filtering threshold value
        """
        season_bounds = np.linspace(0, len(turbulence), self.n_seasons+1).astype(int)
        changepoints = []
        
        for i in range(self.n_seasons):
            season_mask = slice(season_bounds[i], season_bounds[i+1])
            season_temps = temp[season_mask]
            
            # Create temperature classes
            temp_bounds = np.percentile(season_temps, 
                                      np.linspace(0, 100, self.n_temp_classes+1))
            
            for j in range(self.n_temp_classes):
                temp_mask = ((season_temps >= temp_bounds[j]) & 
                           (season_temps < temp_bounds[j+1]))
                
                # Apply CPD
                cp, f_max = self._calculate_changepoint(
                    turbulence[season_mask][temp_mask],
                    flux[season_mask][temp_mask]
                )
                
                if cp is not None and f_max > 10.664:  # Significance threshold
                    changepoints.append(cp)
                    
        if len(changepoints) > 0:
            # Take maximum of seasonal medians
            seasonal_medians = []
            for i in range(self.n_seasons):
                season_cps = changepoints[i::self.n_seasons]
                if len(season_cps) > 0:
                    seasonal_medians.append(np.median(season_cps))
            return max(seasonal_medians)
        return None

class ClusterFilter(FluxFilter):
    """Implementation of K-means clustering filtering method"""
    
    def __init__(self, n_clusters=7, n_seasons=3, n_temp_classes=3, n_rad_classes=6):
        super().__init__(n_seasons, n_temp_classes)
        self.n_clusters = n_clusters
        self.n_rad_classes = n_rad_classes
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit(self, features):
        """Fit K-means clustering model
        
        Args:
            features: Array of normalized input features [σw, temp, wind profiles]
        """
        self.kmeans.fit(features)
        
    def evaluate_clusters(self, clusters, sigma_w, flux, temp, radiation=None):
        """Evaluate clusters using σw-CPD
        
        Args:
            clusters: Array of cluster assignments
            sigma_w: Array of σw measurements
            flux: Array of CO2 flux measurements
            temp: Array of temperature measurements
            radiation: Optional array of radiation measurements
            
        Returns:
            retained_clusters: List of cluster indices to retain
        """
        retained_clusters = []
        
        for cluster in range(self.n_clusters):
            cluster_mask = clusters == cluster
            
            # Skip if cluster is too small
            if np.sum(cluster_mask) < 50:
                continue
                
            f_max_values = []
            
            # Group data
            season_bounds = np.linspace(0, len(flux), self.n_seasons+1).astype(int)
            
            for i in range(self.n_seasons):
                season_mask = slice(season_bounds[i], season_bounds[i+1])
                season_temps = temp[season_mask]
                
                temp_bounds = np.percentile(season_temps,
                                          np.linspace(0, 100, self.n_temp_classes+1))
                
                for j in range(self.n_temp_classes):
                    temp_mask = ((season_temps >= temp_bounds[j]) & 
                               (season_temps < temp_bounds[j+1]))
                    
                    if radiation is not None:
                        season_rad = radiation[season_mask]
                        rad_bounds = np.percentile(season_rad,
                                                 np.linspace(0, 100, self.n_rad_classes+1))
                        
                        for k in range(self.n_rad_classes):
                            rad_mask = ((season_rad >= rad_bounds[k]) &
                                      (season_rad < rad_bounds[k+1]))
                            
                            combined_mask = cluster_mask[season_mask] & temp_mask & rad_mask
                            
                            _, f_max = self._calculate_changepoint(
                                sigma_w[season_mask][combined_mask],
                                flux[season_mask][combined_mask]
                            )
                            if f_max > 0:
                                f_max_values.append(f_max)
                    else:
                        combined_mask = cluster_mask[season_mask] & temp_mask
                        
                        _, f_max = self._calculate_changepoint(
                            sigma_w[season_mask][combined_mask],
                            flux[season_mask][combined_mask]
                        )
                        if f_max > 0:
                            f_max_values.append(f_max)
            
            if len(f_max_values) > 0:
                f_max_75th = np.percentile(f_max_values, 75)
                if f_max_75th < 10.664:  # Significance threshold
                    retained_clusters.append(cluster)
                    
        return retained_clusters

# physical_metrics.py
def calculate_omega(sigma_w, theta, theta_e, h, lai, u_star):
    """Calculate physical decoupling metric Ω
    
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

def calculate_brunt_vaisala(theta, theta_e, z):
    """Calculate squared Brunt-Väisälä frequency
    
    Args:
        theta: Mean potential temperature
        theta_e: Potential temperature of air parcel
        z: Height
        
    Returns:
        N2: Squared Brunt-Väisälä frequency
    """
    g = 9.81
    return g * (theta - theta_e) / (theta_e * z)

# gapfilling.py
from sklearn.ensemble import RandomForestRegressor

def train_gapfilling_model(features, nee, **kwargs):
    """Train Random Forest model for NEE gapfilling
    
    Args:
        features: Array of input features [day of year, soil temp, VPD, radiation]
        nee: Array of NEE measurements
        **kwargs: Additional arguments passed to RandomForestRegressor
        
    Returns:
        model: Trained RandomForestRegressor
    """
    model = RandomForestRegressor(**kwargs)
    model.fit(features, nee)
    return model
