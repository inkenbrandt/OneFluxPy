class MovingWindowUStarThresholdFinder(UStarThresholdFinder):
    """
    Evaluate u* thresholds using moving windows through the year
    
    This approach uses overlapping windows through the year to capture temporal
    variation in the u* threshold, with temperature stratification within each window.
    """
    
    def __init__(self, n_temp_strata: int = 4, n_bins: int = 50,
                points_per_bin: int = 5, n_windows: int = 4):
        """
        Args:
            n_temp_strata: Number of temperature classes within each window
            n_bins: Number of bins for u* in each temperature class
            points_per_bin: Number of points per bin
            n_windows: Number of overlapping windows through the year
        """
        super().__init__(n_temp_strata, n_bins, points_per_bin)
        self.n_windows = n_windows
        
    def evaluate(self, time: np.ndarray, nee: np.ndarray,
                ustar: np.ndarray, temp: np.ndarray,
                night_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate u* thresholds using moving windows
        
        Args:
            time: Timestamp array
            nee: Net Ecosystem Exchange array 
            ustar: Friction velocity array
            temp: Temperature array
            night_mask: Boolean array indicating nighttime periods
            
        Returns:
            Tuple of (cp2, cp3) arrays containing thresholds for each window/temperature class
            Each is shape (n_windows, n_temp_strata)
        """
        # Get nighttime data
        night_idx = np.where(night_mask)[0]
        valid = ~np.isnan(nee[night_idx] + ustar[night_idx] + temp[night_idx])
        night_idx = night_idx[valid]
        
        # Check minimum data requirements
        points_per_window = self.n_temp_strata * self.n_bins * self.points_per_bin
        if len(night_idx) < points_per_window:
            return (np.full((self.n_windows, self.n_temp_strata), np.nan),
                   np.full((self.n_windows, self.n_temp_strata), np.nan))
                   
        # Handle December wraparound by appending data
        year = np.median(time).year
        dec_mask = (time < datetime(year, 1, 1))
        if np.any(dec_mask):
            dec_idx = np.where(dec_mask)[0]
            time = np.concatenate([time[dec_idx], time])
            nee = np.concatenate([nee[dec_idx], nee])
            ustar = np.concatenate([ustar[dec_idx], ustar])
            temp = np.concatenate([temp[dec_idx], temp])
            night_mask = np.concatenate([night_mask[dec_idx], night_mask])
            
            # Update nighttime indices
            night_idx = np.where(night_mask)[0]
            valid = ~np.isnan(nee[night_idx] + ustar[night_idx] + temp[night_idx])
            night_idx = night_idx[valid]
        
        # Initialize outputs
        cp2 = np.full((self.n_windows, self.n_temp_strata), np.nan)
        cp3 = np.full((self.n_windows, self.n_temp_strata), np.nan)
        
        # Calculate window parameters
        window_size = points_per_window
        increment = window_size // 2  # 50% overlap between windows
        
        # Process each window
        for w in range(self.n_windows):
            # Get window indices
            start_idx = w * increment
            end_idx = start_idx + window_size
            if end_idx > len(night_idx):
                end_idx = len(night_idx)
                start_idx = end_idx - window_size
            
            window_idx = night_idx[start_idx:end_idx]
            
            # Split window data into temperature classes
            temp_bounds = np.percentile(temp[window_idx],
                                      np.linspace(0, 100, self.n_temp_strata + 1))
            
            # Process each temperature class
            for t in range(self.n_temp_strata):
                # Get data for temperature class
                t_mask = ((temp[window_idx] >= temp_bounds[t]) & 
                         (temp[window_idx] <= temp_bounds[t+1]))
                t_idx = window_idx[t_mask]
                
                if len(t_idx) < self.n_bins * self.points_per_bin:
                    continue
                    
                # Bin the data
                ustar_bins, nee_bins = self._bin_data(
                    ustar[t_idx],
                    nee[t_idx],
                    self.points_per_bin
                )
                
                # Find change points
                result2, result3 = find_change_point(ustar_bins, nee_bins)
                
                cp2[w,t] = result2.cp
                cp3[w,t] = result3.cp
                
        return cp2, cp3

