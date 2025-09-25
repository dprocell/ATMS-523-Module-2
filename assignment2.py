"""
ATMS 523 Weather and Climate Data Analytics
Project 2: dask & xarray for computing weather and climate diagnostics

Author: Dara Procell
Date: September 26, 2025
Description: Analysis of extreme daily precipitation and associated global circulation patterns
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import dask
import pandas as pd
from scipy import stats


# Enable dask 
dask.config.set({'array.slicing.split_large_chunks': False})


class PrecipitationAnalysis:
    """
    Analyze extreme precipitation events and associated circulation patterns.
    """
    
    def __init__(self, city_name, city_lat, city_lon, start_year=1990, end_year=2020):
        """
        Initialize the precipitation analysis.
        
        Parameters:
        -----------
        city_name : str
            Name of the city for analysis
        city_lat : float
            Latitude of the city
        city_lon : float
            Longitude of the city
        start_year : int
            Start year for analysis (default: 1990)
        end_year : int
            End year for analysis (default: 2020)
        """
        self.city_name = city_name
        self.city_lat = city_lat
        self.city_lon = city_lon
        self.start_year = start_year
        self.end_year = end_year
        
        # Define 5x5 degree box around the city
        self.lat_min = city_lat - 2.5
        self.lat_max = city_lat + 2.5
        self.lon_min = city_lon - 2.5
        self.lon_max = city_lon + 2.5
        
        print(f"Analyzing precipitation for {city_name}")
        print(f"City coordinates: {city_lat:.2f}°N, {city_lon:.2f}°E")
        print(f"Analysis box: {self.lat_min:.1f}°-{self.lat_max:.1f}°N, "
              f"{self.lon_min:.1f}°-{self.lon_max:.1f}°E")
        

    def download_and_load_data(self):
        """
        Download and load ERA-5 precipitation data.
        
        Returns:
        --------
        xarray.Dataset
            Loaded precipitation dataset
        """

        # Load climatology data 
        try:
            ds_climo = xr.open_dataset('ERA-5_total_precipitation_monthly-1981-2020.nc', 
                                     engine='h5netcdf')
        except FileNotFoundError:
            ds_climo = xr.open_dataset(
                'https://www.atmos.illinois.edu/~snesbitt/data/ERA-5_total_precipitation_monthly-1981-2020.nc', 
                engine='h5netcdf'
            )
            ds_climo.to_netcdf('ERA-5_total_precipitation_monthly-1981-2020.nc')
        
        self.ds_climo = ds_climo
        return ds_climo
    
    
    def extract_city_timeseries(self, ds):
        """
        Extract precipitation time series for the 5x5 degree box around the city.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Input precipitation dataset
        
        Returns:
        --------
        xarray.DataArray
            Area-averaged precipitation time series
        """
        
        # Check the actual dimensions and coordinates
        print("Dataset dimensions:", list(ds.dims.keys()))
        print("Dataset coordinates:", list(ds.coords.keys()))
        print("Dataset variables:", list(ds.data_vars.keys()))
        
        # Check coordinate ranges
        print(f"Latitude range: {ds.latitude.min().values:.2f} to {ds.latitude.max().values:.2f}")
        print(f"Longitude range: {ds.longitude.min().values:.2f} to {ds.longitude.max().values:.2f}")
        
        # Determine the time dimension name
        time_dim = None
        for dim in ['valid_time']:
            if dim in ds.dims:
                time_dim = dim
                break
    

        
        # Check if longitude needs conversion (0-360 to -180-180)
        lon_min, lon_max = ds.longitude.min().values, ds.longitude.max().values
        if lon_min >= 0 and lon_max > 180:
            ds_converted = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds_converted = ds_converted.sortby('longitude')
            ds = ds_converted
        
        # Adjust longitude coordinates for the city if needed
        city_lon_adj = self.city_lon
        if city_lon_adj < 0 and ds.longitude.min().values >= 0:
            city_lon_adj = self.city_lon + 360
        
        # Update box coordinates
        lat_min = self.city_lat - 2.5
        lat_max = self.city_lat + 2.5
        lon_min = city_lon_adj - 2.5
        lon_max = city_lon_adj + 2.5
        
        # Select the 5x5 degree box with proper coordinate handling
        if ds.latitude[0] > ds.latitude[-1]:  # Decreasing latitude (ERA5 style)
            city_data = ds.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
        else:  # Increasing latitude
            city_data = ds.sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max)
            )
        
        print(f"Selected data shape: {city_data['tp'].shape}")
        print(f"Selected lat range: {city_data.latitude.min().values:.2f} to {city_data.latitude.max().values:.2f}")
        print(f"Selected lon range: {city_data.longitude.min().values:.2f} to {city_data.longitude.max().values:.2f}")
        
        # Calculate area-weighted mean
        # Convert precipitation from m to mm (multiply by 1000)
        precip_mm = city_data['tp'] * 1000
        
        # Check for NaN values before spatial averaging
        print(f"NaN values in precipitation data: {precip_mm.isnull().sum().values}")
        print(f"Data range: {precip_mm.min().values:.4f} to {precip_mm.max().values:.4f} mm")
        
        # Calculate spatial mean over the 5x5 box
        city_precip = precip_mm.mean(dim=['latitude', 'longitude'])
        
        # Filter for the analysis period
        city_precip = city_precip.sel(**{time_dim: slice(f"{self.start_year}-01", f"{self.end_year}-12")})
        
        print(f"After time filtering: {len(city_precip)} time steps")
        print(f"Precipitation stats: mean={city_precip.mean().values:.3f}, std={city_precip.std().values:.3f} mm/month")
        
        # Convert monthly to daily equivalent (approximate)
        days_per_month = getattr(city_precip, time_dim).dt.days_in_month
        daily_equiv = city_precip / days_per_month
        
        print(f"Daily equivalent stats: mean={daily_equiv.mean().values:.3f}, std={daily_equiv.std().values:.3f} mm/day")
        
        self.city_precip = city_precip
        self.daily_equiv = daily_equiv
        self.time_dim = time_dim  # Store for later use
        
        return city_precip
    
    def calculate_percentiles_and_extremes(self):
        """
        Calculate the 95th percentile and identify extreme precipitation days.
        
        Returns:
        --------
        tuple
            95th percentile value and extreme day indices
        """
        
        # Check for valid data
        valid_data = self.daily_equiv.dropna(self.time_dim)
        
        if len(valid_data) == 0:
            raise ValueError("No valid data points found!!")
        
        # Calculate 95th percentile 
        p95 = valid_data.quantile(0.95)
        self.p95_value = float(p95.values)
        
        # Identify extreme days (above 95th percentile)
        extreme_days = self.daily_equiv > p95
        self.extreme_indices = extreme_days
        
        n_extreme = extreme_days.sum().values
        print(f"95th percentile value: {self.p95_value:.2f} mm/day")
        print(f"Number of extreme events: {n_extreme}")

        return self.p95_value, extreme_days
    

    def plot_cumulative_distribution(self):
        """
        Plot cumulative distribution function of daily precipitation.
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate CDF
        precip_values = self.daily_equiv.values[~np.isnan(self.daily_equiv.values)]
        precip_sorted = np.sort(precip_values)
        p = np.arange(1, len(precip_sorted) + 1) / len(precip_sorted)
        
        # Plot CDF
        ax.plot(precip_sorted, p * 100, 'b-', linewidth=2, label='Daily Precipitation CDF')
        
        # Mark 95th percentile
        ax.axvline(self.p95_value, color='red', linestyle='--', linewidth=2, 
                   label=f'95th percentile ({self.p95_value:.2f} mm/day)')
        
        ax.set_xlabel('Daily Precipitation (mm/day)')
        ax.set_ylabel('Cumulative Probability (%)')
        ax.set_title(f'Cumulative Distribution of Daily Precipitation\n{self.city_name} '
                     f'({self.start_year}-{self.end_year})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.city_name.replace(" ", "_")}_precipitation_CDF.png', 
                   dpi=300, bbox_inches='tight')
        print(f"CDF plot saved as: {self.city_name.replace(' ', '_')}_precipitation_CDF.png")
        plt.close()  
    
    def create_composite_maps(self):
        """
        Create composite mean and anomaly maps for extreme precipitation days.
        """

        time_dim = getattr(self, 'time_dim', 'valid_time')
        
        # Calculate climatology
        climo_data = self.ds_climo.sel(**{time_dim: slice(f'{self.start_year}-01', f'{self.end_year}-12')})
        climo_mean = climo_data['tp'].mean(dim=time_dim) * 1000  # Convert to mm
        
        # Calculate composite mean for extreme days
        extreme_times = getattr(self.city_precip, time_dim).where(self.extreme_indices, drop=True)
        
        # Select extreme days from the full dataset
        try:
            extreme_data = self.ds_climo.sel(**{time_dim: extreme_times})
            composite_mean = extreme_data['tp'].mean(dim=time_dim) * 1000  # Convert to mm
        except Exception as e:
            print("Trying alternative approach with nearest neighbor selection")
            extreme_data = self.ds_climo.sel(**{time_dim: extreme_times, 'method': 'nearest'})
            composite_mean = extreme_data['tp'].mean(dim=time_dim) * 1000  # Convert to mm
        
        # Calculate anomaly
        anomaly = composite_mean - climo_mean
        
        # Define map extent (40x40 degrees around city)
        map_extent = [
            self.city_lon - 20, self.city_lon + 20,
            self.city_lat - 20, self.city_lat + 20
        ]
        
        fig = plt.figure(figsize=(16, 12))
        
        # Composite mean map
        ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax1.set_extent(map_extent, ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        ax1.add_feature(cfeature.OCEAN, alpha=0.5)
        ax1.add_feature(cfeature.LAND, alpha=0.5)
        
        # Plot composite mean
        im1 = composite_mean.plot(
            ax=ax1, transform=ccrs.PlateCarree(),
            cmap='Blues', extend='max',
            cbar_kwargs={'label': 'Precipitation (mm/month)', 'shrink': 0.8}
        )
        
        # Mark city location
        ax1.plot(self.city_lon, self.city_lat, 'ro', markersize=6, 
                transform=ccrs.PlateCarree(), label=self.city_name)
        
        ax1.set_title(f'Composite Mean Precipitation on Extreme Days\n'
                     f'{self.city_name} 95th Percentile Events', fontsize=14)
        ax1.legend()
        
        # Add gridlines
        gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
        gl1.top_labels = False
        gl1.right_labels = False
        
        # Anomaly map
        ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        ax2.set_extent(map_extent, ccrs.PlateCarree())
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        ax2.add_feature(cfeature.OCEAN, alpha=0.5)
        ax2.add_feature(cfeature.LAND, alpha=0.5)
        
        # Plot anomaly
        im2 = anomaly.plot(
            ax=ax2, transform=ccrs.PlateCarree(),
            cmap='RdBu_r', center=0, extend='both',
            cbar_kwargs={'label': 'Precipitation Anomaly (mm/month)', 'shrink': 0.8}
        )
        
        # Mark city location
        ax2.plot(self.city_lon, self.city_lat, 'ro', markersize=6, 
                transform=ccrs.PlateCarree(), label=self.city_name)
        
        ax2.set_title(f'Precipitation Anomaly on Extreme Days\n'
                     f'(Composite - 1990-2020 Climatology)', fontsize=14)
        ax2.legend()
        
        # Add gridlines
        gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
        gl2.top_labels = False
        gl2.right_labels = False
        
        plt.tight_layout()
        plt.savefig(f'{self.city_name.replace(" ", "_")}_composite_maps.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Composite maps saved as: {self.city_name.replace(' ', '_')}_composite_maps.png")
        plt.close() 
    
    
    def save_results(self):
        """
        Save the extracted city precipitation data.
        """

        
        # Save city precipitation time series
        output_filename = f'{self.city_name.replace(" ", "_")}_precipitation_data.nc'
        self.city_precip.to_netcdf(output_filename)
        
        # Save summary statistics
        stats_dict = {
            'city': self.city_name,
            'latitude': self.city_lat,
            'longitude': self.city_lon,
            'analysis_period': f"{self.start_year}-{self.end_year}",
            'p95_value_mm_day': self.p95_value,
            'n_extreme_events': int(self.extreme_indices.sum().values),
            'mean_precipitation_mm_day': float(self.daily_equiv.mean().values),
            'std_precipitation_mm_day': float(self.daily_equiv.std().values)
        }
        
        stats_df = pd.DataFrame([stats_dict])
        stats_df.to_csv(f'{self.city_name.replace(" ", "_")}_stats.csv', index=False)
    
    def run_analysis(self):
        """
        Run the complete analysis workflow.
        """
        # Step 1: Load data
        ds = self.download_and_load_data()
        
        # Step 2: Extract city time series
        self.extract_city_timeseries(ds)
        
        # Step 3: Calculate percentiles
        self.calculate_percentiles_and_extremes()
        
        # Step 4: Plot CDF
        self.plot_cumulative_distribution()
        
        # Step 5: Create composite maps
        self.create_composite_maps()
        
        # Step 6: Save results
        self.save_results()
        

# Example usage for Chicago, IL
if __name__ == "__main__":
    # Define city coordinates
    city_name = "Chicago"
    city_lat = 41.8781
    city_lon = -87.6298  # Western longitude is negative
    
    # Initialize analysis
    analyzer = PrecipitationAnalysis(
        city_name=city_name,
        city_lat=city_lat,
        city_lon=city_lon,
        start_year=1990,
        end_year=2020
    )
    
    # Run the complete analysis
    analyzer.run_analysis()

"""
Data References:
---------------
Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. 
Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
DOI: 10.1002/qj.3803

ERA5 hourly data on single levels from 1940 to present. (2023). 
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 
DOI: 10.24381/cds.adbb2d47
"""