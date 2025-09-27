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
        
        print("Precipitation analysis for ", city_name)

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
            ds_climo = xr.open_dataset('ERA-5_total_precipitation_monthly-1981-2020.nc', # make it quicker if file is already downloaded
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
            Area averaged precipitation time series
        """
        
        # Print dimensions and coordinates
        print("Dataset dimensions:", list(ds.dims.keys()))
        print("Dataset coordinates:", list(ds.coords.keys()))
    
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
        
        # Update box coordinates to 5x5
        lat_min = self.city_lat - 2.5
        lat_max = self.city_lat + 2.5
        lon_min = city_lon_adj - 2.5
        lon_max = city_lon_adj + 2.5
        
        # Select the 5x5 degree box with proper coordinate handling
        if ds.latitude[0] > ds.latitude[-1]:  # Decreasing latitude
            city_data = ds.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
        else:  # Increasing latitude
            city_data = ds.sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max)
            )
        
        print("Selected data shape:", city_data['tp'].shape)
        print("Selected lat range:", city_data.latitude.min().values, "to", city_data.latitude.max().values)
        print("Selected lon range:", city_data.longitude.min().values, "to", city_data.longitude.max().values)
        
        # Calculate area weighted mean
        precip_mm = city_data['tp'] * 1000
        
        # Check for NaN values before spatial averaging
        print("NaN values in precipitation data:", precip_mm.isnull().sum().values)
        print("Data range:", precip_mm.min().values, "to", precip_mm.max().values, "mm")
        
        # Calculate spatial mean over the 5x5 box
        city_precip = precip_mm.mean(dim=['latitude', 'longitude'])
        
        # Filter for the analysis period
        city_precip = city_precip.sel(valid_time=slice(f"{self.start_year}-01", f"{self.end_year}-12"))
        
        print("After time filtering:", len(city_precip), "time steps")
        print("Precipitation stats: mean = ", city_precip.mean().values, "std = ", city_precip.std().values, "mm/month")
        
        # Convert monthly to daily
        days_per_month = city_precip.valid_time.dt.days_in_month
        daily_equiv = city_precip / days_per_month
        
        print("Daily equivalent stats: mean = ", daily_equiv.mean().values, "std = ", daily_equiv.std().values, "mm/day")
        
        self.city_precip = city_precip
        self.daily_equiv = daily_equiv
        
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
        valid_data = self.daily_equiv.dropna('valid_time')
        
        # Identify extreme days above 95th percentile
        p95 = valid_data.quantile(0.95)
        self.p95_value = float(p95.values)
        extreme_days = self.daily_equiv > p95
        self.extreme_indices = extreme_days
        
        n_extreme = extreme_days.sum().values
        print("95th percentile value:", self.p95_value, "mm/day")
        print("Number of extreme events:", n_extreme)

        return self.p95_value, extreme_days
    

    def plot_cumulative_distribution(self):
        """
        Plot cumulative distribution function of daily precipitation
        and highlight 95% percent precipitation (part 2).
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate CDF
        precip_values = self.daily_equiv.values[~np.isnan(self.daily_equiv.values)]
        precip_sorted = np.sort(precip_values)
        p = np.arange(1, len(precip_sorted) + 1) / len(precip_sorted)
        
        # Plot CDF for 95% precipitation values
        ax.plot(precip_sorted, p * 100, 'b-', linewidth=2, label='Daily Precipitation CDF')
        ax.axvline(self.p95_value, color='red', linestyle='--', linewidth=2, 
           label='95th percentile (' + str(round(self.p95_value, 2)) + ' mm/day)')
        
        ax.set_xlabel('Daily Precipitation (mm/day)')
        ax.set_ylabel('Cumulative Probability (%)')
        ax.set_title('Cumulative Distribution of Daily Precipitation\n' + self.city_name + 
                 ' (' + str(self.start_year) + '-' + str(self.end_year) + ')')

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.savefig(self.city_name.replace(" ", "_") + '_precipitation_CDF.png', 
                dpi=300, bbox_inches='tight')
        plt.close()  
    
    def create_composite_maps(self):
        """
        Create composite mean and anomaly maps for extreme precipitation days (part 3).
        """
        
        # Calculate climatology
        climo_data = self.ds_climo.sel(valid_time=slice(str(self.start_year) + '-01', str(self.end_year) + '-12'))
        climo_mean = climo_data['tp'].mean(dim='valid_time') * 1000
        
        # Calculate composite mean for extreme days
        extreme_times = self.city_precip.valid_time.where(self.extreme_indices, drop=True)
        extreme_data = self.ds_climo.sel(valid_time=extreme_times)
        composite_mean = extreme_data['tp'].mean(dim='valid_time') * 1000

        # Calculate anomaly
        anomaly = composite_mean - climo_mean
        
        # Define map extent (40x40 degrees around the city)
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

        composite_plot = composite_mean.where(composite_mean > 0)  # Mask zero/negative values

        vmin_comp = np.nanpercentile(composite_plot.values, 10)
        vmax_comp = np.nanpercentile(composite_plot.values, 95)

        im1 = composite_plot.plot(
            ax=ax1, transform=ccrs.PlateCarree(),
            cmap='Blues', extend='max',
            vmin=vmin_comp, vmax=vmax_comp,
            cbar_kwargs={'label': 'Precipitation (mm/month)', 'shrink': 0.8}
        )

        ax1.plot(self.city_lon, self.city_lat, 'ro', markersize=6, 
                transform=ccrs.PlateCarree(), label=self.city_name)
        
        ax1.set_title('Composite Mean Precipitation on Extreme Days\n' + 
                      self.city_name + ' 95th Percentile Events', fontsize=14)
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
        
        anom_max = 4

        im2 = anomaly.plot(
            ax=ax2, transform=ccrs.PlateCarree(),
            cmap='coolwarm', center=0,
            vmin=-anom_max, vmax=anom_max,
            cbar_kwargs={'label': 'Precipitation Anomaly (mm/month)', 'shrink': 0.8}
        )
        
        ax2.plot(self.city_lon, self.city_lat, 'ro', markersize=6, 
                transform=ccrs.PlateCarree(), label=self.city_name)
        
        ax2.set_title('Precipitation Anomaly on Extreme Days\n(Composite - ' + 
                      str(self.start_year) + '-' + str(self.end_year) + ' Climatology)', fontsize=14)
        ax2.legend()
        
        gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
        gl2.top_labels = False
        gl2.right_labels = False
        
        plt.savefig(self.city_name.replace(" ", "_") + '_composite_maps.png', 
                dpi=300, bbox_inches='tight')
        plt.close() 
    
    
    def save_results(self):
        """
        Save the extracted city precipitation data.
        """

        # Save city precipitation time series
        output_filename = self.city_name.replace(" ", "_") + '_precipitation_data.nc'
        self.city_precip.to_netcdf(output_filename)

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
        stats_df.to_csv(self.city_name.replace(" ", "_") + '_stats.csv', index=False)

    
    def run_analysis(self):
        """
        Run the complete analysis for assignment 2.
        """
        # Step 1: Load data
        ds = self.download_and_load_data()
        
        # Step 2: Extract city time series
        self.extract_city_timeseries(ds)
        
        # Step 3: Calculate percentiles
        self.calculate_percentiles_and_extremes()
        
        # Step 4: Plot CDF (question 2)
        self.plot_cumulative_distribution()
        
        # Step 5: Create composite maps (question 3)
        self.create_composite_maps()
        
        # Step 6: Save results
        self.save_results()
        

# Main function
# Example using Chicago, IL
if __name__ == "__main__":
    # Define city coordinates
    city_name = "Chicago"
    city_lat = 41.8781
    city_lon = -87.6298 
    
    analyzer = PrecipitationAnalysis(
        city_name=city_name,
        city_lat=city_lat,
        city_lon=city_lon,
        start_year=1990,
        end_year=2020
    )
    
    analyzer.run_analysis()

"""
Data References:
---------------
Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. 
Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
DOI: 10.1002/qj.3803
"""