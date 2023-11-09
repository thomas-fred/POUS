#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from tqdm.notebook import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances


# In[ ]:


"""
This script is currently broken!

The clustering isn't working as it used to, need to investigate.

Commit once it's working!
"""


# In[ ]:


plt.style.use('dark_background')  # for cool points


# In[ ]:


# input parameters

outage_threshold = 0.05  # OutageFraction above this is considered an outage
resample_freq = "1D"  # resample raw hourly data to this resolution, then check for outage state
start_buffer = "2D"  # when plotting outage timeseries, start this delta ahead of first outage period
end_buffer = "1W"  # when plotting outage timeseries, end this delta after last outage period

# outage clustering (DBSCAN) parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# temporal clustering
temp_epsilon_days = 6
temp_min_samples = 1
# spatial clustering
geo_epsilon_deg = 1.2
geo_min_samples = 1


# In[ ]:


# define file paths (except for plots, defined later)

years = list(range(2017, 2023))  # years we have POUS data for

root_dir = "data"
states = pd.read_csv(
    os.path.join(
        root_dir,
        "raw",
        "states",
        "state_codes.csv"
    )
).set_index("state_fips_code")
county_boundaries: gpd.GeoDataFrame = gpd.read_file(
    os.path.join(
        root_dir,
        "raw",
        "counties",
        "cb_2018_us_county_500k.shp"
    )
)
all_counties_hourly_path = os.path.join(
    root_dir,
    "processed",
    "outage",
    f"all_counties_hourly.parquet"
)
outage_integrals_path = os.path.join(
    root_dir,
    "processed",
    "outage",
    f"{resample_freq}_county_outage_integrals.csv"
)
outage_attr_path = os.path.join(
    root_dir,
    "processed",
    "outage",
    f"{resample_freq}_{outage_threshold}_outage_attributes_for_clustering.csv"
)
plot_dir = os.path.join(
    "plots",
    "outage_timeseries_county_grouped_dbscan",
    f"resample_{resample_freq}",
    f"threshold_{outage_threshold}",
    f"{temp_epsilon_days=:.1f}",
    f"{temp_min_samples=:d}",
    f"{geo_epsilon_deg=:.1f}",
    f"{geo_min_samples=:d}",
)
duration_dir = os.path.join(
    root_dir,
    "processed",
    "outage",
    "durations",
    f"resample_{resample_freq}",
    f"threshold_{outage_threshold}",
    f"{temp_epsilon_days=:.1f}",
    f"{temp_min_samples=:d}",
    f"{geo_epsilon_deg=:.1f}",
    f"{geo_min_samples=:d}",
)


# In[ ]:


# if os.path.exists(all_counties_hourly_path):
#     print("Loading hourly data")
#     all_counties_hourly = pd.read_parquet(all_counties_hourly_path)
# else:
print("Building hourly data")
# read source outage data
data_by_year = {}
for year in years:
    processed_path = os.path.join(root_dir, f"processed/outage/{year}.parquet")
    data = pd.read_parquet(processed_path)
    data.OutageFraction = np.clip(data.OutageFraction, 0, 1)
    data_by_year[year] = data

# another view of source data, concat into single dataframe
all_counties_hourly = pd.concat(data_by_year).drop(columns=["CustomersTracked", "CustomersOut"])
all_counties_hourly = all_counties_hourly.droplevel(0)
all_counties_hourly.to_parquet(all_counties_hourly_path)
    
# find set of all counties in data
counties = sorted(set(all_counties_hourly.index.get_level_values("CountyFIPS")))


# In[ ]:


# construct complete timeseries of outage data
# resample to resample_freq and take mean of OutageFraction
# save to disk as cache

# if os.path.exists(outage_integrals_path):
#     print("Loading resampled data")
#     df = pd.read_csv(outage_integrals_path, dtype={"CountyFIPS": str}) 
    
# else:
print("Building resampled data")
resampled_data_by_year = []
for county_code in tqdm(counties):

    try:
        data = all_counties_hourly.loc(axis=0)[:, county_code].reset_index(level="CountyFIPS")
        complete_index = pd.date_range(f"{min(years)}-04-01", f"{max(years)}-10-31", freq="1H")
        data = data.reindex(index=complete_index, fill_value=0)
        data.index.name = "RecordDateTime"
    except KeyError:
        continue

    data = data.drop(columns=["CountyFIPS"]).resample(resample_freq).mean()
    data["CountyFIPS"] = county_code
    resampled_data_by_year.append(data)

df = pd.concat(resampled_data_by_year)
df.to_csv(outage_integrals_path)


# In[ ]:


# build a table of outages
# look through resampled data and identify periods of extended poor service for each county

# if os.path.exists(outage_attr_path):
#     print("Loading outage attributes")
#     outage_attributes = pd.read_csv(outage_attr_path, dtype={"CountyFIPS": str})

# else:
print("Building outage attributes")
data_start: pd.Timestamp = pd.to_datetime(f"{min(years)}-04-01")

# take the resampled data and filter to periods with OutageFraction above a threshold
outages = df.set_index(pd.to_datetime(df.RecordDateTime)).drop("RecordDateTime", axis=1)
outages = outages.drop(["CustomersTracked", "CustomersOut"], axis=1)
outages = df[df.OutageFraction > outage_threshold]

# duration of single resampling period in nanoseconds
length_of_resample_period_ns = pd.Timedelta(resample_freq).total_seconds() * 1E9

outage_attributes = []
for county_code in tqdm(set(outages.CountyFIPS)):

    county_outages_resampled: pd.DataFrame = outages[outages.CountyFIPS == county_code]
    county_data_hourly: pd.DataFrame = all_counties_hourly.loc[(slice(None), county_code), :]

    # picking out runs of resampled outage periods
    start = 0
    outage_period_resampled_indicies: list[tuple[int, int]] = []
    for i, time_gap_ns in enumerate(np.diff(county_outages_resampled.index.values)):
        if np.abs((float(time_gap_ns) / length_of_resample_period_ns) - 1) < 0.01:
            outage_period_resampled_indicies.append((start, i))
            start = i + 1

    county_centroid = county_boundaries.set_index("GEOID").loc[county_code].geometry.centroid

    for period_indicies in outage_period_resampled_indicies:

        start_i, end_i = period_indicies
        n_periods: int = end_i - start_i

        # check outage is at least 1 resample period long
        if n_periods >= 1:

            # retrieve indicies of resampled run of outage periods
            group_datetimeindex: pd.DatetimeIndex = county_outages_resampled.iloc[start_i: end_i + 1].index

            outage_attributes.append(
                (
                    county_code,
                    group_datetimeindex[0].date(),
                    (group_datetimeindex[0] - data_start).value / (1E9 * 60 * 60 * 24),
                    group_datetimeindex[0].year,
                    group_datetimeindex[0].day_of_year,
                    n_periods,
                    county_centroid.x,
                    county_centroid.y
                )
            )

outage_attributes = pd.DataFrame(
    outage_attributes,
    columns=[
        "CountyFIPS",
        "start",
        "days_since_data_start",
        "year",
        "day_of_year",
        "n_periods",
        "county_long",
        "county_lat"
    ]
)
outage_attributes.to_csv(outage_attr_path, index=False)

outage_attributes = outage_attributes.sort_values("days_since_data_start").reset_index()


# In[ ]:


outage_attributes


# In[ ]:


outage_attributes


# In[ ]:


def us_admin_name(
    county_code: str,
    county_boundaries: gpd.GeoDataFrame,
    states: pd.DataFrame
) -> tuple[str, str]:
    """
    Lookup county name, containing state name and code, given a 5-digit string county code.
    """
    
    try:
        county_admin_data: pd.Series = \
            county_boundaries.sort_values("GEOID").set_index("GEOID").loc[county_code, :]
        state_code: str = county_admin_data.STATEFP
        state_name: str = states.loc[int(state_code), "state_name"]
        county_name: str = county_admin_data.NAME
    except Exception as e:
        state_name = "-"
        county_name = "-"
        
    return county_name, state_name, state_code


# In[ ]:


outage_attributes


# In[ ]:


# calculate the pairwise distance in time between outage events

arr = outage_attributes.days_since_data_start.values
distance = np.abs(arr - arr[:, None])
f, ax = plt.subplots()
f.subplots_adjust(right=0.9)
cbar_ax = f.add_axes([0.85, 0.11, 0.04, 0.77])
img = ax.imshow(distance)
f.colorbar(img, cax=cbar_ax, label="Time [days]")
ax.set_title("Pairwise distance")


# In[ ]:


# cluster events temporally using the pairwise distance matrix

def temporal_cluster(distance: np.ndarray, epsilon_days: int, min_samples: int):
    dbscan = DBSCAN(
        eps=epsilon_days,
        min_samples=min_samples,
        metric="precomputed",
    )
    dbscan.fit(distance)
    return pd.Series(dbscan.labels_)

outage_attributes["time_cluster_id"] = temporal_cluster(distance, temp_epsilon_days, temp_min_samples)

f, ax = plt.subplots(figsize=(12,3))
for cluster_id in set(outage_attributes.time_cluster_id):
    data = outage_attributes[outage_attributes.time_cluster_id == cluster_id]
    ax.bar(
        data.days_since_data_start,
        np.ones(len(data)) * cluster_id,
        width=np.ones(len(data)) * 1,
        label=cluster_id
    )
ax.grid(alpha=0.2)
ax.set_xlabel("Time [days since start]")


# In[ ]:


# spatially cluster within each temporal cluster

def geo_cluster(lat: np.ndarray, long: np.ndarray, epsilon_deg, min_samples):
    dbscan = DBSCAN(
        eps=np.deg2rad(epsilon_deg),
        min_samples=min_samples,
        metric='haversine'
    )
    lat_lng_pts = [x for x in zip(lat, long)]
    dbscan.fit(np.radians(lat_lng_pts))
    return pd.Series(dbscan.labels_)

for time_cluster_id in set(outage_attributes.time_cluster_id):
    
    time_cluster_mask = outage_attributes.time_cluster_id == time_cluster_id

    outage_attributes.loc[time_cluster_mask, "geo_cluster_id"] = geo_cluster(
        outage_attributes.county_lat.values,
        outage_attributes.county_long.values,
        geo_epsilon_deg,  # epsilon degrees 
        geo_min_samples
    )

    outage_attributes.loc[time_cluster_mask, "geometry"] = gpd.points_from_xy(
        outage_attributes.loc[time_cluster_mask, "county_long"],
        outage_attributes.loc[time_cluster_mask, "county_lat"]
    )
    outage_attributes = gpd.GeoDataFrame(outage_attributes)
    
outage_attributes.geo_cluster_id = outage_attributes.geo_cluster_id.astype(int)

# generate a unique spatio-temporal cluster id
outage_attributes["cluster_id"] = outage_attributes.apply(
    lambda row: tuple([row.time_cluster_id, int(row.geo_cluster_id)]),
    axis=1
)


# In[ ]:


os.makedirs(duration_dir, exist_ok=True)
outage_attributes.loc[:, ["CountyFIPS", "start", "n_periods", "cluster_id"]].to_csv(
    os.path.join(
        duration_dir,
        f"POUS_{resample_freq=}_{outage_threshold=}_durations.csv",
    )
)


# In[ ]:


f, ax = plt.subplots()
outage_attributes.n_periods.plot(
    kind="hist",
    bins=30,
    ax=ax
)
ax.set_yscale("log")
ax.grid(alpha=0.2, which="both")
ax.set_xlabel(f"Outage duration [{resample_freq}]")
ax.set_title(f"POUS distribution of outage durations, {min(years)}-{max(years)}")


# In[ ]:


f, ax = plt.subplots()
ax.scatter(
    outage_attributes.day_of_year,
    outage_attributes.n_periods
)
ax.set_xlabel("Day of year")
ax.set_ylabel(f"Outage duration [{resample_freq}]")
ax.set_title(f"POUS outage duration seasonality, {min(years)}-{max(years)}")


# In[ ]:


# plot timeseries for each event, with a little inset map of relevant counties

os.makedirs(plot_dir, exist_ok=True)
cmap = matplotlib.colormaps['spring']
max_plot_length = "90D"

#for i, cluster_id in [(1, (35, 0))]:
for i, cluster_id in enumerate(outage_attributes.cluster_id.unique()):
    
    if -1 in cluster_id:
        continue  # couldn't cluster, usually noise
    
    event = outage_attributes[outage_attributes.cluster_id == cluster_id]
        
    time_cluster_id, = event.time_cluster_id.unique()
    geo_cluster_id, = event.geo_cluster_id.unique()
    geo_cluster_id = int(geo_cluster_id)
        
    f, ax = plt.subplots(figsize=(16, 10))
    
    ax.axhline(1 - outage_threshold, ls="--", color="white", label="Outage threshold")
    
    for outage_attr in event.itertuples():

        county_data_hourly: pd.DataFrame = all_counties_hourly.loc[(slice(None), outage_attr.CountyFIPS), :]

        event_duration = pd.Timedelta(resample_freq) * outage_attr.n_periods
        if event_duration > pd.Timedelta(max_plot_length):
            continue  # do not plot counties with outages over 90 days, this is probably an error

        # add a buffer around the start and end of the run            
        event_start_datetime = pd.to_datetime(outage_attr.start)
        plot_start: str = str((event_start_datetime - pd.Timedelta(start_buffer)).date())
        event_end_datetime = event_start_datetime + event_duration
        plot_end: str = str((event_end_datetime + pd.Timedelta(end_buffer)).date())

        county_name, state_name, state_code = us_admin_name(outage_attr.CountyFIPS, county_boundaries, states)
            
        # select our hourly data to plot
        try:
            label_str = f"{county_name}, {states.loc[int(state_code), 'state_alpha_2_code']}"
        except Exception as e:
            label_str = f"{county_name}, ?"
        timeseries = 1 - county_data_hourly.droplevel(1).loc[plot_start: plot_end, "OutageFraction"]
        timeseries.plot(
            ax=ax,
            x_compat=True,  # enforce standard matplotlib date tick labelling "2023-09-21"
            label=label_str,
            color=cmap(hash(label_str) % 100 / 100)
        )

    ax.set_ylabel("1 - Fraction of customers in county without power", labelpad=20)
    ax.set_xlabel("Time", labelpad=20)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"POUS outage cluster {cluster_id}")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if len(handles) < 30:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(1.08, 0.98),
            ncols=max(1, int(np.ceil(len(event) / 35))),
            loc="upper right",
            prop={'size':7}
        )
     
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
    
    # inset map of county centres
    ax_map = f.add_axes([0.73, 0.1, 0.3, 0.2]) 
    affected_counties = county_boundaries[county_boundaries.GEOID.isin(event.CountyFIPS)]
    affected_counties.loc[:, ["GEOID", "geometry"]].merge(
        event.loc[:, ["CountyFIPS", "days_since_data_start"]],
        left_on="GEOID",
        right_on="CountyFIPS"
    ).plot(
        column="days_since_data_start",
        cmap="Blues",
        ax=ax_map
    )
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    usa = world[world.iso_a3 == "USA"]
    usa.boundary.plot(ax=ax_map, alpha=0.5)
    ax_map.grid(alpha=0.2)
    ax_map.set_xlim(-130, -65)
    ax_map.set_ylim(22, 53)
    ax_map.set_ylabel("Latitude [deg]")
    ax_map.yaxis.set_label_position("right")
    ax_map.yaxis.tick_right()
    ax_map.set_xlabel("Longitude [deg]")
    
    # save to disk
    f.savefig(
        os.path.join(
            plot_dir,
            f"{time_cluster_id=:d}_{geo_cluster_id=:d}_{plot_start=}_{plot_end=}.png"
        )
    )
    plt.close(f)


# In[ ]:


outage_attributes[outage_attributes.cluster_id == (15,2)]

