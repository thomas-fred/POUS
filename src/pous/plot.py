import datetime
import os
import multiprocess
import warnings

import geopandas as gpd
from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from .admin import us_county_name


def map_outage_at_time(
    outage_data: gpd.GeoDataFrame,
    timestamp: str,
    quantile_timeseries: dict,
    bbox: np.ndarray[float],
    title: str,
    plot_dir: str
) -> None:
    """
    Chloropleth map `outage_data.OutageFraction` in between 0 and 1 for `outage_data.geometry` polygons
    at `timestamp` time. Also plot inset of whole event timeseries as inset, with current time highlighted.

    Args:
        outage_data: Table of `OutageFraction` data with a DatetimeIndex and `geometry`.
        timestamp: Timestamp we wish to map
        quantile_timeseries: Dict of quantile level -> pandas DataFrame containing at least `index` and `OutageFraction`
        bbox: Map bounding box; iterable of min_x, min_y, max_x, max_y
        title: Figure title (N.B. map axis has timestamp as title)
        plot_dir: Location to save map
    """

    filepath = os.path.join(plot_dir, timestamp.replace(" ", "_") + ".png")
    if os.path.exists(filepath):
        return

    outage_mask = outage_data.OutageFraction > 0.01
    outage_data.loc[~outage_mask, "OutageFraction"] = -1

    bounds = np.array([0, 0.05, 0.2, 0.4, 0.6, 0.8, 1])
    n_bins = len(bounds)
    cmap = plt.cm.get_cmap("inferno", n_bins + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    colours = outage_data.OutageFraction.apply(cmap)

    f, ax = plt.subplots(figsize=(14, 9))
    ax_line = ax.inset_axes([0.1, 0.12, 0.35, 0.25])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    outage_data.plot(
        ax=ax,
        color=colours,
        cax=cax,
    )

    ax.set_xlabel("Longitude", labelpad=15)
    ax.set_ylabel("Latitude", labelpad=15)
    ax.grid(alpha=0.2)
    ax.set_title(timestamp)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    borders = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    borders.plot(ax=ax, facecolor="none", edgecolor="grey", alpha=0.2)

    min_x, min_y, max_x, max_y = bbox
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    cb = matplotlib.colorbar.ColorbarBase(
        cax,
        cmap=cmap,
        norm=norm,
        spacing='proportional',
        ticks=bounds,
        boundaries=bounds,
        format='%.2f'
    )
    cax.set_ylabel("Outage fraction", labelpad=10)

    quantile_colours = {
        # less than median not visible on linear plot
        0.5: "lightcyan",
        0.75: "cyan",
        0.9: "mediumturquoise",
        0.95: "mediumseagreen",
        0.99: "olivedrab"
    }
    for quantile, timeseries in quantile_timeseries.items():
        ax_line.plot(timeseries.index, timeseries.OutageFraction, label=f"{quantile}", c=quantile_colours[quantile])
    ax_line.axvline(timeseries.index[timeseries.index == timestamp])
    ax_line.set_ylim(0, 1)
    ax_line.set_ylabel("Outage fraction")
    legend = ax_line.legend(prop={'size': 6}, loc="upper right")
    legend.set_title(title="Quantile", prop={'size': 8})
    ax_line.tick_params(axis='x', labelsize=8)
    ax_line.set_title("Distribution of county outage rates")
    plt.setp(ax_line.get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    f.suptitle(title)

    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)

    f.savefig(filepath)
    plt.close(f)

    return


def map_outage(outage: gpd.GeoDataFrame, event_name: str, n_cpu: int, title: str, plot_dir: str) -> None:
    """
    Map `outage.OutageFraction` on `outage.geometry`. Produces frames for each timestamp
    and then animates them.

    Args:
        outage: Table containing outage information. Must include `geometry` and
            `OutageFraction`, a float between 0 and 1.
        event_name: Used as title and filename of frames and animation.
        n_cpu: Number of processors used to create individual frames in parallel.
        title: Figure title (N.B. map axis has timestamp as title).
        plot_dir: Where to place event subfolder containing plot files.
    """

    # compute a map window from the bbox that encompasses all features, plus a buffer
    buffer_fraction = 0.05
    min_x, min_y, max_x, max_y = outage.total_bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    buffer = buffer_fraction * max([span_x, span_y])
    bbox = [min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer]

    # compute the timeseries once and pass in to plotter
    quantile_timeseries = {}
    for quantile in [0.5, 0.75, 0.9, 0.95, 0.99]:
        quantile_timeseries[quantile] = outage[["OutageFraction"]].groupby("RecordDateTime").quantile(quantile)

    event_dir = os.path.join(plot_dir, event_name)
    if not os.path.exists(event_dir):
        os.makedirs(event_dir)

    print(f"Making frames with {n_cpu=}...")
    args = []
    for timestamp in sorted(set(outage.index)):
        args.append((outage.loc[timestamp], str(timestamp), quantile_timeseries, bbox, title, event_dir))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The get_cmap function was deprecated")
        with multiprocess.Pool(processes=n_cpu) as pool:
            pool.starmap(map_outage_at_time, args)

    print("Animating into .gif with imagemagick...")
    plot_paths = glob(os.path.join(event_dir, "*.png"))
    os.system(f"convert -delay 20 {' '.join(sorted(plot_paths))} {os.path.join(plot_dir, f'{event_name}.gif')}")

    print("Done")
    return


def plot_event_cluster(
    cluster_id: tuple[int, int],
    events: pd.DataFrame,
    hourly: pd.DataFrame,
    outage_threshold: float,
    country: gpd.GeoDataFrame,
    resample_freq: str,
    counties: gpd.GeoDataFrame,
    states: pd.DataFrame,
    plot_dir: str,
):

    plt.style.use('dark_background')  # for cool points

    max_plot_length = "60D"
    start_buffer = "2D"
    end_buffer = "5D"
    cmap = matplotlib.colormaps['spring']

    if -1 in cluster_id:
        return  # couldn't cluster, usually noise

    cluster = events[events.cluster_id == cluster_id]

    f, ax = plt.subplots(figsize=(16, 10))
    ax.axhline(1 - outage_threshold, ls="--", color="white", label="Outage threshold")

    for outage_attr in cluster.itertuples():

        event_duration = pd.Timedelta(resample_freq) * outage_attr.n_periods
        if event_duration > pd.Timedelta(max_plot_length):
            print(f"{event_duration=} > {max_plot_length=} for {cluster_id=}, skipping")
            continue

        # add a buffer around the start and end of the run
        event_start_datetime = pd.to_datetime(outage_attr.event_start)
        plot_start: str = str((event_start_datetime - pd.Timedelta(start_buffer)).date())
        event_end_datetime = event_start_datetime + event_duration
        plot_end: str = str((event_end_datetime + pd.Timedelta(end_buffer)).date())

        county_hourly: pd.DataFrame = hourly.loc[(slice(plot_start, plot_end), outage_attr.CountyFIPS), :]
        county_name, state_name, state_code = us_county_name(outage_attr.CountyFIPS, counties, states)

        # select our hourly data to plot
        try:
            label_str = f"{county_name}, {states.loc[int(state_code), 'state_alpha_2_code']}"
        except Exception as e:
            label_str = f"{county_name}, ?"
        timeseries = 1 - county_hourly.droplevel(1).loc[:, "OutageFraction"]
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
            ncols=max(1, int(np.ceil(len(cluster) / 35))),
            loc="upper right",
            prop={'size':7}
        )

    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    # inset map of county centres
    ax_map = f.add_axes([0.73, 0.1, 0.3, 0.2])
    affected_counties = counties[counties.GEOID.isin(cluster.CountyFIPS)]
    affected_counties.loc[:, ["GEOID", "geometry"]].merge(
        cluster.loc[:, ["CountyFIPS", "days_since_data_start"]],
        left_on="GEOID",
        right_on="CountyFIPS"
    ).plot(
        column="days_since_data_start",
        cmap="Blues",
        ax=ax_map
    )
    country.boundary.plot(ax=ax_map, alpha=0.5)
    ax_map.grid(alpha=0.2)
    ax_map.set_xlim(-130, -65)
    ax_map.set_ylim(22, 53)
    ax_map.set_ylabel("Latitude [deg]")
    ax_map.yaxis.set_label_position("right")
    ax_map.yaxis.tick_right()
    ax_map.set_xlabel("Longitude [deg]")

    # save to disk
    time, space = cluster_id
    filename = f"{time}_{space}_{plot_start}_{plot_end}.png"
    filepath = os.path.join(plot_dir, filename)
    print(filepath)
    f.savefig(filepath)
    plt.close(f)


def plot_event(
    outage_threshold: float,
    event_start: datetime.date,
    event_end: datetime.date,
    county_code: str,
    event_duration: pd.Timedelta,
    county_hourly: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    states: pd.DataFrame,
    plot_dir: str
):
    """
    Plot timeseries of single event, along with inferred event start and end.
    """

    plt.style.use('dark_background')  # for cool points
    cmap = matplotlib.colormaps['spring']

    f, ax = plt.subplots(figsize=(16, 10))
    ax.axhline(1 - outage_threshold, ls="--", color="white", label="Outage threshold")

    ax.axvline(event_start, label="Event start", ls="--", color="green")
    ax.axvline(event_end, label="Event end", ls="--", color="red")

    county_name, state_name, state_code = us_county_name(county_code, counties, states)

    # select our hourly data to plot
    try:
        admin_str = f"{county_name}, {states.loc[int(state_code), 'state_alpha_2_code']}"
    except Exception as e:
        admin_str = f"{county_name}, ?"
    timeseries = 1 - county_hourly.droplevel(1).loc[:, "OutageFraction"]
    timeseries.plot(
        ax=ax,
        x_compat=True,  # enforce standard matplotlib date tick labelling "2023-09-21"
        label="Outage timeseries",
    )

    ax.set_ylabel("1 - Fraction of customers in county without power", labelpad=20)
    ax.set_xlabel("Time", labelpad=20)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3, which="both")
    duration_days: float = event_duration.total_seconds() / (60 * 60 * 24)
    ax.set_title(
        f"POUS outage {admin_str}\n"
        f"{duration_days:.2f} days"
    )
    ax.legend()
    filename = f"{event_start.date()}_{county_code}.png"
    filepath = os.path.join(plot_dir, filename)
    print(filepath)
    f.savefig(filepath)
    plt.close(f)
