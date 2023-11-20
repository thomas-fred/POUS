import datetime
import os
import multiprocess
from typing import Any
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
import snakemake

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
    cluster_name: str,
    cluster: pd.DataFrame,
    pop_affected: pd.DataFrame,
    country: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    plot_dir: str,
):

    plt.style.use('dark_background')  # for cool points
    cmap = matplotlib.colormaps['spring']

    f, ax = plt.subplots(figsize=(16, 8))

    plot_start = pop_affected.index.min()
    plot_end = pop_affected.index.max()

    pop_affected.plot(
        stacked=True,
        legend=False,
        x_compat=True,  # daily x ticks
        ax=ax,
        cmap=cmap,
        alpha=0.5,
        drawstyle="steps-post",
        linewidth=1
    )

    ax.set_ylabel("Population disconnected", labelpad=20)
    ax.set_xlabel("Time", labelpad=20)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.grid(alpha=0.3, which="major")
    peak_pop_affected: int = int(np.round(pop_affected.sum(axis=1).max()))
    pop_hours_lost: int = int(np.round(cluster.pop_hours_supply_lost.sum()))
    ax.set_title(
        f"Outage cluster {cluster_name}\n\n"
        f"Peak population affected: {peak_pop_affected:,d}\n"
        f"Person-hours supply lost: {pop_hours_lost:,d}"
    )

    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    # inset map of county centres
    ax_map = f.add_axes([0.62, 0.66, 0.4, 0.3])
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
    filename = f"{cluster_name}_{str(plot_start).replace(' ', '_')}_{str(plot_end).replace(' ', '_')}.png"
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
    event_magnitude: float,
    population_hours_lost: float,
    county_hourly: pd.DataFrame,
    county_resampled: pd.DataFrame,
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

    # shade the area of the integral
    outage_data = county_resampled.loc[event_start: event_end]
    ax.fill_between(
        outage_data.index,
        outage_data,
        np.ones_like(outage_data.values),
        step="post",
        alpha=0.4,
        label="Outage"
    )

    ax.axvline(event_start, label="Event start", ls="--", color="green")
    ax.axvline(event_end, label="Event end", ls="--", color="red")

    county_name, state_name, state_alpha_code = us_county_name(county_code, counties, states)

    # select our hourly data to plot
    admin_str = f"{county_name}, {state_alpha_code}"
    ax.step(
        county_hourly.index.values,
        county_hourly.values,
        label="Hourly timeseries",
        where="post",
    )
    if len(county_hourly) != len(county_resampled):
        ax.step(
            county_resampled.index.values,
            county_resampled.values,
            label="Resampled timeseries",
            where="post",
        )

    ax.set_ylabel("1 - Fraction of customers in county without power", labelpad=20)
    ax.set_xlabel("Time", labelpad=20)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3, which="both")
    duration_hours: int = int(np.round(event_duration.total_seconds() / (60 * 60)))
    ax.set_title(
        "Electricity outage event\n"
        f"{event_start}, {admin_str} ({county_code})\n\n"
        f"{duration_hours:,d} hours duration, {int(np.round(population_hours_lost)):,d} person-hours supply lost\n"
        f"{event_magnitude:.2f} magnitude, {event_magnitude / duration_hours:.2f} magnitude / duration"
    )
    ax.legend(loc="lower right")
    filename = f"{event_start.date()}_{county_code}.png"
    filepath = os.path.join(plot_dir, filename)
    print(filepath)
    f.savefig(filepath)
    plt.close(f)


def plot_magnitude_duration_density(events: pd.DataFrame, wildcards: snakemake.io.Wildcards, path: str, xscale: str = "log") -> None:
    """
    Plot the event density of the duration normalised magnitude as a function of duration.
    """
    n_bins_x = int(np.round(np.cbrt(len(events))))

    # run hexbinning once to find the counts per bin
    g, g_ax = plt.subplots(figsize=(12, 8))
    hexbin_counts = g_ax.hexbin(events.duration_hours, events.integral_norm, gridsize=n_bins_x, xscale=xscale, yscale="log").get_array()
    plt.close(g)

    f, ax = plt.subplots(figsize=(18,11))

    # use the counts to run again with a colour normalisation that saturates at p95
    hexbin = ax.hexbin(
        events.duration_hours,
        events.integral_norm,
        gridsize=n_bins_x,
        cmap=matplotlib.colormaps["magma"],
        xscale=xscale,
        yscale="log",
        norm=matplotlib.colors.LogNorm(
            vmin=1,
            vmax=np.quantile(hexbin_counts, 0.96)
        ),
        mincnt=1,
    )
    cbar = f.colorbar(hexbin, ax=ax, label='Frequency', extend="max")
    ax.set_ylabel("Time-integrated outage magnitude / Outage duration")
    ax.set_xlabel("Outage duration [hours]")
    max_hours = events.duration_hours.max()
    duration_label = [(24, "Day"), (24 * 7, "Week"), (24 * 31, "Month")]
    for duration, label in duration_label:
        if max_hours > duration:
            ax.axvline(duration, ls="--", alpha=0.5)
            ax.text(
                duration * 1.05,
                0.9 * events.integral_norm.max(),
                label,
                horizontalalignment="left",
                verticalalignment="top",
                rotation=90,
            )
    ax.grid(which="both", alpha=0.2)
    ax.set_title(
        f"Temporal binning: {wildcards.RESAMPLE_FREQ}, outage threshold: {wildcards.THRESHOLD}\n"
        f"Number of county outage events: {len(events):,}"
    )
    f.savefig(path)


def plot_events_summary(
    wildcards: snakemake.io.Wildcards,
    events: pd.DataFrame,
    county_boundaries: gpd.GeoDataFrame,
    usa: pd.Series,
    frequency_map_path: str,
    duration_histogram_path: str,
    duration_magnitude_norm_scatter_path: str,
    duration_magnitude_norm_significant_scatter_path: str,
) -> None:
    """
    Plots that summarise identified events.
    """

    plt.style.use('dark_background')  # for cool points

    events["geometry"] = gpd.points_from_xy(events.longitude, events.latitude)
    events = gpd.GeoDataFrame(events)

    f, ax = plt.subplots(figsize=(12,8))
    event_count = events.loc[:, ["CountyFIPS", "n_periods"]].groupby("CountyFIPS").sum()
    counties = county_boundaries.loc[:, ["GEOID", "geometry"]].set_index("GEOID")
    outage_events_per_county = event_count.merge(counties, left_on="CountyFIPS", right_on="GEOID")
    outage_events_per_county = gpd.GeoDataFrame(outage_events_per_county)
    outage_events_per_county.plot(
        column="n_periods",
        ax=ax,
        cmap=matplotlib.colormaps["spring"],
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=outage_events_per_county.n_periods.max()),
        legend=True,
        legend_kwds={
            "label": f"Number of {wildcards.RESAMPLE_FREQ} periods county experienced outage",
            "shrink": 0.82,
        }
    )
    usa.boundary.plot(ax=ax, alpha=0.5)
    ax.set_title(
        f"Temporal binning: {wildcards.RESAMPLE_FREQ}, outage threshold: {wildcards.THRESHOLD}\n"
        f"Number of county outage events: {len(events):,}"
    )
    ax.grid(alpha=0.2)
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 53)
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlabel("Longitude [deg]")
    f.savefig(frequency_map_path)

    f, ax = plt.subplots(figsize=(12,8))
    freq, bins, patches = ax.hist(events.duration_hours, bins=50, alpha=0.6, label="Distribution")
    max_hours = events.duration_hours.max()
    duration_label = [(24, "Day"), (24 * 7, "Week"), (24 * 31, "Month")]
    for duration, label in duration_label:
        if max_hours > duration:
            ax.axvline(duration, ls="--", alpha=0.5)
            ax.text(
                duration + 0.01 * max_hours,
                0.9 * max(freq),
                label,
                horizontalalignment="left",
                verticalalignment="top",
                rotation=90,
            )
    ax.set_yscale("log")
    ax.set_ylabel("Freqency")
    ax.set_xlabel("Outage duration [hours]")
    ax.grid(alpha=0.2)
    ax.set_title(
        f"Temporal binning: {wildcards.RESAMPLE_FREQ}, outage threshold: {wildcards.THRESHOLD}\n"
        f"Number of county outage events: {len(events):,}"
    )
    f.savefig(duration_histogram_path)

    events["integral_norm"] = events.integral / events.duration_hours

    plot_magnitude_duration_density(events, wildcards, duration_magnitude_norm_scatter_path,)

    events = events[events.integral_norm > 0.05]
    events = events[events.duration_hours > 3]
    plot_magnitude_duration_density(events, wildcards, duration_magnitude_norm_significant_scatter_path)
