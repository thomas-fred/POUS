import os
import multiprocess
import warnings

import geopandas as gpd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


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
