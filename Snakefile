"""
Workflow for analysing Power Outage US (POUS) data.
"""

number_regex = "[-+]?\d*\.?\d+|[-+]?\d+"
wildcard_constraints:
    YEAR="\d{4}",
    RESAMPLE_FREQ="\d+(?:H|D|W)",
    THRESHOLD=number_regex,
    TIME_DAYS=number_regex,
    SPACE_DEG=number_regex,

YEARS = range(2017, 2023)  # years of available data


rule extract_raw_outage_csv:
    """
    Extract CSV files and rename to their year.
    """
    input:
        archive = "data/input/outage/POUS_Export_CountyByUtility_Hourly_{YEAR}04-{YEAR}10.zip"
    output:
        csv = temp("data/output/outage/by_year/{YEAR}.csv")
    shell:
        """
        TEMP_DIR=$(mktemp -d)
        unzip {input.archive} -d $TEMP_DIR
        mv $TEMP_DIR/*.csv {output.csv}
        """


rule parse_raw_outage_csv:
    """
    Read from CSV, make datetime/county index, drop superfluous columns, save as parquet.
    """
    input:
        csv = rules.extract_raw_outage_csv.output.csv
    output:
        cleaned = "data/output/outage/by_year/{YEAR}.pq"
    run:
        from pous.io import parse_pous_csv

        parse_pous_csv(input.csv).to_parquet(output.cleaned)


rule resample_outages:
    """
    Take all years, gap-fill with zeros, resample to desired frequency and concatenate.
    """
    input:
        years = expand(
            "data/output/outage/by_year/{year}.pq",
            year=YEARS
        )
    output:
        resampled = "data/output/outage/{RESAMPLE_FREQ}/timeseries.pq"
    run:
        import pandas as pd
        from tqdm import tqdm

        hourly_with_gaps = pd.concat(
            [pd.read_parquet(path, columns=["OutageFraction", "CustomersTracked"]) for path in input.years]
        )
        counties = sorted(hourly_with_gaps.index.get_level_values("CountyFIPS").unique())

        datetimes = hourly_with_gaps.index.get_level_values(0)
        start_year = datetimes.min().year
        end_year = datetimes.max().year

        gap_filled_resampled = []
        for county_code in tqdm(counties):

            try:
                data = hourly_with_gaps.loc(axis=0)[:, county_code].reset_index(level="CountyFIPS")
                complete_index = pd.date_range(
                    f"{start_year}-01-01",
                    f"{end_year}-12-31",
                    freq="1H",
                )
                data = data.reindex(index=complete_index, fill_value=0)
                data.index.name = "RecordDateTime"
            except KeyError:
                continue

            data = data.drop(columns=["CountyFIPS"]).resample(wildcards.RESAMPLE_FREQ).mean()
            data["CountyFIPS"] = county_code
            gap_filled_resampled.append(data)

        resampled = pd.concat(gap_filled_resampled)
        resampled = resampled.set_index([resampled.index, resampled.CountyFIPS]).drop(columns=["CountyFIPS"])
        resampled = resampled.sort_index(level=["RecordDateTime", "CountyFIPS"])
        print(resampled)
        resampled.to_parquet(output.resampled)


rule parse_county_population:
    """
    Parse the US census county population data.
    """
    input:
        data = "data/input/counties/population/co-est2022-alldata.csv",
    output:
        parsed = "data/input/counties/population/2022.pq"
    run:
        import pandas as pd

        data = pd.read_csv(input.data, usecols=["SUMLEV", "STATE", "COUNTY", "CTYNAME", "POPESTIMATE2022"])
        data = data.rename(columns={"CTYNAME": "CountyName", "POPESTIMATE2022": "CountyPop2022"})

        # filter to only county rows (remove e.g. state total rows)
        data = data[data.SUMLEV == 50]
        data = data.drop(columns=["SUMLEV"])

        data["CountyFIPS"] = data.apply(lambda row: f"{row.STATE:02d}{row.COUNTY:03d}", axis=1).astype(str)
        data.loc[:, ["CountyFIPS", "CountyPop2022", "CountyName"]].to_parquet(output.parsed)


rule identify_events:
    """
    Look for events where OutageFraction exceeded threshold in resampled data.
    """
    input:
        resampled = rules.resample_outages.output.resampled,
        counties = "data/input/counties/geometry/cb_2018_us_county_500k.shp",
        county_pop = rules.parse_county_population.output.parsed,
    output:
        events = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/events.pq",
    run:
        from typing import Tuple

        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        # county timeseries must have at least this duration (after a potential outage end)
        # operating nominally before an outage may be considered over
        min_time_nominal: pd.Timedelta = pd.Timedelta("2D")

        # duration before outage start to compute average CustomersTracked for
        pre_outage_window: pd.Timedelta = pd.Timedelta("1W")

        counties = gpd.read_file(input.counties)
        county_pop = pd.read_parquet(input.county_pop)
        # annotate counties table with population data, `CountyPop2022`
        counties = counties.merge(county_pop, left_on="GEOID", right_on="CountyFIPS").drop(columns=["CountyFIPS", "CountyName"])
        counties = counties.set_index("GEOID")

        # take the resampled data and filter to periods with OutageFraction above a threshold
        resampled = pd.read_parquet(input.resampled, columns=["OutageFraction", "CustomersTracked"])

        data_start: pd.Timestamp = resampled.index.get_level_values(level="RecordDateTime").min()

        resample_period: pd.Timedelta = pd.Timedelta(wildcards.RESAMPLE_FREQ)
        resample_period_ns: float = resample_period.total_seconds() * 1E9
        day_in_ns: float = 1E9 * 60 * 60 * 24

        def approx_equal_period(period_a: float, period_b: float, rtol: float = 0.1) -> bool:
            """Check two durations are within `rtol`."""
            return np.abs((period_a / period_b) - 1) < rtol

        def start_end_datetimes(index: pd.DatetimeIndex, start_index: int, end_index: int, period: pd.Timedelta) -> Tuple[pd.Timestamp, pd.Timestamp]:
            """Lookup the start (we label left) and end (we label right) times in `index`."""
            return index[start_index], index[end_index] + period

        def next_duration_nominal(outage_timeseries: pd.DataFrame, from_time: pd.Timestamp, duration: pd.Timedelta) -> bool:
            """Check no outage states in `outage_timeseries` for `duration` from `from_time`."""
            if outage_timeseries.loc[from_time: from_time + duration, :].empty:
                return True
            else:
                return False

        events = []
        for county_code in tqdm(resampled.index.get_level_values("CountyFIPS").unique()):

            # all county data at this resampling period
            county_resampled: pd.DataFrame = resampled.loc[(slice(None), county_code), :]
            # county data at this resampling period in excess of outage threshold
            county_resampled_outage: pd.DataFrame = county_resampled[county_resampled.OutageFraction > float(wildcards.THRESHOLD)]

            # drop superfluous indicies
            county_resampled: pd.DataFrame = county_resampled.reset_index(level="CountyFIPS")
            county_resampled_outage: pd.DataFrame = county_resampled_outage.reset_index(level="CountyFIPS")

            # lookup county geometry centroid
            try:
                county_centroid = counties.loc[county_code].geometry.centroid
                county_pop = counties.loc[county_code].CountyPop2022
            except KeyError:
                print(f"Missing {county_code=}, skipping...")

            # can't take a diff along one row
            if len(county_resampled_outage) < 2:
                continue

            # picking out runs of resampled outage periods
            run_start_index = 0
            outages_start_end: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for i, time_gap_ns in enumerate(np.diff(county_resampled_outage.index.values)):

                if approx_equal_period(float(time_gap_ns), resample_period_ns):
                    # the time period between these two resampled periods (in excess of the threshold)
                    # is approximately equal to the resampling period, ergo, this is a continued outage
                    continue

                else:
                    if next_duration_nominal(county_resampled_outage, county_resampled_outage.index[i] + resample_period, min_time_nominal):
                        # next timestep in thresholded outage data is not equal to resample period,
                        # this is the end of an outage, record start and end datetimes
                        outages_start_end.append(start_end_datetimes(county_resampled_outage.index, run_start_index, i, resample_period))

                        # reset start index (moving to next outage event)
                        run_start_index = i + 1
                    else:
                        # there is now a run of resampled periods in nominal state ahead of us,
                        # but not enough to clear min_time_nominal, continue logging outage
                        continue

            else:
                # if we're still in an outage state at the end of the data, record it here
                if approx_equal_period(float(time_gap_ns), resample_period_ns):
                    outages_start_end.append(start_end_datetimes(county_resampled_outage.index, run_start_index, i + 1, resample_period))

            # start of first outage bin (labelled left), end of last outage bin (labelled right)
            for event_start, event_end in outages_start_end:

                # N.B. bins are generally time labelled left

                pre_outage_window_start = pd.to_datetime(event_start) - pre_outage_window
                pre_outage_tracked_customers: float = county_resampled_outage.loc[pre_outage_window_start: event_start, "CustomersTracked"].mean()

                duration: pd.Timedelta = event_end - event_start
                duration_hours: float = duration.total_seconds() / (60 * 60)

                # in units of hours
                outage_magnitude: float = county_resampled_outage.loc[event_start: event_end, "OutageFraction"].sum()

                n_periods = duration / resample_period
                assert int(n_periods) == n_periods

                # ensure the number of tracked customers is no less than 5% of the county population
                # note that customers are very likely households, not inhabitants
                if pre_outage_tracked_customers / county_pop < 0.05:
                    # event has very few tracked customers relative to county population, discard
                    continue

                events.append(
                    (
                        county_code,
                        county_centroid.x,
                        county_centroid.y,
                        county_pop,
                        pre_outage_tracked_customers,
                        event_start,
                        (event_start - data_start).value / day_in_ns,
                        duration_hours,
                        int(n_periods),
                        outage_magnitude,
                        outage_magnitude * county_pop,
                    )
                )

        events = pd.DataFrame(
            events,
            columns=[
                "CountyFIPS",
                "longitude",
                "latitude",
                "county_pop",
                "pre_outage_tracked_customers",
                "event_start",
                "days_since_data_start",
                "duration_hours",
                "n_periods",
                "integral",
                "pop_hours_supply_lost",
            ]
        )
        events = events.sort_values("days_since_data_start").reset_index(drop=True)
        print(events)
        events.to_parquet(output.events)


rule plot_events_summary:
    """
    Plot maps, scatters and histograms of event set.
    """
    input:
        events = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/events.pq",
        counties = "data/input/counties/geometry/cb_2018_us_county_500k.shp",
        countries = "data/input/countries/ne_110m_admin_0_countries.shp",
    output:
        frequency_map = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_frequency_map.png",
        duration_histogram = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_duration_histogram.png",
        duration_magnitude_norm_density = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_duration_magnitude_norm_density.png",
        duration_magnitude_norm_significant_density = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_duration_magnitude_norm_significant_density.png",
    run:
        import geopandas as gpd
        import pandas as pd

        from pous.plot import plot_events_summary

        events = pd.read_parquet(input.events)
        counties = gpd.read_file(input.counties)
        countries = gpd.read_file(input.countries)

        plot_events_summary(
            wildcards,
            events,
            counties,
            countries[countries.ISO_A3 == "USA"],
            output.frequency_map,
            output.duration_histogram,
            output.duration_magnitude_norm_density,
            output.duration_magnitude_norm_significant_density,
        )


rule cluster_events_by_storm:
    """
    Join event set with storm track data, plot outages associated with each storm.

    Save aggregate outage statistics for storms.
    """
    input:
        events = rules.identify_events.output.events,
        hourly = "data/output/outage/1H/timeseries.pq",
        storm_tracks = "data/input/storm_tracks/IBTrACS.gpq",
        counties = "data/input/counties/geometry/cb_2018_us_county_500k.shp",
        states = "data/input/states/state_codes.csv",
        countries = "data/input/countries/ne_110m_admin_0_countries.shp",
    output:
        plots = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/storm_cluster_plots"),
        storm_cluster_summary = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/storm_clusters.gpq",
        storm_clusters = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/storm_clusters"),
    run:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import shapely

        from pous.plot import plot_event_cluster

        print("Reading input data...")
        events = pd.read_parquet(input.events)
        tracks = gpd.read_parquet(input.storm_tracks)
        counties = gpd.read_file(input.counties)
        countries = gpd.read_file(input.countries)
        states = pd.read_csv(input.states)
        hourly = pd.read_parquet(input.hourly)

        # filter to relevant timespan
        tracks = tracks.sort_index().loc[str(min(YEARS)): str(max(YEARS))]

        # filter spatially
        usa = countries[countries.ISO_A3 == "USA"]
        aoi_buffer_deg = 2
        area_of_interest, = usa.geometry.buffer(aoi_buffer_deg).values
        tracks = tracks[tracks.within(area_of_interest)]

        events["geometry"] = gpd.points_from_xy(events.longitude, events.latitude)
        events = gpd.GeoDataFrame(events.drop(columns=["longitude", "latitude"]))
        events = events.set_index("event_start").sort_index()

        os.makedirs(output.plots, exist_ok=True)
        os.makedirs(output.storm_clusters, exist_ok=True)

        print("Plotting...")
        storm_clusters = []
        for track_id, track in tracks.groupby(tracks.track_id):

            try:
                track_line = gpd.GeoDataFrame({"geometry": shapely.LineString(track.geometry)}, index=[0])
            except shapely.errors.GEOSException:
                continue

            track_line_buffered = track_line.copy(deep=True)
            track_line_buffered["geometry"] = track_line.geometry.buffer(2)

            # track points have been filtered to within an `aoi_buffer_deg` of US coastline
            # allow for events up to 2D prior to storm eye crossing into this buffer
            plot_start = track.index.min() - pd.Timedelta("2D")

            # find events spatially within buffered track polygon
            # then filter to those starting between start of track and 3D after end of track
            cluster = events.loc[plot_start: track.index.max() + pd.Timedelta("3D")].sjoin(track_line_buffered).reset_index()

            if cluster.empty:
                print(f"No intersection for {track_id=}, skipping...")
                continue

            storm_name, = track.name.drop_duplicates()

            # time 3 days after end of last county-outage event
            plot_end = (cluster.event_start + cluster.duration_hours.apply(lambda d: d * pd.Timedelta("1H"))).max() + pd.Timedelta("3D")

            county_hourly: pd.DataFrame = hourly.loc[(slice(plot_start, plot_end), cluster.CountyFIPS.unique()), ["OutageFraction"]]
            county_population: pd.Series = cluster.loc[cluster.CountyFIPS.drop_duplicates().index].set_index("CountyFIPS").loc[:, "county_pop"]
            pop_affected: pd.DataFrame = county_hourly.mul(county_population, level="CountyFIPS", axis="index")
            pop_affected = pop_affected.reset_index(level=1).pivot(columns=["CountyFIPS"])

            peak_pop_affected: int = int(np.round(pop_affected.sum(axis=1).max()))
            pop_hours_lost: int = int(np.round(cluster.pop_hours_supply_lost.sum()))

            track_dir = os.path.join(output.storm_clusters, track_id)
            os.makedirs(track_dir, exist_ok=True)
            track_line.to_parquet(os.path.join(track_dir, "track.gpq"))
            cluster.to_parquet(os.path.join(track_dir, "events.gpq"))
            pop_affected.to_parquet(os.path.join(track_dir, "pop_affected.pq"))

            storm_clusters.append(
                (
                    storm_name,
                    track_id,
                    plot_start,
                    peak_pop_affected,
                    pop_hours_lost,
                    track_line.geometry.iloc[0],
                )
            )

            plot_event_cluster(
                storm_name,
                cluster,
                pop_affected,
                usa,
                counties,
                output.plots,
            )

        storm_clusters = gpd.GeoDataFrame(
            storm_clusters,
            columns=[
                "storm_name",
                "track_id",
                "start_date",
                "peak_pop_affected",
                "pop_hours_supply_lost",
                "geometry",
            ]
        )
        storm_clusters.to_parquet(output.storm_cluster_summary)


rule plot_event_duration_against_wind_speed:
    """
    Join event set with storm track data, plot outages associated with each storm.

    Save aggregate outage statistics for storms.
    """
    input:
        max_wind_field = "data/input/max_wind_fields/IBTrACS_2017-2022.nc",
        counties = "data/input/counties/geometry/cb_2018_us_county_500k.shp",
        states = "data/input/states/state_codes.csv",
        storm_cluster_summary = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/storm_clusters.gpq",
        storm_clusters = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/storm_clusters"),
    output:
        duration_wind_speed_scatter = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/duration_wind_speed_scatter.png",
        duration_wind_speed_scatter_linear = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/duration_wind_speed_scatter_linear.png",
        duration_wind_speed_density = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/duration_wind_speed_density.png",
    run:
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np
        import pandas as pd
        import xarray as xr

        plt.style.use('dark_background')  # for cool points

        print("Reading input data...")
        max_wind_fields = xr.open_dataset(input.max_wind_field).max_wind_speed
        counties = gpd.read_file(input.counties)
        states = pd.read_csv(input.states)
        storm_clusters = pd.read_parquet(input.storm_cluster_summary).set_index("track_id")

        all_events = []
        for track_id in storm_clusters.index:

            try:
                max_wind_field = max_wind_fields.sel(event_id=track_id)
            except KeyError:
                print(f"Missing wind speed data for {track_id}")
                continue

            track_dir = os.path.join(input.storm_clusters, track_id)
            track = gpd.read_parquet(os.path.join(track_dir, "track.gpq"))
            pop_affected = pd.read_parquet(os.path.join(track_dir, "pop_affected.pq"))
            events = gpd.read_parquet(os.path.join(track_dir, "events.gpq"))

            events["track_id"] = track_id
            events["longitude"] = events.geometry.x
            events["latitude"] = events.geometry.y
            events["max_wind_speed_ms"] = max_wind_field.sel(
                longitude=events.longitude.to_xarray(),
                latitude=events.latitude.to_xarray(),
                method="nearest"
            ).values

            all_events.append(events)

        all_events = pd.concat(all_events).reset_index(drop=True)

        # scatter plot
        def pop_markersize(x: np.array) -> np.array:
            """County population -> marker size"""
            return np.log10(x) ** 4.5 / 10

        track_ids = all_events.track_id.unique()
        colour_mapping = dict(zip(track_ids, matplotlib.colormaps["spring"](np.linspace(0, 1, len(all_events.track_id.unique())))))
        f, ax = plt.subplots(figsize=(16, 8))
        track_categories, _ = pd.factorize(all_events.track_id)
        ax.grid(alpha=0.2, which="both")
        for i, track_id in enumerate(track_ids):
            to_plot = all_events[all_events.track_id == track_id]
            storm_name: str = storm_clusters.loc[track_id, "storm_name"]
            storm_year: int = storm_clusters.loc[track_id, "start_date"].year
            ax.scatter(
                to_plot.max_wind_speed_ms,
                to_plot.duration_hours,
                s=pop_markersize(to_plot.pop_hours_supply_lost),
                label=f"{storm_name}, {storm_year}",
                color=colour_mapping[track_id],
                marker="o",
                facecolors="none",
                alpha=0.8,
            )
        storm_legend = ax.legend(
            prop={'size':6.5},
            ncols=1,
            loc="upper right",
            title="Storm"
        )
        for handle in storm_legend.legend_handles:
            handle.set_sizes([8])

        pop_min_10 = 4
        pop_max_10 = 7
        pop_handles = [
            # N.B. need the sqrt around the markersize for equality between scatter markers and legend markers
            Line2D(
                [],
                [],
                color="white",
                lw=0,
                marker="o",
                fillstyle="none",
                markersize=np.sqrt(pop_markersize(p)),
                label=f"$10^{int(np.log10(p)):d}$"
            )
            for p in np.logspace(pop_min_10, pop_max_10, pop_max_10 - pop_min_10 + 1)
        ]
        pop_legend = ax.legend(
            handles=pop_handles,
            title="County population",
            loc="upper left",
            ncol=len(pop_handles),
            borderpad=1.3,
            prop={'size':8}
        )

        ax.add_artist(pop_legend)
        ax.add_artist(storm_legend)

        ax.set_xlabel("Modelled maximum wind speed [ms-1]", labelpad=20)
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max * 1.15)

        ax.set_yscale("log")
        ax.set_ylabel("Outage duration [hours]", labelpad=20)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 7)

        max_hours = all_events.duration_hours.max()
        duration_label = [(24, "Day"), (24 * 7, "Week"), (24 * 31, "Month")]
        log_artists = []
        for duration, label in duration_label:
            if max_hours > duration:
                ax.axhline(duration, ls="--", alpha=0.5)
                log_artists.append(
                    ax.text(
                        0.04 * all_events.max_wind_speed_ms.max(),
                        duration * 1.15,
                        label,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )
                )

        years = set(all_events.event_start.apply(lambda dt: dt.year))
        ax.set_title(f"County level outage events {min(years)}-{max(years)}", pad=10)

        f.savefig(output.duration_wind_speed_scatter)

        # linear yscale version of same scatter plot
        ax.set_yscale("linear")
        ax.set_ylim(0, all_events.duration_hours.max() * 1.2)

        # remove previous annotations, redraw with appropriate offset for linear scale
        [artist.remove() for artist in log_artists]
        for duration, label in duration_label:
            if max_hours > duration:
                ax.text(
                    0.04 * all_events.max_wind_speed_ms.max(),
                    duration + 10,
                    label,
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )

        f.savefig(output.duration_wind_speed_scatter_linear)

        # density plot
        f, ax = plt.subplots(figsize=(16, 8))

        xscale = "linear"
        n_bins_x = int(np.round(3 * np.cbrt(len(all_events))))

        # run hexbinning once to find the counts per bin
        g, g_ax = plt.subplots(figsize=(16, 8))
        hexbin_counts = g_ax.hexbin(
            all_events.max_wind_speed_ms,
            all_events.duration_hours,
            gridsize=n_bins_x,
            xscale=xscale,
            yscale="log"
        ).get_array()
        plt.close(g)

        hexbin = ax.hexbin(
            all_events.max_wind_speed_ms,
            all_events.duration_hours,
            gridsize=n_bins_x,
            cmap=matplotlib.colormaps["magma"],
            xscale=xscale,
            yscale="log",
            norm=matplotlib.colors.LogNorm(
                vmin=0.5,
                vmax=np.quantile(hexbin_counts, 0.95)
            ),
            mincnt=1,
        )
        cbar = f.colorbar(hexbin, ax=ax, label='Frequency', extend="max")

        max_hours = all_events.duration_hours.max()
        duration_label = [(24, "Day"), (24 * 7, "Week"), (24 * 31, "Month")]
        for duration, label in duration_label:
            if max_hours > duration:
                ax.axhline(duration, ls="--", alpha=0.5)
                ax.text(
                    0.97 * all_events.max_wind_speed_ms.max(),
                    duration * 1.05,
                    label,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

        ax.grid(alpha=0.2, which="both")
        ymin, ymax = ax.set_ylim()
        ax.set_ylim(6, ymax)
        ax.set_xlabel("Modelled maximum wind speed [ms-1]", labelpad=20)
        ax.set_ylabel("Outage duration [hours]", labelpad=20)

        ax.set_title(f"County level outage event density {min(years)}-{max(years)}", pad=10)

        f.savefig(output.duration_wind_speed_density)


rule plot_storm_events_bar_chart:
    """
    Plot bar chart of person-hours of lost supply due to storms.
    """
    input:
        storm_clusters = rules.cluster_events_by_storm.output.storm_clusters,
    output:
        bar_chart = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/storm_clusters_hours_lost.png",
    run:
        import geopandas as gpd
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use('dark_background')  # for cool points

        events = gpd.read_parquet(input.storm_clusters)
        events["name_year"] = events.apply(lambda row: f"{row.storm_name}, {row.start_date.year}", axis=1)
        events = events.set_index("name_year").sort_values("pop_hours_supply_lost")
        events = events[events.pop_hours_supply_lost > 1E6]

        f, ax = plt.subplots(figsize=(16, 8))
        events.pop_hours_supply_lost.plot(kind="bar", ax=ax)
        ax.bar_label(ax.containers[0], fmt="%.2E", rotation=90, padding=10)
        ax.set_title("Largest storm-induced electricity outages")
        ax.set_yscale("log")
        ax.set_xlabel("Storm", labelpad=10)
        ax.set_ylabel("Electricity supply lost [person-hours]", labelpad=20)
        ax.grid(which="both", alpha=0.2)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 5 * ymax)
        plt.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9)
        f.savefig(output.bar_chart)


rule plot_events:
    """
    Plot county-event timeseries (no event clustering). Overlay inferred durations.
    """
    input:
        events = rules.identify_events.output.events,
        hourly = "data/output/outage/1H/timeseries.pq",
        resampled = "data/output/outage/{RESAMPLE_FREQ}/timeseries.pq",
        counties = "data/input/counties/geometry/cb_2018_us_county_500k.shp",
        states = "data/input/states/state_codes.csv",
    output:
        plots = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_plots")
    run:
        import os

        import geopandas as gpd
        import pandas as pd

        from pous.plot import plot_event

        outage_threshold = float(wildcards.THRESHOLD)
        counties = gpd.read_file(input.counties)
        states = pd.read_csv(input.states)
        events = pd.read_parquet(input.events)
        hourly = pd.read_parquet(input.hourly)
        if input.hourly == input.resampled:
            resampled = hourly
        else:
            resampled = pd.read_parquet(input.resampled)

        os.makedirs(output.plots, exist_ok=True)

        max_event_length = "60D"
        min_event_length = "1D"
        start_buffer = "2D"
        end_buffer = "5D"
        min_norm_magnitude = 0.2

        events["integral_norm"] = events.integral / events.duration_hours

        for outage_attr in events.itertuples():

            event_duration = pd.Timedelta(wildcards.RESAMPLE_FREQ) * outage_attr.n_periods

            if event_duration > pd.Timedelta(max_event_length):
                print(f"{event_duration=} > {max_event_length=}, skipping")
                continue

            if event_duration < pd.Timedelta(min_event_length):
                print(f"{event_duration=} < {min_event_length=}, skipping")
                continue

            if outage_attr.integral_norm < min_norm_magnitude:
                print(f"{outage_attr.integral_norm=:.3f} < {min_norm_magnitude=}, skipping")
                continue

            event_start_datetime = pd.to_datetime(outage_attr.event_start)
            plot_start: str = str((event_start_datetime - pd.Timedelta(start_buffer)).date())
            event_end_datetime = event_start_datetime + event_duration
            plot_end: str = str((event_end_datetime + pd.Timedelta(end_buffer)).date())

            plot_event(
                outage_threshold,
                event_start_datetime,
                event_end_datetime,
                outage_attr.CountyFIPS,
                event_duration,
                outage_attr.integral,
                outage_attr.pop_hours_supply_lost,
                1 - hourly.loc[(slice(plot_start, plot_end), outage_attr.CountyFIPS), "OutageFraction"].droplevel(1),
                1 - resampled.loc[(slice(plot_start, plot_end), outage_attr.CountyFIPS), "OutageFraction"].droplevel(1),
                counties,
                states,
                output.plots
            )


rule cluster_events:
    """
    Use pairwise distance in days between events to cluster. Then cluster over
    space for each temporal cluster.
    """
    input:
        events = rules.identify_events.output.events
    output:
        clusters = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/{TIME_DAYS}/{SPACE_DEG}/clusters.pq"
    run:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from sklearn.cluster import DBSCAN

        min_integral_norm: float = 0.05

        # create matrix of pairwise distance in time between events
        events = pd.read_parquet(input.events)
        print(f"{len(events)=}")

        # filter events
        events["integral_norm"] = events.integral / events.duration_hours
        print(f"Filtering to events with {min_integral_norm=} or greater")
        events = events[events.integral_norm >= min_integral_norm]
        print(f"{len(events)=}")

        arr = events.days_since_data_start.values
        distance = np.abs(arr - arr[:, None])

        # cluster in time
        dbscan = DBSCAN(
            eps=float(wildcards.TIME_DAYS),
            min_samples=1,
            metric="precomputed",
        )
        dbscan.fit(distance)
        events["time_cluster_id"] = pd.Series(dbscan.labels_)
        print(events.time_cluster_id.value_counts())

        # cluster in space
        def geo_cluster(lat: np.ndarray, long: np.ndarray, epsilon_deg, min_samples=1):
            """
            Find the spatial clusters of events.
            """
            dbscan = DBSCAN(
                eps=np.deg2rad(epsilon_deg),
                min_samples=min_samples,
                metric='haversine'
            )
            lat_lng_pts = [x for x in zip(lat, long)]
            dbscan.fit(np.radians(lat_lng_pts))
            return pd.Series(dbscan.labels_)

        for time_cluster_id in events.time_cluster_id.unique():
            time_cluster_mask = events.time_cluster_id == time_cluster_id
            events.loc[time_cluster_mask, "geo_cluster_id"] = geo_cluster(
                events.latitude.values,
                events.longitude.values,
                float(wildcards.SPACE_DEG),  # epsilon degrees
            )

        events = events[~events.time_cluster_id.isna()]
        events.time_cluster_id = events.time_cluster_id.astype(int)
        events = events[~events.geo_cluster_id.isna()]
        events.geo_cluster_id = events.geo_cluster_id.astype(int)

        # generate a unique spatio-temporal cluster id
        # don't save this with events, we can't easily deserialise it later (pyarrow casts to np.array)
        cluster_id: pd.Series = events.apply(
            lambda row: tuple([row.time_cluster_id, row.geo_cluster_id]),
            axis=1
        )

        print(events)
        events_per_cluster = cluster_id.value_counts()
        print(events_per_cluster[events_per_cluster > 1])
        events.to_parquet(output.clusters, index=False)


rule plot_clusters:
    """
    Take clusters and plot a timeseries (from the original data) with an inset
    map of the affected counties.
    """
    input:
        clustered_events = rules.cluster_events.output.clusters,
        hourly = "data/output/outage/1H/timeseries.pq",
        counties = "data/input/counties/geometry/cb_2018_us_county_500k.shp",
        states = "data/input/states/state_codes.csv",
        countries = "data/input/countries/ne_110m_admin_0_countries.shp",
    output:
        plots = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/{TIME_DAYS}/{SPACE_DEG}/plots")
    run:
        import geopandas as gpd
        import pandas as pd

        from pous.plot import plot_event_cluster

        min_counties = 5
        min_affected_person_hours = 1E5

        print("Reading input data...")
        events = pd.read_parquet(input.clustered_events)
        events["cluster_id"] = events.apply(
            lambda row: tuple([row.time_cluster_id, int(row.geo_cluster_id)]),
            axis=1
        )
        counties = gpd.read_file(input.counties)
        countries = gpd.read_file(input.countries)
        states = pd.read_csv(input.states)
        hourly = pd.read_parquet(input.hourly)

        os.makedirs(output.plots, exist_ok=True)
        usa = countries[countries.ISO_A3 == "USA"]

        print("Plotting...")
        for cluster_id in events.cluster_id.unique():

            cluster = events[events.cluster_id == cluster_id]
            county_codes = cluster.CountyFIPS.sort_values()

            if len(county_codes.unique()) < min_counties:
                # do not plot very small clusters
                continue

            if cluster.pop_hours_supply_lost.sum() < min_affected_person_hours:
                # do not plot clusters with very few affected customers
                continue

            county_hourly = hourly.loc[(slice(None), county_codes), :]
            plot_event_cluster(
                cluster_id,
                cluster,
                county_hourly,
                float(wildcards.THRESHOLD),
                usa,
                wildcards.RESAMPLE_FREQ,
                counties,
                states,
                output.plots,
            )
