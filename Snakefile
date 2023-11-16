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

        hourly_with_gaps = pd.concat([pd.read_parquet(path, columns=["OutageFraction"]) for path in input.years])
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


rule identify_events:
    """
    Look for events where OutageFraction exceeded threshold in resampled data.
    """
    input:
        resampled = rules.resample_outages.output.resampled,
        counties = "data/input/counties/cb_2018_us_county_500k.shp",
        countries = "data/input/countries/ne_110m_admin_0_countries.shp",
    output:
        events = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/events.pq",
        frequency_map = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_frequency_map.png",
        duration_histogram = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_duration_histogram.png",
        duration_magnitude_scatter = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_duration_magnitude_scatter.png",
    run:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        from pous.plot import plot_events_summary

        counties = gpd.read_file(input.counties)
        countries = gpd.read_file(input.countries)

        # take the resampled data and filter to periods with OutageFraction above a threshold
        resampled = pd.read_parquet(input.resampled, columns=["OutageFraction"])
        resampled_outages = resampled[resampled.OutageFraction > float(wildcards.THRESHOLD)]

        data_start: pd.Timestamp = resampled.index.get_level_values(level="RecordDateTime").min()

        resample_period: pd.Timedelta = pd.Timedelta(wildcards.RESAMPLE_FREQ)
        resample_period_ns: float = resample_period.total_seconds() * 1E9
        day_in_ns: float = 1E9 * 60 * 60 * 24

        events = []
        for county_code in tqdm(resampled_outages.index.get_level_values("CountyFIPS").unique()):

            # lookup county
            county_centroid = counties.set_index("GEOID").loc[county_code].geometry.centroid

            county_resampled: pd.DataFrame = resampled_outages.loc[(slice(None), county_code), :]
            county_resampled = county_resampled.reset_index(level="CountyFIPS")

            # can't take a diff along one row
            if len(county_resampled) < 2:
                continue

            # picking out runs of resampled outage periods
            max_relative_timestep_error = 0.1
            run_start_index = 0
            outages_start_end: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for i, time_gap_ns in enumerate(np.diff(county_resampled.index.values)):

                if np.abs((float(time_gap_ns) / resample_period_ns) - 1) < max_relative_timestep_error:
                    continue
                else:
                    # next timestep in thresholded outage data is not equal to resample period,
                    # this is the end of an outage, record start and end datetimes
                    run_start: pd.Timestamp = county_resampled.index[run_start_index]
                    n_periods: int = i - run_start_index + 1
                    run_end: pd.Timestamp = run_start + resample_period * n_periods
                    outages_start_end.append((run_start, run_end))

                    # reset start index (moving to next outage event)
                    run_start_index = i + 1
            else:
                # store the last run which we'd otherwise miss
                if np.abs((float(time_gap_ns) / resample_period_ns) - 1) < max_relative_timestep_error:
                    run_start: pd.Timestamp = county_resampled.index[run_start_index]
                    n_periods: int = i - run_start_index + 1
                    run_end: pd.Timestamp = run_start + resample_period * n_periods
                    outages_start_end.append((run_start, run_end))

            # start of first outage bin (labelled left), end of last outage bin (label right)
            for event_start, event_end in outages_start_end:

                # N.B. resampled bins are time labelled left

                # retrieve indicies of resampled run of outage periods
                outage_data: pd.Series = county_resampled.loc[event_start: event_end]["OutageFraction"]

                duration: pd.Timedelta = event_end - event_start
                duration_hours: float = duration.total_seconds() / (60 * 60)
                outage_magnitude: float = outage_data.sum()

                n_periods = duration / resample_period
                assert int(n_periods) == n_periods

                events.append(
                    (
                        county_code,
                        county_centroid.x,
                        county_centroid.y,
                        event_start,
                        (event_start - data_start).value / day_in_ns,
                        duration_hours,
                        int(n_periods),
                        outage_magnitude,
                    )
                )

        events = pd.DataFrame(
            events,
            columns=[
                "CountyFIPS",
                "longitude",
                "latitude",
                "event_start",
                "days_since_data_start",
                "duration_hours",
                "n_periods",
                "integral",
            ]
        )
        events = events.sort_values("days_since_data_start").reset_index(drop=True)
        print(events)
        events.to_parquet(output.events)

        plot_events_summary(
            wildcards,
            events,
            counties,
            countries[countries.ISO_A3 == "USA"],
            output.frequency_map,
            output.duration_histogram,
            output.duration_magnitude_scatter,
        )



rule plot_events:
    """
    Plot county-event timeseries (no event clustering). Overlay inferred durations.
    """
    input:
        events = rules.identify_events.output.events,
        hourly = "data/output/outage/1H/timeseries.pq",
        resampled = "data/output/outage/{RESAMPLE_FREQ}/timeseries.pq",
        counties = "data/input/counties/cb_2018_us_county_500k.shp",
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

        for outage_attr in events.itertuples():

            if outage_attr.CountyFIPS != "39113":
                continue

            event_duration = pd.Timedelta(wildcards.RESAMPLE_FREQ) * outage_attr.n_periods

            if event_duration > pd.Timedelta(max_event_length):
                print(f"{event_duration=} > {max_event_length=}, skipping")
                continue

            if event_duration < pd.Timedelta(min_event_length):
                print(f"{event_duration=} < {min_event_length=}, skipping")
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

        # create matrix of pairwise distance in time between events
        events = pd.read_parquet(input.events)
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

        events.geo_cluster_id = events.geo_cluster_id.astype(int)

        # generate a unique spatio-temporal cluster id
        # don't save this with events, we can't easily deserialise it later (pyarrow casts to np.array)
        cluster_id: pd.Series = events.apply(
            lambda row: tuple([row.time_cluster_id, int(row.geo_cluster_id)]),
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
        counties = "data/input/counties/cb_2018_us_county_500k.shp",
        states = "data/input/states/state_codes.csv",
        countries = "data/input/countries/ne_110m_admin_0_countries.shp",
    output:
        plots = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/{TIME_DAYS}/{SPACE_DEG}/plots")
    run:
        import geopandas as gpd
        import pandas as pd

        from pous.plot import plot_event_cluster

        min_events = 5

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
            county_codes = events[events.cluster_id == cluster_id].CountyFIPS.sort_values()

            if len(county_codes.unique()) < min_events:
                # do not plot very small clusters
                continue

            county_hourly = hourly.loc[(slice(None), county_codes), :]
            plot_event_cluster(
                cluster_id,
                events,
                county_hourly,
                float(wildcards.THRESHOLD),
                usa,
                wildcards.RESAMPLE_FREQ,
                counties,
                states,
                output.plots,
            )
