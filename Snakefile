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
        plot = "data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/events.png",
    run:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        plt.style.use('dark_background')  # for cool points

        county_boundaries = gpd.read_file(input.counties)

        # take the resampled data and filter to periods with OutageFraction above a threshold
        resampled = pd.read_parquet(input.resampled, columns=["OutageFraction"])
        resampled_outages = resampled[resampled.OutageFraction > float(wildcards.THRESHOLD)]

        data_start: pd.Timestamp = resampled.index.get_level_values(level="RecordDateTime").min()

        resample_period_ns = pd.Timedelta(wildcards.RESAMPLE_FREQ).total_seconds() * 1E9
        day_in_ns = 1E9 * 60 * 60 * 24

        events = []
        for county_code in tqdm(resampled_outages.index.get_level_values("CountyFIPS").unique()):

            # lookup county
            county_centroid = county_boundaries.set_index("GEOID").loc[county_code].geometry.centroid

            county_resampled: pd.DataFrame = resampled_outages.loc[(slice(None), county_code), :]
            county_resampled = county_resampled.reset_index(level="CountyFIPS")

            # picking out runs of resampled outage periods
            run_start_index = 0
            outage_period_resampled_indicies: list[tuple[int, int]] = []
            for i, time_gap_ns in enumerate(np.diff(county_resampled.index.values)):
                if np.abs((float(time_gap_ns) / resample_period_ns) - 1) < 0.1:
                    continue
                else:
                    outage_period_resampled_indicies.append((run_start_index, i))
                    run_start_index = i + 1
            else:
                # store the last run which we'd otherwise miss
                if np.abs((float(time_gap_ns) / resample_period_ns) - 1) < 0.1:
                    outage_period_resampled_indicies.append((run_start_index, i))

            for period_indicies in outage_period_resampled_indicies:

                start_i, end_i = period_indicies
                n_periods: int = end_i - start_i
                if n_periods > 1:
                    # retrieve indicies of resampled run of outage periods
                    event_start, *_ = county_resampled.iloc[start_i: end_i + 1].index
                    events.append(
                        (
                            county_code,
                            county_centroid.x,
                            county_centroid.y,
                            event_start.date(),
                            (event_start - data_start).value / day_in_ns,
                            n_periods,
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
                "n_periods",
            ]
        )
        events = events.sort_values("days_since_data_start").reset_index(drop=True)
        print(events)
        events.to_parquet(output.events)

        countries = gpd.read_file(input.countries)
        usa = countries[countries.ISO_A3 == "USA"]
        events["geometry"] = gpd.points_from_xy(events.longitude, events.latitude)
        events = gpd.GeoDataFrame(events)

        f, ax = plt.subplots(figsize=(12,8))
        event_count = events.loc[:, ["CountyFIPS", "n_periods"]].groupby("CountyFIPS").sum()
        counties = county_boundaries.loc[:, ["GEOID", "geometry"]].set_index("GEOID")
        outage_events_per_county = event_count.merge(counties, left_on="CountyFIPS", right_on="GEOID")
        outage_events_per_county = gpd.GeoDataFrame(outage_events_per_county)
        outage_events_per_county.plot(column="n_periods", ax=ax, cmap=matplotlib.colormaps["spring"])
        usa.boundary.plot(ax=ax, alpha=0.5)
        ax.set_title(
            f"Resample period: {wildcards.RESAMPLE_FREQ}, threshold: {wildcards.THRESHOLD}\n"
            f"Number of discrete county outage events: {len(events)}"
        )
        ax.grid(alpha=0.2)
        ax.set_xlim(-130, -65)
        ax.set_ylim(22, 53)
        ax.set_ylabel("Latitude [deg]")
        ax.set_xlabel("Longitude [deg]")
        f.savefig(output.plot)


rule plot_events:
    """
    Plot county-event timeseries (no event clustering). Overlay inferred durations.
    """
    input:
        events = rules.identify_events.output.events,
        hourly = "data/output/outage/1H/timeseries.pq",
        counties = "data/input/counties/cb_2018_us_county_500k.shp",
        states = "data/input/states/state_codes.csv",
    output:
        plots = directory("data/output/outage/{RESAMPLE_FREQ}/{THRESHOLD}/event_plots")
    run:
        import os
        import multiprocessing
        multiprocessing.set_start_method("spawn")

        import geopandas as gpd
        import pandas as pd

        from pous.plot import plot_event


        outage_threshold = float(wildcards.THRESHOLD)
        counties = gpd.read_file(input.counties)
        states = pd.read_csv(input.states)
        events = pd.read_parquet(input.events)
        hourly = pd.read_parquet(input.hourly)

        os.makedirs(output.plots, exist_ok=True)

        max_plot_length = "60D"
        start_buffer = "2D"
        end_buffer = "5D"

        tasks = []
        with multiprocessing.Pool(processes=workflow.cores) as pool:
            for outage_attr in events.itertuples():

                event_duration = pd.Timedelta(wildcards.RESAMPLE_FREQ) * outage_attr.n_periods
                if event_duration > pd.Timedelta(max_plot_length):
                    print(f"{event_duration=} > {max_plot_length=} for {cluster_id=}, skipping")
                    continue

                event_start_datetime = pd.to_datetime(outage_attr.event_start)
                plot_start: str = str((event_start_datetime - pd.Timedelta(start_buffer)).date())
                event_end_datetime = event_start_datetime + event_duration
                plot_end: str = str((event_end_datetime + pd.Timedelta(end_buffer)).date())

                print(outage_attr.event_start, outage_attr.CountyFIPS)
                task = pool.apply_async(
                    plot_event,
                    (
                        outage_threshold,
                        event_start_datetime,
                        event_end_datetime,
                        outage_attr.CountyFIPS,
                        event_duration,
                        hourly.loc[(slice(plot_start, plot_end), outage_attr.CountyFIPS), :].copy(deep=True),
                        counties,
                        states,
                        output.plots
                    )
                )
                tasks.append(task)

            [task.get() for task in tasks]



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

        print(events)
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
        import multiprocessing
        multiprocessing.set_start_method("spawn")

        import geopandas as gpd
        import pandas as pd

        from pous.plot import plot_event_cluster

        print("Reading input data...")
        events = pd.read_parquet(input.clustered_events)
        # generate a unique spatio-temporal cluster id
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
        task_results = []
        with multiprocessing.Pool(processes=workflow.cores) as pool:
            for cluster_id in events.cluster_id.unique():
                county_codes = events[events.cluster_id == cluster_id].CountyFIPS.sort_values()
                county_hourly = hourly.loc[(slice(None), county_codes), :].copy(deep=True)
                task_result = pool.apply_async(
                    plot_event_cluster,
                    (
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
                )
                task_results.append(task_result)

            [task.get() for task in task_results]
