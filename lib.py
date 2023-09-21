import geopandas as gpd
import pandas as pd


def merge_in_geometry(county_hour: pd.DataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Combine the outage timeseries data with county border polygons.

    Args:
        county_hour: Timeseries data containing `CountyFIPS` column to join on, along with
            `RecordDateTime`, `CustomersOut` and `CustomersTracked`.
        counties: Geographic data containing `STATEFP`, `geometry` and `GEOID`.
    """

    # merge in county boundary polygons
    long_format = county_hour.reset_index()
    outage_geography = (
        long_format.merge(
            counties[["STATEFP", "GEOID", "geometry"]],
            left_on="CountyFIPS",
            right_on="GEOID")
        .drop(columns=["GEOID"])
    )
    # recreate our DateTimeIndex
    outage = (
        outage_geography.set_index(
            pd.to_datetime(outage_geography.RecordDateTime)
        ).drop(columns=["RecordDateTime"])
    )
    outage = gpd.GeoDataFrame(outage)

    return outage
