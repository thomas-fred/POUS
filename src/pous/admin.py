import geopandas as gpd
import pandas as pd


def us_county_name(county_code: str, county_boundaries: gpd.GeoDataFrame, states: pd.DataFrame) -> tuple[str, str]:
    """
    Lookup county name, containing state name and code, given a 5-digit string county code.
    """

    try:
        county_admin_data: pd.Series = \
            county_boundaries.sort_values("GEOID").set_index("GEOID").loc[county_code, :]
        state_code: str = county_admin_data.STATEFP
        state_name: str = states.loc[states["state_fips_code"] == int(state_code), "state_name"]
        county_name: str = county_admin_data.NAME
    except Exception as e:
        state_name = "-"
        county_name = "-"

    return county_name, state_name, state_code
