import geopandas as gpd
import pandas as pd


def us_county_name(county_code: str, county_boundaries: gpd.GeoDataFrame, states: pd.DataFrame) -> tuple[str, str]:
    """
    Lookup county name, containing state name and code, given a 5-digit string county code.

    Args:
        county_code: Unique county FIPS code, e.g. "12093"
        county_boundaries: Table of counties with `GEOID` column containing
            county FIPS codes, `NAME` column and `STATEFP` column.
        states: Table of states with `state_fips_code`, `state_name` and
            `state_alpha_2_code`.

    Returns:
        County name, e.g. Okeechobee
        State name, e.g. Florida
        State 2 letter alphabetic code, e.g. FL
    """

    # county lookup
    county_admin_data: pd.Series = \
        county_boundaries.sort_values("GEOID").set_index("GEOID").loc[county_code, :]
    county_name: str = county_admin_data.NAME

    # state lookup
    states: pd.DataFrame = states.set_index(states["state_fips_code"])
    state_code: str = county_admin_data.STATEFP
    try:
        state_name: str = states.loc[int(state_code), "state_name"]
        state_alpha_code: str = states.loc[int(state_code), "state_alpha_2_code"]
    except KeyError:
        # STATEFP code is not in states data (Puerto Rico, among others)
        state_name = "-"
        state_alpha_code = "-"

    return county_name, state_name, state_alpha_code
