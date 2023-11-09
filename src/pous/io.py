import numpy as np
import pandas as pd


def read_POUS_csv(csv_path: str) -> pd.DataFrame:
    """
    Read poweroutage.us CSV data from disk.

    Args:
        csv_path: Path to CSV file
    """

    # read in data (note the unusual encoding)
    # read in county ID codes as strings to prevent dropping of leading 0
    raw = pd.read_csv(
        csv_path,
        encoding="utf-16",
        dtype={
            "CountyFIPS": str,
            "CustomersTracked": np.int32,
            "CustomersOut": np.int32,
        }
    )

    # generate a DateTimeIndex
    clean = raw.set_index(pd.to_datetime(raw.RecordDateTime))
    clean = clean.drop(columns=["RecordDateTime"])

    # drop rows without county codes (empty string in CSV -> NaN in dataframe)
    clean = clean[~clean.CountyFIPS.isna()]

    # discard utility information and groupby the hour and county
    county_hour = clean.groupby([clean.index, clean.CountyFIPS]).sum(numeric_only=True)

    # calculate an outage fraction
    county_hour["OutageFraction"] = np.clip(
        county_hour.CustomersOut / county_hour.CustomersTracked,
        0,
        1
    )

    return county_hour
