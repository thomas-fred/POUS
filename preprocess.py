"""
Preprocess raw Power Outage US CSV data into a more convenient format.

Expect source data to be in raw/outage/ with a naming convention of:
POUS_Export_CountyByUtility_Hourly_202004-202010.csv (for the year 2020).

Processing steps for each yearly source CSV file:
- Combine utilities for each county
- Calculate fraction of customers disconnected
- Save as parquet
"""


from glob import glob
import os
import re

import pandas as pd
import numpy as np


def read_POUS(csv_path: str) -> pd.DataFrame:
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

    return county_hour


if __name__ == "__main__":

    root_dir = "data"

    for csv_path in sorted(glob(os.path.join(root_dir, "raw/outage/POUS_Export_CountyByUtility_Hourly_*.csv"))):

        year_start, year_end = re.match(r".*_(\d{4})04-(\d{4})10.csv$", csv_path).groups()
        year_start == year_end
        assert year_start == year_end
        year = year_start

        processed_path = os.path.join(root_dir, f"processed/outage/{year}.parquet")

        if os.path.exists(processed_path):
            print(f"{processed_path} exists, skipping")

        else:
            print(f"{processed_path} does not exist, creating...")

            # read CSV records of outages
            county_hour = read_POUS(csv_path)

            # calculate an outage fraction
            county_hour["OutageFraction"] = county_hour.CustomersOut / county_hour.CustomersTracked

            county_hour.to_parquet(processed_path)
