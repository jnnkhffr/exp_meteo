"""
Import and combine SLS20 scintillometer data.

The importer reads SLS20 RES and DGN files.

RES files
---------
RES files contain time-resolved measurements in the form

    Run Date Time value1 value2 ...

The observed SLS20 RES format contains 23 columns:

    Run
    Date
    Time
    res_01 ... res_20

The device timestamps are preserved with a fixed UTC+01:00
timezone.

RES measurements are expected at 30-second intervals.

Missing RES measurement periods are inserted with NaN values.

DGN files
---------
DGN files contain diagnostic measurements in the form

    Run Sub value1 value2 ...

The observed SLS20 DGN format contains 23 columns:

    Run
    Sub
    dgn_01 ... dgn_20

DGN files do not contain timestamps. Therefore no artificial
timestamp is assigned to DGN measurements.

Multiple DGN files are combined and duplicate Run/Sub
measurements are handled.

Only RES measurements from 2026-08-10 00:00:00+01:00 onward
are used.

The importer does not assign physical names to the measurement
columns because the exact SLS20 column definitions have not yet
been established from the instrument documentation.
"""


from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FOLDER = Path(
    r"X:\SLS20"
)


START_DATE = datetime(
    2026,
    8,
    10,
    0,
    0,
    0,
    tzinfo=timezone(timedelta(hours=1)),
)


DEVICE_TIMEZONE = timezone(
    timedelta(hours=1)
)


RES_INTERVAL = pd.Timedelta(
    seconds=30
)


SUPPORTED_SUFFIXES = {
    ".res",
    ".dgn",
}


RES_COLUMNS = [
    "Run",
    "Date",
    "Time",
    "pressure",
    "temperature",
    "temperature_difference",
    "path_length",
    "height",
    "siglogX",
    "siglogY",
    "logCor",
    "Nok",
    "C2_n",
    "l_0",
    "C2_T",
    "DIss",
    "Heat_day",
    "Heat_night",
    "Mom_day",
    "Mom_night",
    "MO-L_day",
    "MO-L_night",
    "Wind",
]

DGN_COLUMNS = [
    "Run",
    "Sub",
] + [
    f"dgn_{i:02d}"
    for i in range(1, 21)
]


# ============================================================================
# GENERAL FILE HELPERS
# ============================================================================

def file_sort_key(
    path: Path,
) -> tuple[str, str]:
    """
    Create a deterministic sorting key for SLS20 input files.
    """

    return (
        path.stem.lower(),
        path.suffix.lower(),
    )


def read_text_file(
    path: Path,
) -> list[str]:
    """
    Read an SLS20 file as text.

    Invalid characters are ignored because the SLS20 files
    are plain numerical/text files.
    """

    with open(
        path,
        "r",
        errors="ignore",
    ) as file:

        return file.readlines()


# ============================================================================
# VALUE CONVERSION
# ============================================================================

def convert_value(
    value: str,
):
    """
    Convert a single SLS20 value.

    Numeric values become floats.
    Missing values become NaN.
    Non-numeric values remain strings.
    """

    value = value.strip()

    if value.upper() in {
        "",
        "N/A",
        "NA",
        "NAN",
    }:
        return np.nan

    try:
        return float(value)

    except ValueError:
        return value


def convert_values(
    values: list[str],
) -> list:
    """
    Convert a list of SLS20 values.
    """

    return [
        convert_value(value)
        for value in values
    ]


# ============================================================================
# RES PARSING
# ============================================================================

def is_res_data_line(
    line: str,
) -> bool:
    """
    Determine whether a line is an SLS20 RES measurement line.

    Expected structure:

        Run Date Time value1 value2 ...
    """

    parts = line.split()

    if len(parts) < 4:
        return False

    # Run number
    try:
        int(parts[0])
    except ValueError:
        return False

    # Date
    if len(parts[1]) != 10:
        return False

    if parts[1][4] != "-":
        return False

    # Time
    if len(parts[2]) != 8:
        return False

    if parts[2][2] != ":":
        return False

    return True


def parse_res_line(
    line: str,
) -> dict | None:
    """
    Parse one SLS20 RES measurement line.

    Returns
    -------
    dict or None
        Parsed measurement.
    """

    line = line.strip()

    if not line:
        return None

    if not is_res_data_line(line):
        return None

    parts = line.split()

    run = int(parts[0])
    date_string = parts[1]
    time_string = parts[2]

    timestamp_string = (
        f"{date_string} {time_string}"
    )

    try:

        timestamp = pd.Timestamp(
            datetime.strptime(
                timestamp_string,
                "%Y-%m-%d %H:%M:%S",
            ).replace(
                tzinfo=DEVICE_TIMEZONE
            )
        )

    except ValueError:

        return None

    measurement_values = parts[3:]

    expected_measurements = 20

    if len(measurement_values) < expected_measurements:

        measurement_values = (
            measurement_values
            + ["N/A"]
            * (
                expected_measurements
                - len(measurement_values)
            )
        )

    elif len(measurement_values) > expected_measurements:

        measurement_values = (
            measurement_values[
                :expected_measurements
            ]
        )

    converted = convert_values(
        measurement_values
    )

    row = {
        "timestamp": timestamp,
        "Run": run,
        "Date": date_string,
        "Time": time_string,
    }

    row.update(
        dict(
            zip(
                RES_COLUMNS[3:],
                converted,
            )
        )
    )

    return row


def read_res_file(
    path: Path,
) -> pd.DataFrame:
    """
    Read one SLS20 RES file.
    """

    lines = read_text_file(path)

    rows = []

    for line in lines:

        row = parse_res_line(line)

        if row is not None:
            rows.append(row)

    if not rows:

        return pd.DataFrame(
            columns=[
                "timestamp",
                *RES_COLUMNS,
            ]
        )

    df = pd.DataFrame(
        rows
    )

    return df


# ============================================================================
# DGN PARSING
# ============================================================================

def is_dgn_data_line(
    line: str,
) -> bool:
    """
    Determine whether a line is an SLS20 DGN measurement line.

    Expected structure:

        Run Sub value1 value2 ...
    """

    parts = line.split()

    if len(parts) < 3:
        return False

    try:
        int(parts[0])
        int(parts[1])
        float(parts[2])

    except ValueError:
        return False

    return True


def parse_dgn_line(
    line: str,
) -> dict | None:
    """
    Parse one SLS20 DGN line.
    """

    line = line.strip()

    if not line:
        return None

    if not is_dgn_data_line(line):
        return None

    parts = line.split()

    run = int(parts[0])
    sub = int(parts[1])

    measurement_values = parts[2:]

    expected_measurements = 21

    if len(measurement_values) < expected_measurements:

        measurement_values = (
            measurement_values
            + ["N/A"]
            * (
                expected_measurements
                - len(measurement_values)
            )
        )

    elif len(measurement_values) > expected_measurements:

        measurement_values = (
            measurement_values[
                :expected_measurements
            ]
        )

    converted = convert_values(
        measurement_values
    )

    row = {
        "Run": run,
        "Sub": sub,
    }

    row.update(
        dict(
            zip(
                DGN_COLUMNS[2:],
                converted,
            )
        )
    )

    return row


def read_dgn_file(
    path: Path,
) -> pd.DataFrame:
    """
    Read one SLS20 DGN file.
    """

    lines = read_text_file(path)

    rows = []

    for line in lines:

        row = parse_dgn_line(line)

        if row is not None:
            rows.append(row)

    if not rows:

        return pd.DataFrame(
            columns=DGN_COLUMNS
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# DATE FILTERING
# ============================================================================

def filter_start_date(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Keep only RES measurements from start_date onward.
    """

    if df.empty:
        return df

    mask = (
        df["timestamp"]
        >= start_date
    )

    return df.loc[
        mask
    ].copy()


# ============================================================================
# RES DUPLICATE HANDLING
# ============================================================================

def combine_duplicate_rows(
    group: pd.DataFrame,
) -> pd.Series:
    """
    Combine multiple RES measurements with the same timestamp.

    The first available non-missing value is retained.
    """

    if len(group) == 1:
        return group.iloc[0]

    result = {}

    for column in group.columns:

        values = group[column]

        non_missing = (
            values.dropna()
        )

        if non_missing.empty:

            result[column] = np.nan

        else:

            result[column] = (
                non_missing.iloc[0]
            )

    return pd.Series(
        result
    )


def remove_res_duplicate_timestamps(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate RES timestamps.
    """

    if df.empty:
        return df

    duplicate_mask = (
        df["timestamp"]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    if duplicate_rows == 0:

        return df

    duplicate_timestamps = (
        df.loc[
            duplicate_mask,
            "timestamp",
        ]
        .drop_duplicates()
        .sort_values()
    )

    print(
        f"RES: {duplicate_rows} rows belong to "
        f"{len(duplicate_timestamps)} duplicate timestamps."
    )

    combined_rows = []

    for timestamp, group in df.groupby(
        "timestamp",
        sort=True,
        dropna=False,
    ):

        combined_rows.append(
            combine_duplicate_rows(
                group
            )
        )

    result = pd.DataFrame(
        combined_rows
    )

    return result.reset_index(
        drop=True
    )


# ============================================================================
# DGN DUPLICATE HANDLING
# ============================================================================

def remove_dgn_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate DGN Run/Sub combinations.

    If duplicate rows contain different values, the first
    non-missing value for each column is retained.
    """

    if df.empty:
        return df

    duplicate_mask = (
        df.duplicated(
            subset=[
                "Run",
                "Sub",
            ],
            keep=False,
        )
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    if duplicate_rows == 0:
        return df

    duplicate_keys = (
        df.loc[
            duplicate_mask,
            [
                "Run",
                "Sub",
            ],
        ]
        .drop_duplicates()
    )

    print(
        f"DGN: {duplicate_rows} rows belong to "
        f"{len(duplicate_keys)} duplicate Run/Sub combinations."
    )

    combined_rows = []

    for (
        run,
        sub,
    ), group in df.groupby(
        [
            "Run",
            "Sub",
        ],
        sort=True,
        dropna=False,
    ):

        if len(group) == 1:

            combined_rows.append(
                group.iloc[0]
            )

            continue

        result = {}

        for column in group.columns:

            values = group[column]

            non_missing = (
                values.dropna()
            )

            if non_missing.empty:

                result[column] = np.nan

            else:

                result[column] = (
                    non_missing.iloc[0]
                )

        combined_rows.append(
            pd.Series(result)
        )

    return pd.DataFrame(
        combined_rows
    ).reset_index(
        drop=True
    )


# ============================================================================
# RES TIME AXIS
# ============================================================================

def create_complete_res_time_axis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a continuous 30-second RES time axis.

    The axis starts at START_DATE and ends at the last
    available measurement timestamp.
    """

    if df.empty:
        return df

    df = (
        df.sort_values(
            "timestamp"
        )
        .copy()
    )

    df = df.drop_duplicates(
        subset="timestamp",
        keep="first",
    )

    last_timestamp = (
        df["timestamp"].max()
    )

    full_index = pd.date_range(
        start=START_DATE,
        end=last_timestamp,
        freq=RES_INTERVAL,
        tz=DEVICE_TIMEZONE,
    )

    original_length = len(df)

    df = df.set_index(
        "timestamp"
    )

    result = df.reindex(
        full_index
    )

    result.index.name = (
        "timestamp"
    )

    result = result.reset_index()

    missing_count = (
        len(result)
        - original_length
    )

    if missing_count > 0:

        print(
            f"RES: Added {missing_count} "
            f"missing 30-second time step(s) as NaN."
        )

    # Reconstruct Date and Time for inserted rows
    result["Date"] = (
        result["timestamp"]
        .dt.strftime("%Y-%m-%d")
    )

    result["Time"] = (
        result["timestamp"]
        .dt.strftime("%H:%M:%S")
    )

    return result


# ============================================================================
# INTERVAL DETECTION
# ============================================================================

def detect_res_measurement_interval(
    df: pd.DataFrame,
) -> pd.Timedelta | None:
    """
    Detect the most common interval between RES measurements.
    """

    if df.empty or len(df) < 2:
        return None

    timestamps = (
        df["timestamp"]
        .sort_values()
        .drop_duplicates()
    )

    differences = (
        timestamps.diff()
        .dropna()
    )

    if differences.empty:
        return None

    counts = (
        differences.value_counts()
    )

    print()
    print(
        "RES time difference statistics:"
    )

    print(
        counts.head(10)
    )

    interval = counts.index[0]

    print(
        f"Detected typical RES measurement interval: "
        f"{interval} "
        f"({counts.iloc[0]} occurrences)"
    )

    return interval


# ============================================================================
# COMBINE RES
# ============================================================================

def combine_res_dataframes(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine multiple parsed RES DataFrames.
    """

    if not dataframes:

        print()
        print(
            "No RES data available."
        )

        return pd.DataFrame()

    combined = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined.sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    combined = (
        remove_res_duplicate_timestamps(
            combined
        )
    )

    detected_interval = (
        detect_res_measurement_interval(
            combined
        )
    )

    if detected_interval is not None:

        if abs(
            detected_interval.total_seconds()
            - RES_INTERVAL.total_seconds()
        ) > 0.1:

            print(
                "WARNING: Detected RES interval "
                f"{detected_interval} differs from "
                f"expected interval {RES_INTERVAL}."
            )

    combined = (
        create_complete_res_time_axis(
            combined
        )
    )

    return combined


# ============================================================================
# COMBINE DGN
# ============================================================================

def combine_dgn_dataframes(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine multiple parsed DGN DataFrames.

    DGN data are sorted by Run and Sub.
    """

    if not dataframes:

        print()
        print(
            "No DGN data available."
        )

        return pd.DataFrame()

    combined = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined.sort_values(
            [
                "Run",
                "Sub",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    combined = (
        remove_dgn_duplicates(
            combined
        )
    )

    return combined


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def print_res_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a summary of the combined RES data.
    """

    print()
    print("=" * 80)
    print("RES DATA SUMMARY")
    print("=" * 80)

    if df.empty:

        print(
            "No combined RES data available."
        )

        return

    print()
    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"From: {df['timestamp'].min()}"
    )

    print(
        f"To:   {df['timestamp'].max()}"
    )

    print()
    print(
        "First rows:"
    )

    print(
        df.head()
    )

    print()
    print(
        "Last rows:"
    )

    print(
        df.tail()
    )

    measurement_columns = [
        column
        for column in df.columns
        if column.startswith("res_")
    ]

    if measurement_columns:

        all_nan_mask = (
            df[
                measurement_columns
            ]
            .isna()
            .all(axis=1)
        )

        print()
        print(
            "RES rows containing only "
            f"NaN measurements: "
            f"{int(all_nan_mask.sum())}"
        )


def print_dgn_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a summary of the combined DGN data.
    """

    print()
    print("=" * 80)
    print("DGN DATA SUMMARY")
    print("=" * 80)

    if df.empty:

        print(
            "No combined DGN data available."
        )

        return

    print()
    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"First Run: {df['Run'].min()}"
    )

    print(
        f"Last Run:  {df['Run'].max()}"
    )

    print(
        f"Unique Runs: "
        f"{df['Run'].nunique()}"
    )

    print(
        f"Unique Sub values: "
        f"{sorted(df['Sub'].dropna().unique())}"
    )

    print()
    print(
        "First rows:"
    )

    print(
        df.head(10)
    )

    print()
    print(
        "Last rows:"
    )

    print(
        df.tail(10)
    )


# ============================================================================
# FILE REPORT
# ============================================================================

def print_loaded_files(
    res_files: list[Path],
    dgn_files: list[Path],
) -> None:
    """
    Print the files contributing to the datasets.
    """

    print()
    print("=" * 80)
    print("LOADED FILES")
    print("=" * 80)

    print()
    print("RES files:")

    for file in res_files:

        print(
            f"  {file.name}"
        )

    if not res_files:

        print(
            "  None"
        )

    print()
    print("DGN files:")

    for file in dgn_files:

        print(
            f"  {file.name}"
        )

    if not dgn_files:

        print(
            "  None"
        )


# ============================================================================
# MAIN LOADER
# ============================================================================

def load_sls20_data(
    folder: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load all SLS20 RES and DGN files from a directory.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]

        res
            Continuous time-resolved RES dataset.

        dgn
            Combined DGN diagnostic dataset.
    """

    if not folder.exists():

        raise FileNotFoundError(
            f"Data folder does not exist: "
            f"{folder}"
        )

    if not folder.is_dir():

        raise NotADirectoryError(
            f"Not a directory: "
            f"{folder}"
        )

    print()
    print("=" * 80)
    print("SLS20 DATA IMPORT")
    print("=" * 80)

    print()
    print(
        f"Data folder: {folder}"
    )

    print(
        f"Using RES data from: "
        f"{START_DATE}"
    )

    print(
        "Device timestamp handling: "
        "fixed UTC+01:00"
    )

    res_dataframes = []
    dgn_dataframes = []

    valid_res_files = []
    valid_dgn_files = []

    files = sorted(
        folder.iterdir(),
        key=file_sort_key,
    )

    for file in files:

        # ------------------------------------------------------------
        # Hidden/system files
        # ------------------------------------------------------------

        if file.name.startswith("."):

            print(
                f"Skipping hidden file: "
                f"{file.name}"
            )

            continue

        # ------------------------------------------------------------
        # Other file types
        # ------------------------------------------------------------

        if (
            not file.is_file()
            or file.suffix.lower()
            not in SUPPORTED_SUFFIXES
        ):

            print(
                f"Skipping non-data file: "
                f"{file.name}"
            )

            continue

        suffix = (
            file.suffix.lower()
        )

        print()
        print(
            f"Reading {file.name}"
        )

        # ------------------------------------------------------------
        # RES
        # ------------------------------------------------------------

        if suffix == ".res":

            try:

                df = read_res_file(
                    file
                )

                if df.empty:

                    print(
                        "  No RES measurement "
                        "rows found."
                    )

                    continue

                df = filter_start_date(
                    df,
                    pd.Timestamp(
                        START_DATE
                    ),
                )

                if df.empty:

                    print(
                        "  0 valid rows after "
                        "date filtering."
                    )

                    continue

                print(
                    f"  -> {len(df)} valid "
                    "RES rows after date filtering"
                )

                res_dataframes.append(
                    df
                )

                valid_res_files.append(
                    file
                )

            except Exception as exc:

                print(
                    f"  ERROR while reading "
                    f"{file.name}: {exc}"
                )

        # ------------------------------------------------------------
        # DGN
        # ------------------------------------------------------------

        elif suffix == ".dgn":

            try:

                df = read_dgn_file(
                    file
                )

                if df.empty:

                    print(
                        "  No DGN measurement "
                        "rows found."
                    )

                    continue

                print(
                    f"  -> {len(df)} valid "
                    "DGN rows"
                )

                dgn_dataframes.append(
                    df
                )

                valid_dgn_files.append(
                    file
                )

            except Exception as exc:

                print(
                    f"  ERROR while reading "
                    f"{file.name}: {exc}"
                )

    # ========================================================================
    # COMBINE DATA
    # ========================================================================

    res = combine_res_dataframes(
        res_dataframes
    )

    dgn = combine_dgn_dataframes(
        dgn_dataframes
    )

    # ========================================================================
    # REPORT
    # ========================================================================

    print_loaded_files(
        valid_res_files,
        valid_dgn_files,
    )

    return res, dgn


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    res, dgn = load_sls20_data(
        DATA_FOLDER
    )

    print_res_summary(
        res
    )

    print_dgn_summary(
        dgn
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)