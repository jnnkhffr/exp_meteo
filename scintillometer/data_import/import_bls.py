"""
Import and combine BLS scintillometer data.

The importer reads DGN diagnosis files and MND main-data files,
extracts the column definitions from the file headers, preserves
the device timestamps in their original UTC+01:00 time zone,
removes duplicate measurements, and creates a continuous time axis.

Only measurements from 2026-08-10 00:00:00+01:00 onward are used.

DGN data are expected at 30-second intervals.
MND data are expected at 60-second intervals.

Missing measurement periods are inserted with NaN values.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# Configuration

DATA_FOLDER = Path(r"Z:\Daten\Aufzeichnung\sci\output")

START_DATE = datetime(
    2026,
    8,
    10,
    0,
    0,
    0,
    tzinfo=timezone(timedelta(hours=1)),
)

DEVICE_TIMEZONE = timezone(timedelta(hours=1))

DGN_INTERVAL = pd.Timedelta(seconds=30)
MND_INTERVAL = pd.Timedelta(minutes=1)

SUPPORTED_SUFFIXES = {".dgn", ".mnd"}


def file_sort_key(path: Path) -> tuple[str, int, str]:
    """
    Create a deterministic sorting key for BLS input files.

    If an original file and a file named with the suffix
    ``" - Copy"`` both exist, the original is processed first
    and the copy immediately afterward.

    Parameters
    ----------
    path
        Path to the BLS input file.

    Returns
    -------
    tuple[str, int, str]
        Sorting key based on the base filename, copy status,
        and file type.
    """
    stem = path.stem
    copy_suffix = " - Copy"

    if stem.endswith(copy_suffix):
        base_stem = stem[: -len(copy_suffix)]
        copy_priority = 1
    else:
        base_stem = stem
        copy_priority = 0

    return (
        base_stem.lower(),
        copy_priority,
        path.suffix.lower(),
    )


# File parsing

def read_text_file(path: Path) -> list[str]:
    """
    Read a BLS data file as text.

    UTF-8 is tried first. If that fails, cp1252 is used as a fallback.

    Parameters
    ----------
    path
        Path to the input file.

    Returns
    -------
    list[str]
        File contents split into lines.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")

    return text.splitlines()


def extract_header_columns(lines: list[str]) -> list[str]:
    """
    Extract data column names from the BLS header.

    The BLS format describes columns using lines such as:

        Temperature # temp # deg C # S # 0 # N/A

    The second field is the machine-readable column name.

    The first column is the special time field and is converted to
    the column name 'timestamp'.

    Parameters
    ----------
    lines
        Complete file contents.

    Returns
    -------
    list[str]
        Column names defined by the file header.
    """
    data_header_found = False
    columns = []

    for line in lines:
        stripped = line.strip()

        if stripped in {"Main Data", "Diagnosis Data"}:
            data_header_found = True
            continue

        if not data_header_found:
            continue

        if not stripped:
            if columns:
                break
            continue

        if "#" not in line:
            continue

        parts = [part.strip() for part in line.split("#")]

        if len(parts) < 2:
            continue

        name = parts[1]

        if name == "time":
            name = "timestamp"

        if not name:
            continue

        columns.append(name)

    return make_unique_column_names(columns)


def make_unique_column_names(columns: list[str]) -> list[str]:
    """
    Ensure that all column names are unique.

    Parameters
    ----------
    columns
        Original column names.

    Returns
    -------
    list[str]
        Unique column names.
    """
    counts = Counter()
    result = []

    for column in columns:
        counts[column] += 1

        if counts[column] == 1:
            result.append(column)
        else:
            result.append(f"{column}_{counts[column]}")

    return result


def find_data_start(lines: list[str]) -> int:
    """
    Find the first actual measurement line in a BLS file.

    The measurement data start after the blank line following the
    header definition.

    Parameters
    ----------
    lines
        Complete file contents.

    Returns
    -------
    int
        Line index of the first measurement row.

    Raises
    ------
    ValueError
        If no data section can be identified.
    """
    data_header_found = False
    header_started = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped in {"Main Data", "Diagnosis Data"}:
            data_header_found = True
            continue

        if not data_header_found:
            continue

        if "#" in line:
            header_started = True
            continue

        if header_started and not stripped:
            return i + 1

    raise ValueError("Could not determine beginning of measurement data.")


def split_measurement_time(value: str) -> tuple[str, str]:
    """
    Split the BLS combined averaging-period/timestamp field.

    Example
    -------
    PT00H00M59S/2026-08-13T00:01:00+01:00

    becomes

        averaging period = PT00H00M59S
        timestamp        = 2026-08-13T00:01:00+01:00

    Parameters
    ----------
    value
        Combined BLS time field.

    Returns
    -------
    tuple[str, str]
        Averaging period and timestamp.
    """
    value = value.strip()

    if "/" not in value:
        raise ValueError(f"Invalid BLS time field: {value!r}")

    averaging_period, timestamp = value.split("/", 1)

    return averaging_period.strip(), timestamp.strip()


def parse_device_timestamp(value: str) -> pd.Timestamp:
    """
    Parse a timestamp while preserving the device's UTC+01:00 offset.

    No conversion to UTC or local daylight-saving time is performed.

    Parameters
    ----------
    value
        ISO-8601 timestamp from the BLS file.

    Returns
    -------
    pandas.Timestamp
        Timestamp with the original UTC+01:00 offset.
    """
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEVICE_TIMEZONE)

    return pd.Timestamp(dt)


def convert_measurement_values(
    values: list[str],
    columns: list[str],
) -> list:
    """
    Convert BLS measurement values to appropriate Python values.

    'N/A', empty values and similar missing-value representations are
    converted to NaN. Numeric values are converted to floats where
    possible. Non-numeric values are retained as strings.

    Parameters
    ----------
    values
        Raw values from one measurement line.
    columns
        Target column names.

    Returns
    -------
    list
        Converted measurement values.
    """
    converted = []

    for value in values:
        value = value.strip()

        if value.upper() in {"N/A", "NA", "NAN", ""}:
            converted.append(np.nan)
            continue

        try:
            converted.append(float(value))
        except ValueError:
            converted.append(value)

    return converted


def parse_data_line(
    line: str,
    columns: list[str],
) -> dict | None:
    """
    Parse one BLS measurement line.

    Parameters
    ----------
    line
        Raw measurement line.
    columns
        Column names extracted from the header.

    Returns
    -------
    dict or None
        Parsed row, or None if the line is invalid.
    """
    line = line.strip()

    if not line:
        return None

    raw_values = line.split()

    if not raw_values:
        return None

    try:
        averaging_period, timestamp_string = split_measurement_time(
            raw_values[0]
        )

        timestamp = parse_device_timestamp(timestamp_string)

    except (ValueError, TypeError):
        return None

    measurement_values = raw_values[1:]

    expected_measurements = len(columns) - 2

    if len(measurement_values) < expected_measurements:
        measurement_values = (
            measurement_values
            + ["N/A"] * (expected_measurements - len(measurement_values))
        )

    elif len(measurement_values) > expected_measurements:
        measurement_values = measurement_values[:expected_measurements]

    converted = convert_measurement_values(
        measurement_values,
        columns[2:],
    )

    row = {
        "timestamp": timestamp,
        "averagingPeriod": averaging_period,
    }

    row.update(
        dict(zip(columns[2:], converted))
    )

    return row


# Individual file readers

def read_bls_file(path: Path) -> pd.DataFrame:
    """
    Read one DGN or MND file.

    The file structure is determined from its own header. Therefore
    different generations or configurations of the instrument can be
    handled without assuming a fixed number of columns.

    Parameters
    ----------
    path
        Path to the DGN or MND file.

    Returns
    -------
    pandas.DataFrame
        Parsed measurements from the file.
    """
    lines = read_text_file(path)

    columns = extract_header_columns(lines)

    if not columns:
        raise ValueError("No data columns found in header.")

    data_start = find_data_start(lines)

    rows = []

    for line in lines[data_start:]:
        row = parse_data_line(line, columns)

        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)

    return df


def filter_start_date(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Keep only measurements from the configured start date onward.

    Parameters
    ----------
    df
        Input measurements.
    start_date
        Earliest accepted timestamp.

    Returns
    -------
    pandas.DataFrame
        Filtered measurements.
    """
    if df.empty:
        return df

    mask = df["timestamp"] >= start_date

    return df.loc[mask].copy()


# Duplicate handling

def rows_are_equal(
    first: pd.Series,
    second: pd.Series,
) -> bool:
    """
    Check whether two measurements contain the same values.

    NaN values at the same positions are considered equal.

    Parameters
    ----------
    first
        First measurement.
    second
        Second measurement.

    Returns
    -------
    bool
        True if the measurements are equivalent.
    """
    for column in first.index:
        a = first[column]
        b = second[column]

        if pd.isna(a) and pd.isna(b):
            continue

        if a != b:
            return False

    return True


def combine_duplicate_rows(group: pd.DataFrame) -> pd.Series:
    """
    Combine multiple measurements with the same timestamp.

    Identical rows are reduced to one row.

    If duplicate files contain conflicting values, the first available
    non-missing value is retained for each column. This avoids losing
    valid information while keeping exactly one timestamp.

    Parameters
    ----------
    group
        Rows sharing one timestamp.

    Returns
    -------
    pandas.Series
        Combined measurement.
    """
    if len(group) == 1:
        return group.iloc[0]

    result = {}

    for column in group.columns:
        values = group[column]

        non_missing = values.dropna()

        if non_missing.empty:
            result[column] = np.nan
        else:
            result[column] = non_missing.iloc[0]

    return pd.Series(result)


def remove_duplicate_timestamps(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Remove duplicate timestamps from a combined dataset.

    Parameters
    ----------
    df
        Input measurements.
    label
        Dataset label used for console output.

    Returns
    -------
    pandas.DataFrame
        Dataset with one row per timestamp.
    """
    if df.empty:
        return df

    duplicate_mask = df["timestamp"].duplicated(keep=False)

    duplicate_rows = int(duplicate_mask.sum())

    if duplicate_rows == 0:
        return df

    duplicate_timestamps = (
        df.loc[duplicate_mask, "timestamp"]
        .drop_duplicates()
        .sort_values()
    )

    print(
        f"{label}: {duplicate_rows} rows belong to "
        f"{len(duplicate_timestamps)} duplicate timestamps."
    )

    combined_rows = []

    for timestamp, group in df.groupby(
        "timestamp",
        sort=True,
        dropna=False,
    ):
        combined_rows.append(
            combine_duplicate_rows(group)
        )

    result = pd.DataFrame(combined_rows)

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=False,
    )

    return result.reset_index(drop=True)


# Time-axis handling

def create_complete_time_axis(
    df: pd.DataFrame,
    interval: pd.Timedelta,
    label: str,
) -> pd.DataFrame:
    """
    Create a continuous time axis and insert missing measurements.

    The time axis always starts at START_DATE and ends at the last
    measurement timestamp found in the input data.

    Existing measurements remain unchanged. Missing timestamps are
    inserted with NaN measurement values.

    Parameters
    ----------
    df
        Deduplicated measurements.
    interval
        Expected measurement interval.
    label
        Dataset label used for console output.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a continuous time axis.
    """
    if df.empty:
        return df

    df = df.sort_values("timestamp").copy()

    df = df.drop_duplicates(
        subset="timestamp",
        keep="first",
    )

    last_timestamp = df["timestamp"].max()

    full_index = pd.date_range(
        start=START_DATE,
        end=last_timestamp,
        freq=interval,
        tz=DEVICE_TIMEZONE,
    )

    df = df.set_index("timestamp")

    result = df.reindex(full_index)

    result.index.name = "timestamp"

    result = result.reset_index()

    missing_count = len(result) - len(df)

    if missing_count > 0:
        print(
            f"{label}: Added {missing_count} missing time step(s) as NaN."
        )

    return result


def add_missing_averaging_period(
    df: pd.DataFrame,
    averaging_period: str,
) -> pd.DataFrame:
    """
    Fill the averaging-period metadata for inserted rows.

    The averaging period describes the instrument configuration and
    is therefore retained even when the actual measurement values
    are missing.

    Parameters
    ----------
    df
        Complete time-axis DataFrame.
    averaging_period
        Standard averaging period for this dataset.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame.
    """
    if df.empty or "averagingPeriod" not in df.columns:
        return df

    df["averagingPeriod"] = (
        df["averagingPeriod"].fillna(averaging_period)
    )

    return df


def detect_measurement_interval(
    df: pd.DataFrame,
    label: str,
) -> pd.Timedelta | None:
    """
    Detect the most common interval between measurements.

    Parameters
    ----------
    df
        Measurement DataFrame.
    label
        Dataset label used for console output.

    Returns
    -------
    pandas.Timedelta or None
        Most frequently occurring time interval.
    """
    if df.empty or len(df) < 2:
        return None

    timestamps = (
        df["timestamp"]
        .sort_values()
        .drop_duplicates()
    )

    differences = timestamps.diff().dropna()

    if differences.empty:
        return None

    counts = differences.value_counts()

    print(f"{label} time difference statistics:")
    print(counts.head(10))

    interval = counts.index[0]

    print(
        f"Detected typical measurement interval: "
        f"{interval} ({counts.iloc[0]} occurrences)"
    )

    return interval


# Dataset combination

def combine_files(
    files: list[Path],
    label: str,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    """
    Read, filter, combine and regularize multiple BLS files.

    Parameters
    ----------
    files
        Input files.
    label
        Dataset label, for example 'DGN' or 'MND'.
    interval
        Expected regular measurement interval.

    Returns
    -------
    pandas.DataFrame
        Combined continuous dataset.
    """
    if not files:
        print(f"No {label} files found.")
        return pd.DataFrame()

    print()
    print(f"Combining {len(files)} {label} file(s)...")

    dataframes = []

    for file in files:
        print(f"Reading {file.name}")

        try:
            df = read_bls_file(file)

            if df.empty:
                print("  No measurement rows found.")
                continue

            df = filter_start_date(
                df,
                pd.Timestamp(START_DATE),
            )

            if df.empty:
                print("  0 valid rows after date filtering")
                continue

            df["_source_file"] = file.stem

            print(
                f"  -> {len(df)} valid rows after date filtering"
            )

            dataframes.append(df)

        except Exception as exc:
            print(
                f"  ERROR while reading {file.name}: {exc}"
            )

    if not dataframes:
        print(f"No valid {label} data available.")
        return pd.DataFrame()

    combined = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    combined = combined.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    combined = remove_duplicate_timestamps(
        combined,
        label,
    )

    if "_source_file" in combined.columns:
        combined = combined.drop(
            columns="_source_file"
        )

    detected_interval = detect_measurement_interval(
        combined,
        label,
    )

    if detected_interval is not None:
        if abs(
            detected_interval.total_seconds()
            - interval.total_seconds()
        ) > 0.1:
            print(
                f"Warning: Detected interval {detected_interval} "
                f"differs from expected interval {interval}."
            )

    combined = create_complete_time_axis(
        combined,
        interval,
        label,
    )

    if "averagingPeriod" in combined.columns:
        averaging_period = (
            combined["averagingPeriod"]
            .dropna()
            .iloc[0]
            if combined["averagingPeriod"].notna().any()
            else None
        )

        if averaging_period is not None:
            combined = add_missing_averaging_period(
                combined,
                averaging_period,
            )

    return combined


# Diagnostics

def print_dataset_summary(
    df: pd.DataFrame,
    label: str,
) -> None:
    """
    Print a concise summary of a combined dataset.

    Parameters
    ----------
    df
        Combined DataFrame.
    label
        Dataset label.
    """
    print()
    print(f"{label} DATA SUMMARY")

    if df.empty:
        print(f"No combined {label} data available.")
        return

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"From: {df['timestamp'].min()}")
    print(f"To:   {df['timestamp'].max()}")

    print()
    print("First rows:")
    print(df.head())

    print()
    print("Last rows:")
    print(df.tail())

    measurement_columns = [
        column
        for column in df.columns
        if column not in {"timestamp", "averagingPeriod"}
    ]

    if measurement_columns:
        all_nan_mask = df[measurement_columns].isna().all(axis=1)

        print()
        print(
            f"{label}: "
            f"{int(all_nan_mask.sum())} rows containing "
            f"only NaN measurements"
        )


def print_loaded_files(
    dgn_files: list[Path],
    mnd_files: list[Path],
) -> None:
    """
    Print the files that contributed data to the import.

    Parameters
    ----------
    dgn_files
        DGN files containing accepted measurements.
    mnd_files
        MND files containing accepted measurements.
    """
    print()
    print("LOADED FILES")
    print()

    print("DGN files:")
    print([file.stem for file in dgn_files])

    print()
    print("MND files:")
    print([file.stem for file in mnd_files])


# Main loader

def load_bls900_data(
    folder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all relevant BLS DGN and MND data from a directory.

    Files containing measurements before 2026-08-10 are ignored.
    Files that contain no accepted measurements after filtering are
    not included in the combined datasets.

    Parameters
    ----------
    folder
        Directory containing BLS files.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Combined DGN and MND datasets.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Data folder does not exist: {folder}"
        )

    dgn_candidates = sorted(
        folder.glob("*.dgn")
    )

    mnd_candidates = sorted(
        folder.glob("*.mnd")
    )

    dgn_dataframes = []
    mnd_dataframes = []

    valid_dgn_files = []
    valid_mnd_files = []

    print()
    print("BLS900 DATA IMPORT")
    print()
    print(
        f"Using data from: "
        f"{START_DATE}"
    )
    print(
        "Device timestamp handling: fixed UTC+01:00"
    )

    for file in sorted(folder.iterdir(), key=file_sort_key):
        if file.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"Skipping {file.name}")
            continue

        label = file.suffix.lower().replace(".", "").upper()

        print()
        print(f"Reading {file.name}")

        try:
            df = read_bls_file(file)

            if df.empty:
                print("No measurement rows found.")
                continue

            df = filter_start_date(
                df,
                pd.Timestamp(START_DATE),
            )

            if df.empty:
                print("-> 0 valid rows after date filtering")
                continue

            print(
                f"-> {len(df)} valid rows after date filtering"
            )

            if label == "DGN":
                dgn_dataframes.append(df)
                valid_dgn_files.append(file)

            elif label == "MND":
                mnd_dataframes.append(df)
                valid_mnd_files.append(file)

        except Exception as exc:
            print(
                f"ERROR while reading {file.name}: {exc}"
            )

    dgn = combine_preloaded_dataframes(
        dgn_dataframes,
        "DGN",
        DGN_INTERVAL,
    )

    mnd = combine_preloaded_dataframes(
        mnd_dataframes,
        "MND",
        MND_INTERVAL,
    )

    print_loaded_files(
        valid_dgn_files,
        valid_mnd_files,
    )

    return dgn, mnd


def combine_preloaded_dataframes(
    dataframes: list[pd.DataFrame],
    label: str,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    """
    Combine already parsed BLS DataFrames.

    Parameters
    ----------
    dataframes
        Parsed input DataFrames.
    label
        Dataset label.
    interval
        Expected measurement interval.

    Returns
    -------
    pandas.DataFrame
        Combined and regularized dataset.
    """
    if not dataframes:
        print()
        print(f"No {label} data available.")
        return pd.DataFrame()

    combined = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    combined = combined.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    combined = remove_duplicate_timestamps(
        combined,
        label,
    )

    detected_interval = detect_measurement_interval(
        combined,
        label,
    )

    if detected_interval is not None:
        if abs(
            detected_interval.total_seconds()
            - interval.total_seconds()
        ) > 0.1:
            print(
                f"Warning: Detected interval {detected_interval} "
                f"differs from expected interval {interval}."
            )

    combined = create_complete_time_axis(
        combined,
        interval,
        label,
    )

    if "averagingPeriod" in combined.columns:
        valid_periods = (
            combined["averagingPeriod"]
            .dropna()
        )

        if not valid_periods.empty:
            combined = add_missing_averaging_period(
                combined,
                valid_periods.iloc[0],
            )

    return combined


# Script entry point

if __name__ == "__main__":

    dgn, mnd = load_bls900_data(
        DATA_FOLDER
    )

    print()
    print_dataset_summary(
        dgn,
        "DGN",
    )

    print()
    print_dataset_summary(
        mnd,
        "MND",
    )

    print()
    print("Done.")