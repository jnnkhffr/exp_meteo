"""This script reads in the data from the BLS900 scintillometer."""

from pathlib import Path
import pandas as pd


def find_data_start(lines):
    """
    Find the first line containing measurement data.
    Measurement lines always start with 'PT'.
    """
    for i, line in enumerate(lines):
        if line.startswith("PT"):
            return i
    raise ValueError("No measurement data found.")


def read_dgn_file(file):
    """Read a BLS900 .dgn file."""

    with open(file, "r", errors="ignore") as f:
        lines = f.readlines()

    start = find_data_start(lines)

    df = pd.read_csv(
        file,
        sep=r"\t",
        header=None,
        skiprows=start,
        engine="python"
    )

    df.columns = [
        "time",
        "dgnCounter",
        "XA_mean_corrected",
        "YA_mean_corrected",
        "nSigXA",
        "nSigYA",
        "corXAYA_corrected",
        "numSamples",
        "XA_mean",
        "YA_mean",
        "XB_mean",
        "YB_mean",
        "sigXA",
        "sigYA",
        "sigXB",
        "sigYB",
        "minXA",
        "minYA",
        "minXB",
        "minYB",
        "maxXA",
        "maxYA",
        "maxXB",
        "maxYB",
        "corXAYA",
        "corXBYB",
        "corXAXB",
        "corYAYB",
        "corXAYB",
        "corYAXB",
        "crosswind",
        "sigCrosswind",
        "dmiGndmV",
        "autoBgXA",
        "autoBgYA",
        "autoBgXB",
        "autoBgYB",
        "bite12VExt",
        "bite12VInt",
        "biteMinus12VInt",
        "bite5VInt",
        "biteMinus5VInt",
        "bite12VWsp",
        "bite12VDmi",
        "bite12VReceiver",
        "biteMinus12VReceiver",
        "biteCooler",
        "channelFlagsXA",
        "channelFlagsYA",
        "channelFlagsXB",
        "channelFlagsYB",
        "error"
    ]

    # Split time column
    df[["averagingPeriod", "timestamp"]] = (
        df["time"].str.split("/", expand=True)
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.drop(columns="time", inplace=True)

    cols = ["timestamp", "averagingPeriod"] + [
        c for c in df.columns
        if c not in ["timestamp", "averagingPeriod"]
    ]
    df = df[cols]

    return df


def read_mnd_file(file):
    """Read a BLS900 .mnd file."""

    with open(file, "r", errors="ignore") as f:
        lines = f.readlines()

    start = find_data_start(lines)

    df = pd.read_csv(
        file,
        sep=r"\t",
        header=None,
        skiprows=start,
        engine="python"
    )

    df.columns = [
        "time",
        "Cn2",
        "CT2",
        "H_convection",
        "crosswind",
        "sigCrosswind",
        "pressure",
        "temperature",
        "humidity",
        "pathLength",
        "pathHeight",
        "correctCn2EO",
        "correctCn2Sat",
        "mndCounter",
        "XA_mean_corrected",
        "YA_mean_corrected",
        "nSigXA",
        "nSigYA",
        "corXAYA_corrected",
        "numDgnValid",
        "numDgnValidCrosswind",
        "numDgnTotal",
        "channelFlagsCombined",
        "error"
    ]

    # Split time column
    df[["averagingPeriod", "timestamp"]] = (
        df["time"].str.split("/", expand=True)
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.drop(columns="time", inplace=True)

    cols = ["timestamp", "averagingPeriod"] + [
        c for c in df.columns
        if c not in ["timestamp", "averagingPeriod"]
    ]
    df = df[cols]

    # Convert measurement columns to numeric values
    numeric_columns = [
        "Cn2",
        "CT2",
        "H_convection",
        "crosswind",
        "sigCrosswind",
        "pressure",
        "temperature",
        "humidity",
        "pathLength",
        "pathHeight",
        "correctCn2EO",
        "correctCn2Sat",
        "mndCounter",
        "XA_mean_corrected",
        "YA_mean_corrected",
        "nSigXA",
        "nSigYA",
        "corXAYA_corrected",
        "numDgnValid",
        "numDgnValidCrosswind",
        "numDgnTotal",
        "channelFlagsCombined",
        "error"
    ]


    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def load_bls900_data(folder_path):
    """
    Load all BLS900 files from a folder.

    Returns
    -------
    dgn_data : dict
        Dictionary of DGN DataFrames.
    mnd_data : dict
        Dictionary of MND DataFrames.
    """

    folder = Path(folder_path)

    dgn_data = {}
    mnd_data = {}

    for file in sorted(folder.iterdir()):

        # Skip hidden files
        if file.name.startswith("."):
            continue

        if file.suffix.lower() == ".dgn":
            print(f"Reading {file.name}")
            dgn_data[file.stem] = read_dgn_file(file)

        elif file.suffix.lower() == ".mnd":
            print(f"Reading {file.name}")
            mnd_data[file.stem] = read_mnd_file(file)

        elif file.suffix.lower() == ".device":
            # Ignore device files
            continue

        else:
            print(f"Skipping {file.name}")

    return dgn_data, mnd_data


if __name__ == "__main__":

    folder = r"C:\Users\janni\Desktop\Studium\Master\Experimental Meteo\sample_data\BLS900"

    dgn, mnd = load_bls900_data(folder)

    print("\nLoaded DGN files:")
    print(list(dgn.keys()))

    print("\nLoaded MND files:")
    print(list(mnd.keys()))

    # Example
    first_day = next(iter(mnd))
    print("\nMND preview:")
    print(mnd[first_day].head())

    first_day = next(iter(dgn))
    print("\nDGN preview:")
    print(dgn[first_day].head())

print("stop")