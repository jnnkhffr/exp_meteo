"""
Research Question:
How is turbulence influenced over land compared to water during typical diurnal
cycles over coastal heterogeneous terrain?

This script processes scintillometer measurements to investigate diurnal
variations of turbulence-related quantities (Cn2, CT2, crosswind) and
meteorological variables. The output consists of publication-quality plots
and statistical summaries.

The analysis is designed for BLS900 scintillometer data but can also be applied
to other scintillometer datasets with equivalent variables.
"""


from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scintillometer.data_import.import_bls900 import load_bls900_data

# Folder where all figures and tables will be stored
PLOT_FOLDER = Path(
    r"C:\Users\janni\PycharmProjects\exp_meteo\scintillometer\plots"
    r"\land_water"
)

# Create folder automatically if it does not exist
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)


# Matplotlib style settings
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 300,
    }
)


# Data preparation functions
def combine_daily_files(
    daily_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine multiple daily scintillometer files into one dataframe.

    Parameters
    ----------
    daily_data : dict
        Dictionary containing daily measurement dataframes.
        Keys are dates, values are pandas DataFrames.

    Returns
    -------
    pandas.DataFrame
        Combined dataframe containing all measurement days.
    """

    if not daily_data:
        raise ValueError("No measurement data provided.")

    df = pd.concat(
        daily_data.values(),
        ignore_index=True
    )

    return df



def prepare_dataframe(
    df: pd.DataFrame,
    remove_errors: bool = True
) -> pd.DataFrame:
    """
    Prepare scintillometer data for diurnal cycle analysis.

    The function:
    - converts timestamps
    - removes invalid measurements
    - creates time variables needed for averaging

    Parameters
    ----------
    df : pandas.DataFrame
        Raw scintillometer dataframe.

    remove_errors : bool, optional
        If True, measurements with error flags are removed.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe ready for analysis.
    """

    df = df.copy()


    # Ensure timestamp format
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )


    # Quality control
    if remove_errors:

        if "error" in df.columns:
            df = df[
                df["error"] == 0
            ]

        if "channelFlagsCombined" in df.columns:
            df = df[
                df["channelFlagsCombined"] == 0
            ]


    # Remove missing values from important variables
    required_columns = [
        "Cn2",
        "CT2",
        "crosswind",
        "temperature",
        "humidity"
    ]

    existing_columns = [
        col for col in required_columns
        if col in df.columns
    ]

    df = df.dropna(
        subset=existing_columns
    )


    # Create time variables
    df["hour"] = (
        df["timestamp"]
        .dt.hour
    )

    df["minute"] = (
        df["timestamp"]
        .dt.minute
    )


    # Decimal hour allows smoother diurnal plots
    df["decimal_hour"] = (
        df["hour"]
        +
        df["minute"] / 60
    )


    return df



def calculate_diurnal_cycle(
    df: pd.DataFrame,
    averaging_interval: float = 0.5
) -> pd.DataFrame:
    """
    Calculate the mean diurnal cycle.

    Measurements are grouped according to decimal hour.
    Standard deviations are calculated to show variability.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared scintillometer dataframe.

    averaging_interval : float
        Time resolution in hours.

        Example:
        0.5 -> 30 minute averages
        1.0 -> hourly averages

    Returns
    -------
    pandas.DataFrame
        Mean and standard deviation values for each time interval.
    """

    df = df.copy()


    # Create time bins
    df["time_bin"] = (
        np.floor(
            df["decimal_hour"]
            /
            averaging_interval
        )
        *
        averaging_interval
    )


    mean_cycle = (
        df
        .groupby("time_bin")
        .mean(numeric_only=True)
    )


    std_cycle = (
        df
        .groupby("time_bin")
        .std(numeric_only=True)
    )


    # Add standard deviation columns

    for column in std_cycle.columns:

        mean_cycle[
            f"{column}_std"
        ] = std_cycle[column]


    mean_cycle = (
        mean_cycle
        .reset_index()
    )


    return mean_cycle


# Plotting functions
def save_figure(
    filename: str
) -> None:
    """
    Save the current matplotlib figure.

    Parameters
    ----------
    filename : str
        Name of the output file.
    """

    output_path = PLOT_FOLDER / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_diurnal_variable(
    cycle: pd.DataFrame,
    variable: str,
    ylabel: str,
    title: str,
    filename: str,
    show_std: bool = True
) -> None:
    """
    Create a diurnal cycle plot for one variable.

    The plot shows the mean daily variation and optionally
    the standard deviation.

    Parameters
    ----------
    cycle : pandas.DataFrame
        Dataframe containing the calculated diurnal cycle.

    variable : str
        Name of the variable to plot.

    ylabel : str
        Label for the y-axis.

    title : str
        Plot title.

    filename : str
        Output filename.

    show_std : bool, optional
        If True, the standard deviation range is shown.
    """

    if variable not in cycle.columns:
        print(
            f"Warning: {variable} not available. Skipping plot."
        )
        return


    x = cycle["time_bin"]

    y = cycle[variable]


    plt.figure()


    # Main line
    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        label="Mean"
    )


    # Standard deviation envelope

    std_column = f"{variable}_std"

    if show_std and std_column in cycle.columns:

        std = cycle[std_column]

        plt.fill_between(
            x,
            y - std,
            y + std,
            alpha=0.25,
            label="±1 standard deviation"
        )


    plt.xlabel(
        "Local time (hours)"
    )

    plt.ylabel(
        ylabel
    )


    plt.title(
        title
    )


    plt.xlim(
        0,
        24
    )


    plt.xticks(
        range(0, 25, 3)
    )


    plt.legend()

    plt.tight_layout()


    save_figure(
        filename
    )


# Individual research plots
def plot_cn2(
    cycle: pd.DataFrame
) -> None:
    """
    Plot the diurnal cycle of the refractive index
    structure parameter Cn2.

    Cn2 is used as a measure of atmospheric turbulence intensity.
    """

    plot_diurnal_variable(
        cycle=cycle,
        variable="Cn2",
        ylabel=r"$C_n^2$ [$m^{-2/3}$]",
        title="Mean Diurnal Cycle of Turbulence ($C_n^2$)",
        filename="cn2_diurnal_cycle.png"
    )



def plot_ct2(
    cycle: pd.DataFrame
) -> None:
    """
    Plot the diurnal cycle of the temperature
    structure parameter CT2.

    CT2 describes temperature fluctuations in the atmosphere.
    """

    plot_diurnal_variable(
        cycle=cycle,
        variable="CT2",
        ylabel=r"$C_T^2$ [$K^2 m^{-2/3}$]",
        title="Mean Diurnal Cycle of Temperature Turbulence ($C_T^2$)",
        filename="ct2_diurnal_cycle.png"
    )



def plot_crosswind(
    cycle: pd.DataFrame
) -> None:
    """
    Plot the diurnal variation of crosswind.

    Crosswind influences scintillometer sampling and
    provides information about atmospheric mixing.
    """

    plot_diurnal_variable(
        cycle=cycle,
        variable="crosswind",
        ylabel="Crosswind [m s$^{-1}$]",
        title="Mean Diurnal Cycle of Crosswind",
        filename="crosswind_diurnal_cycle.png"
    )



def plot_temperature(
    cycle: pd.DataFrame
) -> None:
    """
    Plot the diurnal variation of air temperature.
    """

    plot_diurnal_variable(
        cycle=cycle,
        variable="temperature",
        ylabel="Temperature [°C]",
        title="Mean Diurnal Cycle of Air Temperature",
        filename="temperature_diurnal_cycle.png"
    )



def plot_humidity(
    cycle: pd.DataFrame
) -> None:
    """
    Plot the diurnal variation of relative humidity.
    """

    plot_diurnal_variable(
        cycle=cycle,
        variable="humidity",
        ylabel="Relative Humidity [%]",
        title="Mean Diurnal Cycle of Relative Humidity",
        filename="humidity_diurnal_cycle.png"
    )


# Statistical analysis and workflow
def save_summary_statistics(
    df: pd.DataFrame
) -> None:
    """
    Save descriptive statistics of turbulence-related variables.

    The output file can be directly used for documentation
    or further scientific analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared measurement dataframe.
    """

    variables = [
        "Cn2",
        "CT2",
        "crosswind",
        "temperature",
        "humidity"
    ]


    available_variables = [
        var for var in variables
        if var in df.columns
    ]


    statistics = (
        df[available_variables]
        .describe()
        .transpose()
    )


    output_path = (
        PLOT_FOLDER /
        "summary_statistics.csv"
    )


    statistics.to_csv(
        output_path
    )



def run_analysis(
    mnd_data: dict
) -> pd.DataFrame:
    """
    Run the complete land-water turbulence analysis.

    Workflow:
    1. Combine daily measurements.
    2. Apply quality control.
    3. Calculate mean diurnal cycles.
    4. Generate scientific plots.
    5. Export statistical summary.

    Parameters
    ----------
    mnd_data : dict
        Dictionary containing daily scintillometer
        measurement dataframes.

    Returns
    -------
    pandas.DataFrame
        Calculated diurnal cycle dataframe.
    """


    print(
        "Combining daily measurements..."
    )

    df = combine_daily_files(
        mnd_data
    )


    print(
        f"Total measurements: {len(df)}"
    )


    print(
        "Preparing dataframe..."
    )

    df = prepare_dataframe(
        df
    )
    print("\n===== DATA TYPES =====")
    print(df.dtypes)


    print(
        f"Valid measurements after filtering: {len(df)}"
    )


    print(
        "Calculating diurnal cycle..."
    )

    cycle = calculate_diurnal_cycle(
        df
    )


    print(
        "Saving statistical summary..."
    )

    save_summary_statistics(
        df
    )


    print(
        "Creating plots..."
    )


    # Turbulence-related quantities

    plot_cn2(
        cycle
    )

    plot_ct2(
        cycle
    )

    plot_crosswind(
        cycle
    )


    # Meteorological variables

    plot_temperature(
        cycle
    )

    plot_humidity(
        cycle
    )


    print(
        "Analysis completed."
    )
    print(df["Cn2"].describe())
    print(df["CT2"].describe())
    print(df["crosswind"].describe())
    print(df["H_convection"].describe())


    return cycle



# Example execution
if __name__ == "__main__":

    """
    Example workflow.

    The scintillometer reader should be executed first.

    Example:

    from read_bls900 import load_bls900_data

    dgn, mnd = load_bls900_data(folder)

    run_analysis(mnd)

    """


    DATA_FOLDER = (
        r"C:\Users\janni\Desktop\Studium\Master"
        r"\Experimental Meteo\sample_data\BLS900"
    )


    print(
        "Loading BLS900 data..."
    )


    dgn, mnd = load_bls900_data(
        DATA_FOLDER
    )


    # Run analysis

    diurnal_cycle = run_analysis(
        mnd
    )


    print(
        "\nResulting diurnal cycle:"
    )

    print(
        diurnal_cycle.head()
    )
