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
import matplotlib
matplotlib.use("Agg")

# At the moment only with BLS900 Data
from scintillometer.data_import.import_bls import load_bls900_data

# Folder where all figures and tables will be stored
PLOT_FOLDER = Path(
    r"C:\Users\janni\PycharmProjects\exp_meteo\scintillometer\plots" r"\land_water"
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


def save_figure(filename: str) -> None:
    output_path = PLOT_FOLDER / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def prepare_dataframe(df: pd.DataFrame, remove_errors: bool = True) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["decimal_hour"] = df["hour"] + df["minute"] / 60

    return df


def calculate_diurnal_cycle(df: pd.DataFrame, averaging_interval: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["time_bin"] = (np.floor(df["decimal_hour"] / averaging_interval) * averaging_interval)

    mean_cycle = df.groupby("time_bin").mean(numeric_only=True)
    std_cycle = df.groupby("time_bin").std(numeric_only=True)

    for col in std_cycle.columns:
        mean_cycle[f"{col}_std"] = std_cycle[col]

    return mean_cycle.reset_index()


def plot_diurnal_variable(
    cycle: pd.DataFrame,
    variable: str,
    ylabel: str,
    title: str,
    filename: str,
    show_std: bool = True,
):
    if cycle.empty:
        print(f"Kein Plot für {variable}: DataFrame ist leer.")
        return

    if variable not in cycle.columns:
        print(f"Kein Plot für {variable}: Spalte fehlt.")
        return

    x = cycle["time_bin"]
    y = cycle[variable]

    plt.figure()
    plt.plot(x, y, marker="o", linewidth=2, label="Mean")

    std_col = f"{variable}_std"
    if show_std and std_col in cycle.columns:
        std = cycle[std_col]
        plt.fill_between(x, y - std, y + std, alpha=0.25, label="±1 Std")

    plt.xlabel("Local time (hours)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 3))
    plt.legend()
    plt.tight_layout()

    save_figure(filename)


def plot_cn2(cycle):
    plot_diurnal_variable(
        cycle, "Cn2",
        ylabel=r"$C_n^2$ [$m^{-2/3}$]",
        title="Mean Diurnal Cycle of Cn2",
        filename="cn2_full_period.png"
    )

def plot_ct2(cycle):
    plot_diurnal_variable(
        cycle, "CT2",
        ylabel=r"$C_T^2$ [$K^2 m^{-2/3}$]",
        title="Mean Diurnal Cycle of CT2",
        filename="ct2_full_period.png"
    )

def plot_crosswind(cycle):
    plot_diurnal_variable(
        cycle, "crosswind",
        ylabel="Crosswind [m/s]",
        title="Mean Diurnal Cycle of Crosswind",
        filename="crosswind_full_period.png"
    )

def run_analysis_from_import(dgn: pd.DataFrame, mnd: pd.DataFrame):

    print("Analyse startet...")

    # Wir nutzen NUR mnd, weil dort Cn2, CT2, Crosswind drin sind
    df = mnd.copy()

    print("Messungen im MND:", len(df))
    print("Spalten:", df.columns)

    # Zeitstempel vorbereiten
    if "timestamp" not in df.columns:
        raise ValueError("Spalte 'timestamp' fehlt im MND-Dataset!")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # DataFrame für Diurnal Cycle vorbereiten
    df = prepare_dataframe(df)

    # Gesamtzeitraum
    cycle_full = calculate_diurnal_cycle(df)

    plot_cn2(cycle_full)
    plot_ct2(cycle_full)
    plot_crosswind(cycle_full)

    # Tagesplots
    df["date"] = df["timestamp"].dt.date

    for date, df_day in df.groupby("date"):
        print("Tagesplot:", date)
        cycle_day = calculate_diurnal_cycle(df_day)

        plot_diurnal_variable(
            cycle_day, "Cn2",
            ylabel=r"$C_n^2$ [$m^{-2/3}$]",
            title=f"Diurnal Cycle Cn2 – {date}",
            filename=f"cn2_diurnal_{date}.png"
        )

        plot_diurnal_variable(
            cycle_day, "CT2",
            ylabel=r"$C_T^2$ [$K^2 m^{-2/3}$]",
            title=f"Diurnal Cycle CT2 – {date}",
            filename=f"ct2_diurnal_{date}.png"
        )

        plot_diurnal_variable(
            cycle_day, "crosswind",
            ylabel="Crosswind [m/s]",
            title=f"Diurnal Cycle Crosswind – {date}",
            filename=f"crosswind_diurnal_{date}.png"
        )

