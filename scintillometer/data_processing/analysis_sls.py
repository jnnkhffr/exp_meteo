from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scintillometer.data_import.import_sls200 import load_sls20_data


DATA_FOLDER = Path(r"X:\SLS20")

PLOT_FOLDER = Path(
    r"C:\Users\janni\PycharmProjects\exp_meteo\scintillometer\plots"
    r"\land_water"
)


def plot_heat_flux(res: pd.DataFrame) -> None:

    PLOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Check
    required_columns = {
        "timestamp",
        "Heat_day",
        "Heat_night",
    }

    missing = required_columns - set(res.columns)

    if missing:
        raise ValueError(
            f"Missing RES columns: {sorted(missing)}"
        )

    # Prepare timestamps
    res = res.copy()

    res["timestamp"] = pd.to_datetime(
        res["timestamp"]
    )

    res = res.sort_values(
        "timestamp"
    )

    # Plot all days
    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        res["timestamp"],
        res["Heat_day"],
        label="Heat day",
        color="tab:red",
        linewidth=1.0,
    )

    ax.plot(
        res["timestamp"],
        res["Heat_night"],
        label="Heat night",
        color="tab:blue",
        linewidth=1.0,
    )

    ax.set_title(
        "SLS20 – Sensible Heat Flux"
    )

    ax.set_xlabel(
        "Date and time"
    )

    ax.set_ylabel(
        "Sensible heat flux"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    output = (
        PLOT_FOLDER
        / "SLS20_sensible_heat_flux_all_days.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )

    # Individual days
    res["date"] = res["timestamp"].dt.date

    for date, day_data in res.groupby("date"):

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        ax.plot(
            day_data["timestamp"],
            day_data["Heat_day"],
            label="Heat day",
            color="tab:red",
            linewidth=1.2,
        )

        ax.plot(
            day_data["timestamp"],
            day_data["Heat_night"],
            label="Heat night",
            color="tab:blue",
            linewidth=1.2,
        )

        date_string = pd.Timestamp(date).strftime(
            "%Y-%m-%d"
        )

        ax.set_title(
            f"{date_string} – SLS20 – Sensible Heat Flux"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Sensible heat flux"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.autofmt_xdate()

        fig.tight_layout()

        output = (
            PLOT_FOLDER
            / f"{date_string}_SLS20_sensible_heat_flux.png"
        )

        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {output}"
        )


def main():

    print("=" * 80)
    print("SLS20 SENSIBLE HEAT FLUX PLOTS")
    print("=" * 80)

    print(
        f"Data folder:  {DATA_FOLDER}"
    )

    print(
        f"Plot folder:  {PLOT_FOLDER}"
    )

    print()
    print("Loading SLS20 data...")

    res, dgn = load_sls20_data(
        DATA_FOLDER
    )

    print()
    print(
        f"DGN rows: {len(dgn)}"
    )

    print(
        f"RES rows: {len(res)}"
    )

    print()
    print("RES columns:")
    print(list(res.columns))

    plot_heat_flux(res)

    print()
    print("Done.")


if __name__ == "__main__":
    main()