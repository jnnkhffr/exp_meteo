from pathlib import Path

from datetime import timezone, timedelta

from astral import Observer
from astral.sun import sun

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from scintillometer.data_import.import_sls200 import load_sls20_data

DATA_FOLDER = Path(r"X:\SLS20")

PLOT_FOLDER = Path(
    r"C:\Users\janni\PycharmProjects\exp_meteo\scintillometer\plots"
    r"\land_water"
)

CSV_FOLDER = Path(
    r"X:\BLS2000\placetohitsomeone"
)

CSV_FILENAME = (
    "SLS20_sensible_heat_flux_combined.csv"
)

LATITUDE = 54.527748
LONGITUDE = 11.060508

CET = timezone(timedelta(hours=1))

# Length of the continuous interval used to identify
# a stable transition.
TRANSITION_WINDOW_MINUTES = 5

# Minimum number of measurements that must exist
# inside the interval.
TRANSITION_MIN_POINTS = 4

# SETTINGS FOR DAY/NIGHT TRANSITION

# Morning transition:
# Search for the point where Heat_day and Heat_night are closest
# within this time window.
MORNING_SEARCH_START = 6
MORNING_SEARCH_END = 7

# Evening transition:
# Search for the point where Heat_day and Heat_night are closest
# within this time window.
EVENING_SEARCH_START = 18
EVENING_SEARCH_END = 19


# OPTIONAL HARDCODED TRANSITIONS
# If the automatic detection gives a bad result for a specific day,
# you can define the transition times manually here.
#
# Format:
#
# "DD.MM.YYYY": {
#     "morning": "06:00",
#     "evening": "18:00",
# }
#
# Only days that need correction have to be entered.
#
HARDCODED_TRANSITIONS = {
    # "01.08.2026": {
    #     "morning": "06:00",
    #     "evening": "18:00",
    # },
}

def prepare_sls20_timestamps(
    res: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare SLS20 timestamps for plotting.

    The timestamps represent local wall-clock time.
    Any timezone information is therefore removed WITHOUT
    converting the clock time.

    Example:
        2026-08-13 18:29:00+01:00
    becomes:
        2026-08-13 18:29:00

    The clock time itself is not changed.
    """

    res = res.copy()

    res["timestamp"] = pd.to_datetime(
        res["timestamp"]
    )

    # Remove timezone information while preserving
    # the displayed local clock time.
    if isinstance(
        res["timestamp"].dtype,
        pd.DatetimeTZDtype,
    ):
        res["timestamp"] = (
            res["timestamp"]
            .dt.tz_localize(None)
        )

    return res


def format_time_axis(ax):
    """
    Format the x-axis using the local wall-clock timestamps
    contained in the SLS20 data.
    """

    ax.xaxis.set_major_locator(
        mdates.HourLocator(
            byhour=[0, 6, 12, 18],
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%H:%M",
        )
    )

    ax.set_xlabel(
        "Time [CET]"
    )


def add_day_labels(ax, dates):
    """
    Add one rotated date label centered underneath each day.
    """

    for date in dates:

        start = pd.Timestamp(date)

        center = start + pd.Timedelta(
            hours=12
        )

        ax.text(
            center,
            -0.16,
            start.strftime("%d.%m.%Y"),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            rotation=45,
            rotation_mode="anchor",
        )


def get_sunrise_sunset(
    date,
):
    """
    Calculate sunrise and sunset for the given date
    at the specified geographic coordinates.

    The calculation uses a fixed CET timezone
    (UTC+1), not Europe/Berlin.

    This means that daylight-saving time (CEST) is
    intentionally NOT applied.

    The returned timestamps are naive timestamps,
    matching the wall-clock timestamps used by the
    SLS20 data.
    """

    observer = Observer(
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )

    solar_times = sun(
        observer,
        date=date,
        tzinfo=CET,
    )

    sunrise = solar_times[
        "sunrise"
    ].replace(
        tzinfo=None
    )

    sunset = solar_times[
        "sunset"
    ].replace(
        tzinfo=None
    )

    return sunrise, sunset


def find_transition_point(
    day_data: pd.DataFrame,
    start_hour: int,
    end_hour: int,
):
    """
    Find the transition using a continuous time window.

    The search window is evaluated using consecutive time intervals.

    Example with TRANSITION_WINDOW_MINUTES = 5:

        06:00 - 06:05
        06:01 - 06:06
        06:02 - 06:07
        ...

    For every interval, the mean absolute difference between
    Heat_day and Heat_night is calculated.

    The interval with the smallest mean difference is selected.

    The transition time is the center of the selected interval.
    """

    # SEARCH WINDOW
    start_time = (
        day_data["timestamp"].dt.normalize()
        + pd.Timedelta(hours=start_hour)
    )

    end_time = (
        day_data["timestamp"].dt.normalize()
        + pd.Timedelta(hours=end_hour)
    )

    mask = (
        (day_data["timestamp"] >= start_time)
        & (day_data["timestamp"] <= end_time)
    )

    window = day_data.loc[
        mask,
        [
            "timestamp",
            "Heat_day",
            "Heat_night",
        ],
    ].copy()

    window = window.sort_values(
        "timestamp"
    )

    # Remove invalid measurements
    window = window.dropna(
        subset=[
            "Heat_day",
            "Heat_night",
        ]
    )

    if window.empty:
        return None

    # DIFFERENCE BETWEEN DAY AND NIGHT CURVE
    window["difference"] = (
        window["Heat_day"]
        - window["Heat_night"]
    ).abs()

    # SLIDING TIME WINDOWS
    candidates = []

    interval = pd.Timedelta(
        minutes=TRANSITION_WINDOW_MINUTES
    )

    timestamps = window["timestamp"].tolist()

    for start_timestamp in timestamps:

        end_timestamp = (
            start_timestamp
            + interval
        )

        interval_data = window[
            (
                window["timestamp"]
                >= start_timestamp
            )
            & (
                window["timestamp"]
                <= end_timestamp
            )
        ]

        # Need enough actual measurements
        if len(interval_data) < TRANSITION_MIN_POINTS:
            continue

        # Make sure the interval really contains data
        # throughout the requested time period.
        actual_duration = (
            interval_data["timestamp"].max()
            - interval_data["timestamp"].min()
        )

        if actual_duration < interval * 0.8:
            continue

        # QUALITY OF THIS INTERVAL
        mean_difference = (
            interval_data["difference"].mean()
        )

        max_difference = (
            interval_data["difference"].max()
        )

        std_difference = (
            interval_data["difference"].std()
        )

        if pd.isna(std_difference):
            std_difference = 0.0

        candidates.append(
            {
                "start": start_timestamp,
                "end": end_timestamp,
                "mean_difference": mean_difference,
                "max_difference": max_difference,
                "std_difference": std_difference,
            }
        )

    if not candidates:
        return None

    # FIND BEST CONTINUOUS INTERVAL
    candidates = sorted(
        candidates,
        key=lambda x: (
            x["mean_difference"],
            x["std_difference"],
            x["max_difference"],
        ),
    )

    best = candidates[0]

    # TRANSITION = CENTER OF INTERVAL
    transition = (
        best["start"]
        + (
            best["end"]
            - best["start"]
        ) / 2
    )

    return transition


def get_transition_times(
    day_data: pd.DataFrame,
    date,
):
    """
    Determine morning and evening transition times.

    First checks for manually specified transition times.
    Otherwise the transition is determined automatically.
    """

    date_string = pd.Timestamp(
        date
    ).strftime(
        "%d.%m.%Y"
    )

    # HARDCODED TRANSITION
    if date_string in HARDCODED_TRANSITIONS:

        settings = HARDCODED_TRANSITIONS[
            date_string
        ]

        morning = pd.Timestamp(
            f"{date_string} {settings['morning']}"
        )

        evening = pd.Timestamp(
            f"{date_string} {settings['evening']}"
        )

        return morning, evening

    # AUTOMATIC TRANSITION
    morning = find_transition_point(
        day_data,
        MORNING_SEARCH_START,
        MORNING_SEARCH_END,
    )

    evening = find_transition_point(
        day_data,
        EVENING_SEARCH_START,
        EVENING_SEARCH_END,
    )

    return morning, evening


def create_combined_heat_flux(
    day_data: pd.DataFrame,
    morning_transition,
    evening_transition,
):
    """
    Create one combined sensible heat flux curve.

    Before sunrise:
        Heat_night

    Between sunrise and evening transition:
        Heat_day

    After evening transition:
        Heat_night

    No interpolation is performed at the transitions.
    """

    combined = pd.Series(
        index=day_data.index,
        dtype=float,
    )

    # NIGHT BEFORE MORNING TRANSITION
    night_before_morning = (
        day_data["timestamp"]
        < morning_transition
    )

    combined.loc[
        night_before_morning
    ] = day_data.loc[
        night_before_morning,
        "Heat_night",
    ]

    # DAY
    day_period = (
        (day_data["timestamp"] >= morning_transition)
        & (
            day_data["timestamp"]
            <= evening_transition
        )
    )

    combined.loc[
        day_period
    ] = day_data.loc[
        day_period,
        "Heat_day",
    ]

    # NIGHT AFTER EVENING TRANSITION
    night_after_evening = (
        day_data["timestamp"]
        > evening_transition
    )

    combined.loc[
        night_after_evening
    ] = day_data.loc[
        night_after_evening,
        "Heat_night",
    ]

    return combined


def get_annotation_position(
    ax,
    timestamp,
    value,
    day_data,
    position="top",
):
    """
    Calculate a suitable position for a transition annotation.

    Morning transition:
        annotation is placed above and to the LEFT.

    Evening transition:
        annotation is placed below and to the RIGHT.

    Returns
    -------
    tuple
        (x_offset, y_offset) in display points.
    """

    # LOCAL DATA AROUND TRANSITION
    window_start = (
        timestamp
        - pd.Timedelta(minutes=60)
    )

    window_end = (
        timestamp
        + pd.Timedelta(minutes=60)
    )

    local_data = day_data[
        (
            day_data["timestamp"]
            >= window_start
        )
        & (
            day_data["timestamp"]
            <= window_end
        )
    ].copy()

    # FALLBACK
    if local_data.empty:

        if position == "top":
            return -80, 50

        return 15, -55

    # LOCAL VALUES
    local_values = pd.concat(
        [
            local_data["Heat_day"],
            local_data["Heat_night"],
        ]
    ).dropna()

    if local_values.empty:

        if position == "top":
            return -80, 50

        return 15, -55

    local_min = local_values.min()
    local_max = local_values.max()

    local_range = (
        local_max
        - local_min
    )

    if local_range == 0:
        local_range = 1.0

    # MORNING: ABOVE + LEFT
    if position == "top":

        distance = max(
            local_range * 0.35,
            10,
        )

        y_offset = (
            35
            + (
                distance
                / local_range
                * 20
            )
        )

        # Negative = move text to the LEFT
        x_offset = -85

        return x_offset, y_offset

    # EVENING: BELOW + RIGHT
    distance = max(
        local_range * 0.35,
        10,
    )

    y_offset = (
        -35
        - (
            distance
            / local_range
            * 20
        )
    )

    # Positive = move text to the RIGHT
    x_offset = 15

    return x_offset, y_offset


def save_combined_heat_flux_csv(
    res: pd.DataFrame,
) -> None:
    """
    Save the combined black Heat_combined curve
    for all days into one CSV file.

    Only timestamps with a valid combined heat flux
    value are saved.
    """

    CSV_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CHECK REQUIRED COLUMNS
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

    # PREPARE DATA
    res = res.copy()

    res["timestamp"] = pd.to_datetime(
        res["timestamp"]
    )

    res = res.sort_values(
        "timestamp"
    )

    res["date"] = res[
        "timestamp"
    ].dt.date

    combined_rows = []

    # PROCESS EACH DAY
    for date, day_data in res.groupby(
        "date"
    ):

        day_data = day_data.copy()

        # FIND TRANSITIONS
        morning_transition, evening_transition = (
            get_transition_times(
                day_data,
                date,
            )
        )

        # Skip day if transitions cannot be determined
        if (
            morning_transition is None
            or evening_transition is None
        ):
            print(
                f"WARNING: Could not determine transitions for "
                f"{pd.Timestamp(date).strftime('%d.%m.%Y')} "
                f"for CSV export"
            )
            continue

        # CREATE THE SAME BLACK CURVE
        day_data["heat_combined"] = (
            create_combined_heat_flux(
                day_data,
                morning_transition,
                evening_transition,
            )
        )

        # KEEP ONLY VALID COMBINED VALUES
        valid = day_data[
            day_data["heat_combined"].notna()
        ].copy()

        if valid.empty:
            continue

        # Add transition information
        valid["morning_transition"] = (
            morning_transition
        )

        valid["evening_transition"] = (
            evening_transition
        )

        combined_rows.append(
            valid[
                [
                    "timestamp",
                    "heat_combined",
                    "morning_transition",
                    "evening_transition",
                ]
            ]
        )

    # CHECK IF ANY DATA EXISTS
    if not combined_rows:
        print(
            "WARNING: No valid combined heat flux data "
            "available for CSV export."
        )
        return

    # COMBINE ALL DAYS
    output_data = pd.concat(
        combined_rows,
        ignore_index=True,
    )

    output_data = output_data.sort_values(
        "timestamp"
    )

    # SAVE CSV
    output = (
        CSV_FOLDER
        / CSV_FILENAME
    )

    output_data.to_csv(
        output,
        index=False,
        sep=";",
        decimal=".",
    )

    print(
        f"Saved combined heat flux CSV: {output}"
    )

    print(
        f"CSV rows: {len(output_data)}"
    )


def plot_combined_daily_cycle(
    res: pd.DataFrame,
) -> None:
    """
    Create an additional plot for every day containing:

    - combined day/night curve in the foreground
    - Heat_day in the background with 50% transparency
    - Heat_night in the background with 50% transparency
    - automatic or manually defined transition points
    - labels showing the transition times
    """

    PLOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CHECK REQUIRED COLUMNS
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

    # PREPARE DATA
    res = res.copy()

    res["timestamp"] = pd.to_datetime(
        res["timestamp"]
    )

    res = res.sort_values(
        "timestamp"
    )

    res["date"] = res[
        "timestamp"
    ].dt.date

    # INDIVIDUAL DAILY PLOTS
    for date, day_data in res.groupby(
        "date"
    ):

        day_data = day_data.copy()

        # FIND TRANSITIONS
        morning_transition, evening_transition = (
            get_transition_times(
                day_data,
                date,
            )
        )

        # If a transition could not be found, skip the day.
        if (
            morning_transition is None
            or evening_transition is None
        ):
            print(
                f"WARNING: Could not determine transitions for "
                f"{pd.Timestamp(date).strftime('%d.%m.%Y')}"
            )

            continue

        # CREATE COMBINED CURVE
        day_data["Heat_combined"] = (
            create_combined_heat_flux(
                day_data,
                morning_transition,
                evening_transition,
            )
        )

        date_string = pd.Timestamp(
            date
        ).strftime(
            "%d.%m.%Y"
        )

        # SUNRISE / SUNSET
        sunrise, sunset = get_sunrise_sunset(
            date
        )

        # FIGURE
        fig, ax = plt.subplots(
            figsize=(14, 6)
        )


        # BACKGROUND: ORIGINAL DAY CURVE
        ax.plot(
            day_data["timestamp"],
            day_data["Heat_day"],
            label="Heat day (background)",
            color="tab:red",
            linewidth=1.0,
            alpha=0.25,
        )

        # BACKGROUND: ORIGINAL NIGHT CURVE
        ax.plot(
            day_data["timestamp"],
            day_data["Heat_night"],
            label="Heat night (background)",
            color="tab:blue",
            linewidth=1.0,
            alpha=0.25,
        )

        # FOREGROUND: COMBINED CURVE
        ax.plot(
            day_data["timestamp"],
            day_data["Heat_combined"],
            label="combined diurnal cycle",
            color="black",
            linewidth=1.0,
            zorder=5,
        )

        # SUNRISE / SUNSET LINES
        # The solar times are calculated in fixed CET (UTC+1)
        ax.axvline(
            sunrise,
            color="darkorange",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )

        ax.axvline(
            sunset,
            color="darkorange",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )

        # SUNRISE LABEL
        ax.text(
            sunrise,
            0.97,
            f"Sunrise\n{sunrise.strftime('%H:%M')}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="darkorange",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

        # SUNSET LABEL
        ax.text(
            sunset,
            0.97,
            f"Sunset\n{sunset.strftime('%H:%M')}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="darkorange",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

        # TRANSITION POINTS
        # Find actual values at transition points
        morning_row = day_data.iloc[
            (
                day_data["timestamp"]
                - morning_transition
            ).abs().argsort()[:1]
        ]

        evening_row = day_data.iloc[
            (
                day_data["timestamp"]
                - evening_transition
            ).abs().argsort()[:1]
        ]

        # MORNING POINT
        morning_value_day = morning_row[
            "Heat_day"
        ].iloc[0]

        morning_value_night = morning_row[
            "Heat_night"
        ].iloc[0]

        morning_value = (
            morning_value_day
            + morning_value_night
        ) / 2

        ax.axvline(
            morning_transition,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        ax.scatter(
            morning_transition,
            morning_value,
            color="black",
            s=15,
            zorder=10,
        )

        morning_x_offset, morning_y_offset = (
            get_annotation_position(
                ax,
                morning_transition,
                morning_value,
                day_data,
                position="top",
            )
        )

        ax.annotate(
            (
                "Night → Day\n"
                f"{morning_transition.strftime('%H:%M')}"
            ),
            xy=(
                morning_transition,
                morning_value,
            ),
            xytext=(
                morning_x_offset,
                morning_y_offset,
            ),
            textcoords="offset points",
            fontsize=9,
            ha="right",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color="black",
            ),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

        # EVENING POINT
        evening_value_day = evening_row[
            "Heat_day"
        ].iloc[0]

        evening_value_night = evening_row[
            "Heat_night"
        ].iloc[0]

        evening_value = (
            evening_value_day
            + evening_value_night
        ) / 2

        ax.axvline(
            evening_transition,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        ax.scatter(
            evening_transition,
            evening_value,
            color="black",
            s=15,
            zorder=10,
        )

        evening_x_offset, evening_y_offset = (
            get_annotation_position(
                ax,
                evening_transition,
                evening_value,
                day_data,
                position="bottom",
            )
        )

        ax.annotate(
            (
                "Day → Night\n"
                f"{evening_transition.strftime('%H:%M')}"
            ),
            xy=(
                evening_transition,
                evening_value,
            ),
            xytext=(
                evening_x_offset,
                evening_y_offset,
            ),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="->",
                color="black",
            ),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

        # LABEL THE REGIONS
        # Determine y-position for region labels
        y_min, y_max = ax.get_ylim()

        y_label = y_min + 0.92 * (
            y_max - y_min
        )

        # Night before sunrise
        ax.text(
            (
                day_data["timestamp"].min()
                + (
                    morning_transition
                    - day_data["timestamp"].min()
                ) / 2
            ),
            y_label,
            "Night curve visible",
            ha="center",
            va="top",
            fontsize=9,
            color="tab:blue",
            alpha=0.8,
        )

        # Day
        ax.text(
            (
                morning_transition
                + (
                    evening_transition
                    - morning_transition
                ) / 2
            ),
            y_label,
            "Day curve visible",
            ha="center",
            va="top",
            fontsize=9,
            color="tab:red",
            alpha=0.8,
        )

        # Night after sunset
        ax.text(
            (
                evening_transition
                + (
                    day_data["timestamp"].max()
                    - evening_transition
                ) / 2
            ),
            y_label,
            "Night curve visible",
            ha="center",
            va="top",
            fontsize=9,
            color="tab:blue",
            alpha=0.8,
        )

        # FORMATTING
        ax.set_title(
            (
                f"{date_string} – SLS20 – "
                "combined sensible heat flux diurnal cycle "
            )
        )

        ax.set_ylabel(
            "Sensible heat flux [$W/m^2$]"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        format_time_axis(
            ax
        )

        ax.legend()

        fig.tight_layout()

        # SAVE
        output = (
            PLOT_FOLDER
            / (
                f"{date_string}_"
                "SLS20_sensible_heat_flux_combined.png"
            )
        )

        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved combined plot: {output}"
        )

        print(
            f"  Morning transition: "
            f"{morning_transition.strftime('%H:%M')}"
        )

        print(
            f"  Evening transition: "
            f"{evening_transition.strftime('%H:%M')}"
        )


def plot_combined_all_days(
    res: pd.DataFrame,
) -> None:
    """
    Create one combined plot containing all days.

    For every day:
        - Heat_day is shown in the background
        - Heat_night is shown in the background
        - the combined day/night curve is shown in black
        - the automatically detected transition times are marked

    Each day is processed independently so that every day
    gets its own morning and evening transition.
    """

    PLOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CHECK REQUIRED COLUMNS
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

    # PREPARE DATA
    res = res.copy()

    res["timestamp"] = pd.to_datetime(
        res["timestamp"]
    )

    res = res.sort_values(
        "timestamp"
    )

    res["date"] = res[
        "timestamp"
    ].dt.date

    # FIGURE
    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    # PROCESS EACH DAY
    for date, day_data in res.groupby(
        "date"
    ):

        day_data = day_data.copy()

        # FIND TRANSITIONS FOR THIS DAY
        morning_transition, evening_transition = (
            get_transition_times(
                day_data,
                date,
            )
        )

        # Skip day if transition could not be determined
        if (
            morning_transition is None
            or evening_transition is None
        ):
            print(
                f"WARNING: Could not determine transitions for "
                f"{pd.Timestamp(date).strftime('%d.%m.%Y')}"
            )
            continue

        # CREATE COMBINED CURVE
        day_data["Heat_combined"] = (
            create_combined_heat_flux(
                day_data,
                morning_transition,
                evening_transition,
            )
        )

        # BACKGROUND CURVES
        ax.plot(
            day_data["timestamp"],
            day_data["Heat_day"],
            color="tab:red",
            linewidth=0.8,
            alpha=0.20,
        )

        ax.plot(
            day_data["timestamp"],
            day_data["Heat_night"],
            color="tab:blue",
            linewidth=0.8,
            alpha=0.35,
        )

        # COMBINED CURVE
        ax.plot(
            day_data["timestamp"],
            day_data["Heat_combined"],
            color="black",
            linewidth=1.0,
            zorder=5,
        )

        # TRANSITION LINES
        ax.axvline(
            morning_transition,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.25,
        )

        ax.axvline(
            evening_transition,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.25,
        )

        # TRANSITION POINT VALUES
        morning_row = day_data.iloc[
            (
                day_data["timestamp"]
                - morning_transition
            ).abs().argsort()[:1]
        ]

        evening_row = day_data.iloc[
            (
                day_data["timestamp"]
                - evening_transition
            ).abs().argsort()[:1]
        ]

        morning_value_day = morning_row[
            "Heat_day"
        ].iloc[0]

        morning_value_night = morning_row[
            "Heat_night"
        ].iloc[0]

        morning_value = (
            morning_value_day
            + morning_value_night
        ) / 2

        evening_value_day = evening_row[
            "Heat_day"
        ].iloc[0]

        evening_value_night = evening_row[
            "Heat_night"
        ].iloc[0]

        evening_value = (
            evening_value_day
            + evening_value_night
        ) / 2

        #ax.scatter(
        #    morning_transition,
        #    morning_value,
        #    color="black",
        #    s=25,
        #    zorder=10,
        #)

        #ax.scatter(
        #    evening_transition,
        #    evening_value,
        #    color="black",
        #    s=25,
        #    zorder=10,
        #)

    # LABELS
    ax.set_title(
        "SLS20 – combined sensible heat flux diurnal cycles"
    )

    ax.set_ylabel(
        "Sensible heat flux [$W/m^2$]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    format_time_axis(
        ax
    )
    # Only show 00:00 and 12:00
    ax.xaxis.set_major_locator(
        mdates.HourLocator(
            byhour=[0, 12],
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%H:%M",
        )
    )

    # DATE LABELS
    dates = sorted(
        res["timestamp"].dt.date.unique()
    )

    add_day_labels(
        ax,
        dates,
    )

    # LEGEND
    ax.plot(
        [],
        [],
        label="Heat day (background)",
        color="tab:red",
        linewidth=1.0,
        alpha=0.25,
    )

    ax.plot(
        [],
        [],
        label="Heat night (background)",
        color="tab:blue",
        linewidth=1.0,
        alpha=0.5,
    )

    ax.plot(
        [],
        [],
        label="combined diurnal cycle",
        color="black",
        linewidth=1.0,
    )

    ax.legend()

    fig.subplots_adjust(
        bottom=0.25
    )

    # SAVE
    output = (
        PLOT_FOLDER
        / "SLS20_sensible_heat_flux_combined_all_days.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved combined all-days plot: {output}"
    )


# EXISTING PLOT
def plot_heat_flux(res: pd.DataFrame) -> None:

    PLOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CHECK REQUIRED COLUMNS
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

    # PREPARE DATA
    res = res.copy()

    res["timestamp"] = pd.to_datetime(
        res["timestamp"]
    )

    res = res.sort_values(
        "timestamp"
    )

    # ALL DAYS
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

    ax.set_ylabel(
        "Sensible heat flux [$W/m^2$]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    # X-AXIS: ONLY TIME
    format_time_axis(ax)

    # X-AXIS FOR MULTI-DAY PLOT:
    # Only show 00:00 and 12:00
    ax.xaxis.set_major_locator(
        mdates.HourLocator(
            byhour=[0, 12],
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%H:%M",
        )
    )

    # DATE LABELS: ONE DATE PER DAY, CENTERED
    dates = sorted(
        res["timestamp"].dt.date.unique()
    )

    add_day_labels(
        ax,
        dates,
    )

    fig.subplots_adjust(
        bottom=0.20
    )

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

    # INDIVIDUAL DAYS
    res["date"] = res[
        "timestamp"
    ].dt.date

    for date, day_data in res.groupby(
        "date"
    ):

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

        date_string = pd.Timestamp(
            date
        ).strftime(
            "%d.%m.%Y"
        )

        ax.set_title(
            f"{date_string} – SLS20 – Sensible Heat Flux"
        )

        ax.set_ylabel(
            "Sensible heat flux [$W/m^2$]"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        format_time_axis(
            ax
        )

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


# MAIN
def main():

    print(
        "SLS20 SENSIBLE HEAT FLUX PLOTS"
    )

    print(
        f"Data folder:  {DATA_FOLDER}"
    )

    print(
        f"Plot folder:  {PLOT_FOLDER}"
    )

    print()

    print(
        "Loading SLS20 data..."
    )

    res, dgn = load_sls20_data(
        DATA_FOLDER
    )
    res = prepare_sls20_timestamps(
        res
    )

    print()

    print(
        f"DGN rows: {len(dgn)}"
    )

    print(
        f"RES rows: {len(res)}"
    )

    print()

    print(
        "RES columns:"
    )

    print(
        list(res.columns)
    )

    # EXISTING PLOTS
    plot_heat_flux(
        res
    )

    # NEW COMBINED DAY/NIGHT PLOTS
    print()
    print(
        "Creating combined day/night plots..."
    )

    plot_combined_daily_cycle(
        res
    )

    # One plot containing all days
    plot_combined_all_days(
        res
    )

    print()
    print(
        "Done."
    )

    # SAVE BLACK COMBINED CURVE AS CSV
    print()
    print(
        "Saving combined day/night curve to CSV..."
    )

    save_combined_heat_flux_csv(
        res
    )


if __name__ == "__main__":
    main()