"""Experiment 02: does forecast wind improve Penmanshiel site-power forecasts?

Compare 168 hours of past site power with and without archived 100 m wind
forecasts, predicting 24 hours ahead for complete Q4 2024 days. No training
is performed. read_power() assumes ten-minute interval-start timestamps;
encode_wind() represents the direction as sine and cosine.
"""

import csv
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from energyfm.assets import dataset_archive
from energyfm.e1.inputs import prepared_path, weather_path

ONE_HOUR = timedelta(hours=1)


def get_weather():
    """Read the saved Penmanshiel weather forecast used by the report.

    Inputs: No explicit arguments. The configured cache path identifies the archived
        Penmanshiel forecast.
    Outputs: A decoded weather mapping with hourly timestamps, wind speed and direction
        for covariate construction.
    Purpose: Give the two E1 forecast backends access to the same archived forecast.
    """
    return json.loads(
        weather_path("penmanshiel/forecast.json").read_text(encoding="utf-8")
    )


def read_power():
    """Read site-meter measurements and average each complete six-reading hour.

    Inputs: No explicit arguments. The configured meter ZIP supplies the original
        10-minute power readings.
    Outputs: An hour-end to mean-power mapping in kW, with incomplete or conflicting
        hours excluded from downstream cases.
    Purpose: Apply one measurement and timestamp convention before pairing forecast
        inputs with their targets.
    """
    # Store hour -> {minute: power}, so each hour can be checked for all six readings.
    grouped = defaultdict(dict)
    with ZipFile(dataset_archive("penmanshiel")) as archive:
        # Choose the measurement file by its Device_Data_ name, rather than a supporting
        # file in the ZIP.
        name = None
        for archive_name in archive.namelist():
            if archive_name.startswith("Device_Data_"):
                name = archive_name
                break
        if name is None:
            raise FileNotFoundError("No Device_Data_ measurement file in the archive.")
        with archive.open(name) as binfile:
            rows = csv.reader(
                io.TextIOWrapper(binfile, encoding="utf-8-sig", newline="")
            )
            header = None
            for row in rows:
                if not row:
                    continue
                # Find the actual column header after the introductory comment lines.
                if row[0] == "# Date and time":
                    header = row
                    timestamp_column = 0
                    # Select the site-level power column by its name, not by a hard-
                    # coded column number.
                    power_column = header.index("GMS Power (kW)")
                    continue
                if header is None or row[0].startswith("#"):
                    continue
                try:
                    timestamp = datetime.strptime(
                        row[timestamp_column], "%Y-%m-%d %H:%M:%S"
                    )
                    power_text = row[power_column].strip()
                    # A blank or NaN entry is missing power, not zero power.
                    if power_text and power_text.lower() != "nan":
                        value = float(power_text)
                    else:
                        value = float("nan")
                except (ValueError, IndexError):
                    continue
                # Fixed working assumption: each UTC mark starts its ten-minute interval.
                # Samples 12:00..12:50 form the hour ending at 13:00.
                hour_end = (
                    timestamp.replace(minute=0, second=0, microsecond=0) + ONE_HOUR
                )
                # If a duplicate is invalid or disagrees, invalidate that minute; do not
                # choose an arbitrary value.
                if timestamp.minute in grouped[hour_end]:
                    previous_value = grouped[hour_end][timestamp.minute]
                    if not (
                        math.isfinite(previous_value)
                        and math.isfinite(value)
                        and math.isclose(
                            previous_value, value, rel_tol=1e-6, abs_tol=1e-6
                        )
                    ):
                        grouped[hour_end][timestamp.minute] = float("nan")
                else:
                    grouped[hour_end][timestamp.minute] = value
    # Keep only hours with six valid readings, then calculate their mean power.
    series = {}
    expected_minutes = {0, 10, 20, 30, 40, 50}
    for hour_end, minute_readings in grouped.items():
        if set(minute_readings) != expected_minutes:
            continue
        valid_hour = True
        for value in minute_readings.values():
            if not math.isfinite(value):
                valid_hour = False
                break
        if valid_hour:
            series[hour_end] = sum(minute_readings.values()) / 6
    print("Complete hourly site-power readings", len(series), flush=True)
    return series


def encode_wind(weather_by_time, timestamps):
    """Build float32 wind inputs in the same timestamp and feature order.

    Inputs: A timestamp lookup supplies weather speed and direction. The ordered
        timestamps choose the exact case window.
    Outputs: A float32 array with columns for speed, sine and cosine, in the requested
        hour order.
    Purpose: Share the wind representation across variants and avoid a discontinuity at
        the north direction.
    """
    rows = []
    for timestamp in timestamps:
        speed, direction = weather_by_time[timestamp]
        radians = math.radians(direction)
        direction_sine = math.sin(radians)
        direction_cosine = math.cos(radians)
        rows.append([speed, direction_sine, direction_cosine])
    return np.asarray(rows, dtype=np.float32)


def load_wind_lookup():
    """Index archived wind forecasts by their valid hour.

    Inputs: No explicit arguments. The local weather archive supplies timestamps, speed
        and direction.
    Outputs: A timestamp-to-wind mapping for building both models’ weather inputs.
    Purpose: Use the same hour convention and missing-weather policy for both models.
    """
    weather = get_weather()["hourly"]
    # Build a timestamp -> (wind speed, wind direction) lookup for matching weather to
    # power.
    weather_by_time = {}
    weather_rows = zip(
        weather["time"],
        weather["wind_speed_100m_previous_day2"],
        weather["wind_direction_100m_previous_day2"],
        strict=True,
    )
    for timestamp, speed, direction in weather_rows:
        if speed is None or direction is None:
            continue
        hour = datetime.fromisoformat(timestamp)
        weather_by_time[hour] = (speed, direction)
    return weather_by_time


def prepare_inputs():
    """Create the hourly site-power table from the original meter archive."""
    output = prepared_path("penmanshiel/power.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Prepared site power already exists: {output}")
    values = pd.Series(read_power(), name="power_kw", dtype=float).sort_index()
    values.index.name = "hour_end_utc"
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        values.to_csv(temporary)
        output.hardlink_to(temporary)
    print(output)
