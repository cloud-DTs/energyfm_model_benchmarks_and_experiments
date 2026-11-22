"""Experiment 01: does forecast weather improve Hill of Towie wind forecasts?

Compare 168 hours of turbine power alone with the same history plus archived
wind forecasts, predicting the next 24 hours without fine-tuning Chronos-2.
Each origin is weighted equally so dates with more turbines do not dominate.
"""

import csv
import io
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from energyfm.assets import dataset_archive
from energyfm.e1.inputs import prepared_path, weather_path

TABLE = "tblSCTurGrid"


def load_weather() -> dict:
    """Read the saved Hill of Towie wind forecast instead of downloading new weather.

    Inputs: No explicit arguments. Project settings locate the archived weather
        forecast.
    Outputs: A decoded forecast mapping, including hourly wind fields, for constructing
        aligned covariates.
    Purpose: Reuse the archived forecast so the weather input is reproducible.
    """
    cache = weather_path("hill_of_towie/forecast.json")
    # Use the saved weather file so reruns do not silently receive different forecast
    # data.
    return json.loads(cache.read_text(encoding="utf-8"))


def make_features(grid: pd.DatetimeIndex, weather_response: dict) -> dict:
    """Prepare wind forecasts for every hour in the power-data grid.

    Inputs: The UTC hourly grid specifies target timestamps. The archived response
        supplies forecast wind speed and direction at those times.
    Outputs: A feature mapping with wind channels aligned to the requested grid, so they
        can be sliced alongside turbine power.
    Purpose: Use one weather encoding for Chronos and Chronicle and represent circular
        wind direction as sine and cosine.
    """
    hourly = weather_response["hourly"]
    # Label each weather value with its timestamp. reindex later matches hours by time;
    # gaps stay NaN.
    weather_index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True))
    # Use the weather timestamp as the label for each wind-speed value.
    speed = pd.Series(
        pd.to_numeric(hourly["wind_speed_10m_previous_day2"], errors="coerce"),
        index=weather_index,
    )
    direction = pd.Series(
        pd.to_numeric(hourly["wind_direction_10m_previous_day2"], errors="coerce"),
        index=weather_index,
    )
    # 359 degrees and 0 degrees are close directions. Sine and cosine preserve that
    # closeness.
    aligned_direction = direction.reindex(grid)
    direction_degrees = aligned_direction.to_numpy(dtype=float)
    direction_radians = np.deg2rad(direction_degrees)
    aligned_speed = speed.reindex(grid)
    speed_values = aligned_speed.to_numpy(dtype=float)
    weather = {
        # Match the wind values to the power timestamps, then return plain arrays for
        # the model.
        "wind_speed_10m_jma_forecast_48h_lead": speed_values,
        "wind_direction_10m_sin_jma_forecast_48h_lead": np.sin(direction_radians),
        "wind_direction_10m_cos_jma_forecast_48h_lead": np.cos(direction_radians),
    }
    return weather


def load_fleet():
    """Load the prepared hourly turbine measurements.

    Inputs: No explicit arguments. The fleet cache supplies the frozen hourly values.
    Outputs: Turbine IDs, UTC hour ends and a power matrix in kW for case selection.
    Purpose: Run forecasts from stable hourly inputs without rebuilding from SCADA.
    """
    path = prepared_path("hill_of_towie/fleet.npz")
    with np.load(path, allow_pickle=False) as data:
        stations = data["stations"].astype(str).tolist()
        times = pd.DatetimeIndex(pd.to_datetime(data["time"].astype(str), utc=True))
        powers = data["power"].astype(float)
    return stations, times, powers


def save_prepared_npz(path, **arrays):
    """Publish a prepared NumPy archive after a successful write.

    Inputs: The destination and named arrays specify the frozen dataset.
    Outputs: None. Publishes a complete archive or raises if it already exists.
    Purpose: Keep failed or concurrent preparation from replacing valid inputs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        np.savez_compressed(temporary, **arrays)
        # A hard link publishes the complete file without replacing an existing one.
        path.hardlink_to(temporary)


def prepare_inputs():
    """Create the hourly fleet cache from the archived SCADA rows.

    Inputs: No explicit arguments. The raw ZIP supplies ten-minute measurements.
    Outputs: None. Writes a new fleet cache and reports duplicate and conflict counts.
    Purpose: Apply the completeness and timestamp rules once before forecasting.
    """
    output = prepared_path("hill_of_towie/fleet.npz")
    if output.exists():
        raise FileExistsError(f"Prepared fleet already exists: {output}")

    # Each (turbine ID, hour) gets a dictionary of minute -> power readings.
    values = defaultdict(dict)
    duplicates = 0
    conflicts = 0
    archive = dataset_archive("hill-of-towie")
    for month in range(6, 13):
        filename = f"{TABLE}_2023_{month:02d}.csv"
        with ZipFile(archive) as zipped, zipped.open(filename) as stream:
            rows = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            header = next(rows)
            # Find column positions by name so a different CSV column order does not
            # break reading.
            timestamp_column = header.index("TimeStamp")
            station_column = header.index("StationId")
            power_column = header.index("wtc_ActPower_mean")
            for row in rows:
                try:
                    station = row[station_column]
                    # Source: Hill_of_Towie_turbine_metadata.csv. T01-T21 have IDs
                    # 2304510-2304530.
                    # The shared prefix 23045 selects this fleet. The rule checks a
                    # prefix, not an exact ID list.
                    if not station.startswith("23045"):
                        continue
                    timestamp = datetime.fromisoformat(row[timestamp_column])
                    value = float(row[power_column])
                    # The source timestamps mark interval ends. Readings at 12:10
                    # through 13:00 belong to 13:00.
                    # A reading already at 13:00 must stay in that hour.
                    hour_end = timestamp.replace(minute=0, second=0, microsecond=0)
                    if timestamp.minute:
                        hour_end += timedelta(hours=1)
                    minute = timestamp.minute
                    if minute not in (0, 10, 20, 30, 40, 50):
                        continue
                except (ValueError, IndexError):
                    continue
                # Get the minute readings for this turbine and hour; defaultdict creates
                # an empty dictionary if needed.
                samples = values[station, hour_end]
                # Keep matching duplicates. If two values disagree, invalidate that
                # minute instead of choosing one.
                if minute in samples:
                    duplicates += 1
                    if samples[minute] is None or not np.isclose(
                        samples[minute], value, rtol=1e-6, atol=1e-6
                    ):
                        samples[minute] = None
                        conflicts += 1
                else:
                    samples[minute] = value
        print(f"Read raw data: {filename}", flush=True)

    complete = defaultdict(dict)
    # Require all six ten-minute readings. A partial-hour average would cover a
    # different amount of time.
    expected = {0, 10, 20, 30, 40, 50}
    for (station, hour_end), minutes in values.items():
        if set(minutes) != expected:
            continue
        readings_are_valid = True
        for value in minutes.values():
            if value is None or not np.isfinite(value):
                readings_are_valid = False
                break
        if readings_are_valid:
            complete[station][hour_end] = sum(minutes.values()) / 6
    stations = sorted(complete)
    times = pd.date_range("2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z", freq="1h")[
        :-1
    ]
    # Rows are turbines; columns are hours. Start with NaN so missing hours remain
    # visible.
    power = np.full((len(stations), len(times)), np.nan, dtype=np.float32)
    # Map each hour to its column number so readings can be placed in the power table.
    positions = {}
    for hour_index, timestamp in enumerate(times.tz_convert(None)):
        naive_timestamp = timestamp.to_pydatetime().replace(tzinfo=None)
        positions[naive_timestamp] = hour_index
    for turbine_index, station in enumerate(stations):
        for timestamp, value in complete[station].items():
            if timestamp in positions:
                power[turbine_index, positions[timestamp]] = value
    save_prepared_npz(
        output,
        stations=np.asarray(stations),
        time=np.asarray(times.astype(str), dtype=str),
        power=power,
    )
    print(f"Prepared fleet: {output}", flush=True)
    print(
        f"Turbines: {len(stations)}; duplicate rows: {duplicates}; conflicts: {conflicts}",
        flush=True,
    )


def load_grid_weather():
    """Align archived wind forecasts to the experiment’s hourly UTC grid.

    Inputs: No explicit arguments. The archived file and fixed grid define covered
        hours.
    Outputs: The grid, source metadata and aligned weather arrays for both models.
    Purpose: Share the time grid without repeated timestamp conversion.
    """
    grid = pd.date_range("2023-05-25T00:00:00Z", "2024-01-02T00:00:00Z", freq="h")
    response = load_weather()
    return grid, response, make_features(grid, response)
