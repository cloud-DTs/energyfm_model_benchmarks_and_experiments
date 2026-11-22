"""Experiment 08: compare two weather setups for Midtjylland solar production.

Both comparisons predict regional solar production from 24 hours of history:
1. Five locations, each with radiation and cloud forecasts: overall report row.
2. One location with radiation, cloud and temperature: 2025 and 2026 report rows.

Each comparison uses matched forecast cases and a 24-origin bootstrap block.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from energyfm.e1.inputs import MIDTJYLLAND_SITES, prepared_path, weather_path
from energyfm.paths import data_root


def hourly_means(data, timestamp, target, period_minutes=60):
    """Build hourly averages and discard hours with missing measurements.

    Inputs: The source table and column names identify timestamps and observations. The
        source interval specifies how many measurements make a complete hour.
    Outputs: An hourly pandas Series indexed by UTC interval end so forecasts and
        observations can use the same timestamps.
    Purpose: Enforce the same duplicate handling, interval convention and completeness
        rule wherever this aggregation is used.
    """
    # Convert unreadable timestamps and values to missing entries so they can be
    # dropped.
    times = pd.to_datetime(data[timestamp], utc=True, errors="coerce")
    values = pd.to_numeric(data[target], errors="coerce")
    frame = pd.DataFrame({"time": times, "value": values}).dropna()
    if frame.empty:
        raise ValueError("No valid timestamps or measurements")
    # For repeated timestamps, keep the last source row, as in the original experiment.
    frame = frame.drop_duplicates("time", keep="last").set_index("time").sort_index()
    # Label each average by the hour end: for example, 12:00 belongs to the hour ending
    # 13:00.
    # Only keep hours with the expected number of source measurements.
    grouped = frame["value"].resample("1h", label="right", closed="left")
    count = grouped.count()
    result = grouped.mean()[count == 60 // period_minutes]
    return result.dropna()


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
    """Read the Energinet export and freeze complete hourly Midtjylland solar values.

    Inputs: Resolved data paths identify the raw source and prepared destination.
    Outputs: The archive path, containing exact float values and source provenance.
    Purpose: Preserve the time and filtering rules without repeating CSV parsing.
    """
    path = data_root() / "raw/midtjylland/Energinet Regionale Stromerzeugung.csv"
    output = prepared_path("midtjylland/solar_hourly.npz")
    if output.exists():
        raise FileExistsError(f"Prepared solar target already exists: {output}")
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        usecols=["TimeUTC", "RegionName", "ProductionType3", "Production_kWh"],
    )
    # Select solar production in Midtjylland. Other regions and energy types are not
    # part of this target.
    df = df[(df.RegionName == "Region Midtjylland") & (df.ProductionType3 == "Sol")]
    series = hourly_means(df, "TimeUTC", "Production_kWh")
    # Keep the original timestamps and float values without CSV rounding.
    save_prepared_npz(
        output,
        hour_ends=series.index.to_numpy(dtype="datetime64[ns]"),
        kwh=series.to_numpy(),
    )
    print(output)
    return output


def load_midtjylland_solar():
    """Load the frozen hourly Midtjylland solar series.

    Inputs: Resolved settings locate the prepared regional solar archive.
    Outputs: Complete kWh values indexed by UTC hour end for weather alignment.
    Purpose: Share identical targets without reopening the raw export.
    """
    path = prepared_path("midtjylland/solar_hourly.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run 'energyfm fetch data e1'.")
    with np.load(path, allow_pickle=False) as data:
        index = pd.to_datetime(data["hour_ends"], utc=True)
        return pd.Series(data["kwh"], index=index, name="value")


def load_weather(site, cache_dir):
    """Read an existing forecast file for one of the five weather locations.

    Inputs: The site name and cache directory identify the archived JSON file to read.
    Outputs: The decoded forecast mapping, including returned grid coordinates for
        alignment and provenance.
    Purpose: Read each site through the same path in Chronos and Chronicle.
    """
    return json.loads((cache_dir / f"{site}.json").read_text(encoding="utf-8"))


def load_solar_grid():
    """Align the regional solar measurements to the experiment’s hourly grid.

    Inputs: Project settings locate the target export.
    Outputs: The UTC grid and aligned float32 measurements for selecting windows and
        scoring.
    Purpose: Use exactly the same target alignment in the five-site and single-site
        experiments.
    """
    solar = load_midtjylland_solar()
    time_index = pd.date_range("2023-12-20", "2026-09-08 23:00", freq="h", tz="UTC")
    return time_index, solar.reindex(time_index).to_numpy(dtype=np.float32)


def load_five_site_inputs():
    """Load solar power and align five archived weather sites to hour end.

    Inputs: Project settings locate the regional target and saved weather files.
    Outputs: The hourly grid, solar targets, weather channels and returned coordinates
        for both backends.
    Purpose: Share five-site weather preparation and reject duplicated source grid
        cells.
    """
    cache_dir = weather_path("midtjylland")
    time_index, target = load_solar_grid()
    weather = {}
    returned_coordinates = {}
    for site in MIDTJYLLAND_SITES:
        data = load_weather(site, cache_dir)
        returned_coordinates[site] = [data.get("latitude"), data.get("longitude")]
        weather_index = pd.to_datetime(data["hourly"]["time"], utc=True)
        # Radiation already uses hour-end timestamps. Another shift would match weather
        # to the wrong power hour.
        for feature in ("shortwave_radiation", "cloud_cover"):
            key = f"{site}_{feature}"
            saved_values = data["hourly"][f"{feature}_previous_day2"]
            numeric_values = pd.to_numeric(saved_values, errors="coerce")
            hourly_values = pd.Series(numeric_values, index=weather_index)
            aligned_values = hourly_values.reindex(time_index)
            weather[key] = aligned_values.to_numpy(dtype=np.float32)
        print(
            f"Loaded site {site}: grid point {returned_coordinates[site]}", flush=True
        )
    # Different requested locations can return the same weather grid cell.
    # Check the returned coordinates so the five inputs really come from different
    # cells.
    unique_grid_cells = set()
    for coordinates in returned_coordinates.values():
        unique_grid_cells.add(tuple(coordinates))
    if len(unique_grid_cells) != len(MIDTJYLLAND_SITES):
        raise ValueError(
            "At least two requested weather locations map to the same grid point."
        )

    return time_index, target, weather, returned_coordinates


WEATHER_FIELDS = ("shortwave_radiation", "cloud_cover", "temperature_2m")


def load_single_site_inputs():
    """Align the archived radiation, cloud and temperature forecasts to the solar grid."""
    time_index, solar = load_solar_grid()
    path = weather_path("midtjylland/single_site.json")
    weather_data = json.loads(path.read_text(encoding="utf-8"))
    weather_index = pd.to_datetime(weather_data["hourly"]["time"], utc=True)
    weather = {}
    for name in WEATHER_FIELDS:
        values = weather_data["hourly"][f"{name}_previous_day2"]
        hourly = pd.Series(pd.to_numeric(values, errors="coerce"), index=weather_index)
        weather[name] = hourly.reindex(time_index).to_numpy(dtype=np.float32)
    return time_index, solar, weather
