"""Experiment 03: retrospective DK1 solar comparison, all hourly issue times in 2025.

24h history and 24h horizon. Five DK1 weather proxies supply past/future GFS
radiation and cloud forecasts as separate channels. No model training or tuning.
"""

import json

import numpy as np
import pandas as pd

from energyfm.e1.inputs import DK1_SITES, prepared_path, weather_path

FIELDS = ("shortwave_radiation", "cloud_cover")


def load_dk1_series():
    """Read DK1 solar generation and archived weather on a common hourly grid."""
    data = pd.read_csv(prepared_path("dk1_solar_2025/target.csv"))
    times = pd.DatetimeIndex(pd.to_datetime(data.interval_end_utc, utc=True))
    if times.has_duplicates:
        raise ValueError("Duplicate target interval ends")
    grid = pd.date_range("2024-12-25", "2026-01-04", freq="h", tz="UTC")
    target = (
        pd.Series(data.solar_grid_kwh.to_numpy(float), index=times)
        .reindex(grid)
        .to_numpy(np.float32)
    )
    weather = {}
    cells = []
    for site in DK1_SITES:
        response = json.loads(
            weather_path(f"dk1_solar_2025/{site}.json").read_text(encoding="utf-8")
        )
        cells.append((response["latitude"], response["longitude"]))
        index = pd.DatetimeIndex(pd.to_datetime(response["hourly"]["time"], utc=True))
        if index.has_duplicates or response["utc_offset_seconds"] != 0:
            raise ValueError("Invalid weather time grid")
        for field in FIELDS:
            values = pd.to_numeric(
                response["hourly"][f"{field}_previous_day2"], errors="coerce"
            )
            weather[f"{site}_{field}"] = (
                pd.Series(np.asarray(values, float), index=index)
                .reindex(grid)
                .to_numpy(np.float32)
            )
    if len(set(cells)) != len(DK1_SITES):
        raise ValueError("Weather sites are not distinct grid cells")
    return grid, target, weather
