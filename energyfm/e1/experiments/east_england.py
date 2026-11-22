"""Experiment 09: does forecast radiation improve East England solar forecasts?

Use 168 hours of power history, comparing history alone against history plus
past and future archived radiation forecasts. Predict 39 hours from 09:00 and score only the
last 24 hours (the following calendar day). Apply the same nonnegative clipping
to both variants.
"""

from __future__ import annotations

import pandas as pd

from energyfm.e1.inputs import prepared_path, weather_path


def build_radiation_history(weather):
    """Align older next-day radiation forecasts with historical solar hours.

    Inputs: The archived forecast table carries issue dates and lead columns, which
        determine the historical valid times.
    Outputs: An hourly radiation series indexed by valid time so it can be aligned to
        measured power history.
    Purpose: Construct historical covariates from archived forecasts using the
        experiment's issue-time convention.

    Notes: The input table's columns 0 through 39 represent lead hours 9 through 48.
        History uses the previous day's forecasts at leads 25 through 48, indexed by the
        starts of their valid intervals.
    """
    expected_columns = []
    for column in range(40):
        expected_columns.append(str(column))
    if list(weather.columns) != expected_columns or weather.index.has_duplicates:
        raise ValueError(
            "Expected unique GFS dates and ordered lead-hour columns 0 to 39."
        )

    radiation_by_time = {}
    for forecast_day, forecast_row in weather.iterrows():
        # The last 24 values are lead hours 25 to 48. Each describes the hour
        # ending at that lead time, so lead 25 belongs to 00:00-01:00 next day.
        # Use that preceding-day forecast for history; never use a later issue
        # to fill earlier hours. Missing days remain missing rather than filled.
        next_day = forecast_day + pd.Timedelta(days=1)
        for hour, value in enumerate(forecast_row.iloc[-24:]):
            hour_start = next_day + pd.Timedelta(hours=hour)
            radiation_by_time[hour_start] = value
    return pd.Series(radiation_by_time, dtype=float).sort_index()


def load_east_england_inputs():
    """Load hourly solar power and archived radiation features.

    Inputs: No explicit arguments. Project settings locate the solar measurements and
        GFS forecast table.
    Outputs: Hourly solar power, the forecast table and historical radiation for slicing
        either model’s cases.
    Purpose: Apply the same half-hour interval conversion and historical radiation
        alignment to Chronos and Chronicle.
    """
    data_dir = prepared_path("east_england")
    solar_power = pd.read_csv(data_dir / "pv.csv", parse_dates=["datetime_gmt"])
    solar_power = solar_power.set_index("datetime_gmt")["generation_mw"]
    solar_power = solar_power.sort_index()
    # Shift each half-hour timestamp to its interval start before forming hourly
    # averages.
    solar_power.index = solar_power.index.tz_localize(None) - pd.Timedelta(
        30, unit="min"
    )
    hourly_groups = solar_power.resample("h")
    # An hour needs two half-hour readings. Keep incomplete hours missing instead of
    # filling them.
    solar_power = hourly_groups.mean().where(hourly_groups.count() == 2)
    weather = pd.read_csv(
        weather_path("east_england/gfs.csv"), index_col=0, parse_dates=True
    )
    radiation_history = build_radiation_history(weather)

    return solar_power, weather, radiation_history
