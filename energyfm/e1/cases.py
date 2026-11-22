"""Build energy and weather windows for the fixed E1 forecast keys."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energyfm.e1.experiments import (
    dk1_solar,
    east_england,
    hill_of_towie,
    midtjylland,
    penmanshiel,
)
from energyfm.e1.inputs import EXPERIMENTS, load_keys, prepared_path


@dataclass(frozen=True)
class Case:
    issue_time: pd.Timestamp
    entity: str
    history: np.ndarray
    actual: np.ndarray
    past: dict[str, np.ndarray]
    future: dict[str, np.ndarray]
    horizon: int


def aligned_case(issue, entity, target, weather, experiment):
    """Build a forecast case from hour-end-indexed input series.

    Inputs: The UTC issue time, entity, energy series and aligned weather channels.
    Outputs: A complete history, 24 scored target hours and past/future covariates.
    Purpose: Give both models the same measurements, forecast leads and context.
    """
    hour = pd.Timedelta("1h")
    first_target = issue + hour
    first_history = first_target - experiment.history * hour
    span = pd.date_range(
        first_history, periods=experiment.history + experiment.horizon, freq="h"
    )
    values = target.reindex(span).to_numpy(dtype=np.float32)
    channels = {}
    for name, series in weather.items():
        channels[name] = series.reindex(span).to_numpy(dtype=np.float32)
    for array in [values, *channels.values()]:
        if not np.isfinite(array).all():
            raise ValueError(f"Incomplete input at {issue.isoformat()} for {entity}.")

    cut = experiment.history
    return Case(
        issue,
        entity,
        values[:cut],
        values[-24:],
        {name: values[:cut] for name, values in channels.items()},
        {name: values[cut:] for name, values in channels.items()},
        experiment.horizon,
    )


def array_series(grid, channels):
    """Index weather arrays by valid time."""
    return {name: pd.Series(values, index=grid) for name, values in channels.items()}


def hill_inputs():
    stations, times, power = hill_of_towie.load_fleet()
    target = {}
    for station, values in zip(stations, power, strict=True):
        target[station] = pd.Series(values, index=times)
    grid, _, weather = hill_of_towie.load_grid_weather()
    return target, array_series(grid, weather)


def penmanshiel_inputs():
    table = pd.read_csv(prepared_path("penmanshiel/power.csv"))
    index = pd.to_datetime(table.hour_end_utc, utc=True)
    power = pd.Series(table.power_kw.to_numpy(), index=index)
    lookup = penmanshiel.load_wind_lookup()
    timestamps = sorted(lookup)
    wind = penmanshiel.encode_wind(lookup, timestamps)
    names = ("wind_speed_100m", "wind_direction_sin", "wind_direction_cos")
    weather = dict(zip(names, wind.T, strict=True))
    grid = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    return {"site": power}, array_series(grid, weather)


def dk1_inputs():
    grid, solar, weather = dk1_solar.load_dk1_series()
    return {"DK1": pd.Series(solar, index=grid)}, array_series(grid, weather)


def midtjylland_inputs(single_site=False):
    if single_site:
        grid, solar, weather = midtjylland.load_single_site_inputs()
    else:
        grid, solar, weather, _ = midtjylland.load_five_site_inputs()
    return {"Midtjylland": pd.Series(solar, index=grid)}, array_series(grid, weather)


def east_england_cases(keys, experiment):
    """Use preceding-day GFS for history and current-issue GFS for the 39-hour rollout."""
    solar, weather, past_radiation = east_england.load_east_england_inputs()
    for issue, entity in keys:
        issue_local = issue.tz_localize(None)
        issue_day = issue_local.normalize()
        hour = pd.Timedelta("1h")
        hist = pd.date_range(
            issue_local - experiment.history * hour,
            periods=experiment.history,
            freq="h",
        )
        rollout = pd.date_range(issue_local, periods=experiment.horizon, freq="h")
        values = solar.reindex(hist).to_numpy(dtype=np.float32)
        actual = solar.reindex(rollout[-24:]).to_numpy(dtype=np.float32)
        past = past_radiation.reindex(hist).to_numpy(dtype=np.float32)
        future = weather.loc[issue_day].iloc[1:].to_numpy(dtype=np.float32)
        if future.size != experiment.horizon:
            raise ValueError(f"Unexpected GFS window length at {issue.isoformat()}.")
        for array in (values, actual, past, future):
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Incomplete East England input at {issue.isoformat()}."
                )
        yield Case(
            issue,
            entity,
            values,
            actual,
            {"radiation": past},
            {"radiation": future},
            experiment.horizon,
        )


def load_cases(number, limit=None):
    """Build the windows for one E1 experiment in the stored issue order.

    Inputs: An experiment ID and an optional case limit for short runs.
    Outputs: Complete cases built from the versioned keys and source series.
    Purpose: Keep both forecast variants on the same eligible windows.
    """
    keys = load_keys(number)
    if limit is not None:
        if number == "08-single":
            hour = pd.Timedelta("1h")
            keys = [key for key in keys if (key[0] + hour).year in (2025, 2026)]
        keys = keys[:limit]
    experiment = EXPERIMENTS[number]
    if number == "09":
        return list(east_england_cases(keys, experiment))
    if number == "01":
        target, weather = hill_inputs()
    elif number == "02":
        target, weather = penmanshiel_inputs()
    elif number == "03":
        target, weather = dk1_inputs()
    else:
        target, weather = midtjylland_inputs(single_site=number == "08-single")
    return [
        aligned_case(issue, entity, target[entity], weather, experiment)
        for issue, entity in keys
    ]
