"""Locate the versioned E1 inputs and define the portable case identifiers."""

from dataclasses import dataclass

from energyfm.paths import data_root


@dataclass(frozen=True)
class Experiment:
    title: str
    unit: str
    history: int
    horizon: int
    block_size: int
    chronos_batch_size: int
    clip: bool = False
    bootstrap_seed: int = 23


EXPERIMENTS = {
    "01": Experiment("Hill of Towie wind", "kW", 168, 24, 7, 64),
    "02": Experiment("Penmanshiel wind", "kW", 168, 24, 7, 8, bootstrap_seed=20261111),
    "03": Experiment("Energinet DK1 solar", "kWh", 24, 24, 168, 32, clip=True),
    "08": Experiment("Midtjylland solar, five weather sites", "kWh", 24, 24, 24, 32),
    "08-single": Experiment(
        "Midtjylland solar, one weather site", "kWh", 24, 24, 24, 64
    ),
    "09": Experiment("East England solar", "MW", 168, 39, 7, 32, clip=True),
}

DK1_SITES = {
    "north_jutland": (57.05, 9.92),
    "west_jutland": (56.14, 8.98),
    "central_jutland": (56.20, 9.50),
    "south_jutland": (55.25, 9.30),
    "funen": (55.40, 10.38),
}
MIDTJYLLAND_SITES = {
    "center": (56.20, 9.50),
    "west": (56.14, 8.98),
    "east": (56.16, 10.21),
    "north": (56.45, 9.40),
    "south": (55.86, 9.85),
}


def prepared_path(name):
    return data_root() / "e1" / "prepared" / name


def weather_path(name):
    return data_root() / "e1" / "weather" / name


def case_path(number):
    return data_root() / "e1" / "cases" / f"{number}.csv"


def bundle_files():
    """List the input files required by the published E1 protocol."""
    names = [
        "sources.json",
        "prepared/hill_of_towie/fleet.npz",
        "prepared/penmanshiel/power.csv",
        "prepared/dk1_solar_2025/target.csv",
        "prepared/midtjylland/solar_hourly.npz",
        "prepared/east_england/pv.csv",
        "weather/hill_of_towie/forecast.json",
        "weather/penmanshiel/forecast.json",
        "weather/midtjylland/single_site.json",
        "weather/east_england/gfs.csv",
    ]
    names.extend(f"weather/dk1_solar_2025/{site}.json" for site in DK1_SITES)
    names.extend(f"weather/midtjylland/{site}.json" for site in MIDTJYLLAND_SITES)
    names.extend(f"cases/{number}.csv" for number in EXPERIMENTS)
    return names


def load_keys(number):
    """Read unique forecast issue times and entity identifiers in their fixed order."""
    import pandas as pd

    path = case_path(number)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run 'energyfm fetch data e1'.")
    rows = pd.read_csv(path, dtype={"entity": str})
    keys = pd.to_datetime(rows.issue_time_utc, utc=True)
    rows["issue_time_utc"] = keys
    if rows.empty or rows.isna().any().any() or rows.duplicated().any():
        raise ValueError(f"Expected nonempty, unique forecast keys in {path}.")
    return list(rows.itertuples(index=False, name=None))
