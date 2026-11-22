"""Prepare pinned DK1 solar targets from the local Energinet export."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from energyfm.e1.inputs import prepared_path
from energyfm.paths import data_root

SOLAR_COLUMNS = (
    "SolarPowerLt10kW_MWh",
    "SolarPowerGe10Lt40kW_MWh",
    "SolarPowerGe40kW_MWh",
)


def prepare_inputs():
    """Create the DK1 solar targets from the archived Energinet export.

    Inputs: No explicit arguments. The project configuration provides the Energinet
        export and destination. Module constants define the weather sites and fields.
    Outputs: None. Writes the target CSV.
    Purpose: Recreate the target independently of forecasts or network requests.
    """
    output = prepared_path("dk1_solar_2025/target.csv")
    if output.exists():
        raise FileExistsError(f"Prepared DK1 inputs already exist: {output}")
    raw_dir = data_root() / "raw" / "dk1_solar_2025"
    source = raw_dir / "Energinet Production and Consumption.csv"
    data = pd.read_csv(
        source, sep=";", decimal=",", usecols=["HourUTC", "PriceArea", *SOLAR_COLUMNS]
    )
    data = data[
        (data.PriceArea == "DK1")
        & (data.HourUTC >= "2024-12-25")
        & (data.HourUTC < "2026-01-04")
    ].copy()
    times = pd.to_datetime(data.HourUTC, utc=True)
    if times.duplicated().any():
        raise ValueError("Duplicate selected DK1 source timestamps")
    values = data[list(SOLAR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    # Require all size classes. Exclude the separate own-consumption estimate.
    target = pd.DataFrame(
        {
            "interval_start_utc": times,
            "interval_end_utc": times + pd.Timedelta(hours=1),
            "solar_grid_kwh": 1000 * values.sum(axis=1, min_count=len(SOLAR_COLUMNS)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        target.sort_values("interval_end_utc").to_csv(staged, index=False)
        output.hardlink_to(staged)
    print("Frozen inputs:", output, flush=True)
