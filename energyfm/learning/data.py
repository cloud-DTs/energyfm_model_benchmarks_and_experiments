"""Load forecasting datasets and build aligned daily input/target windows."""

import csv
import io
import math
import zipfile
from dataclasses import dataclass

import pandas as pd
import torch

from energyfm.assets import dataset_archive
from energyfm.learning.settings import CONTEXT, HORIZON


@dataclass
class ForecastWindows:
    """Store aligned inputs, measured targets, and case identifiers.

    Fields: histories contain past measurements; targets contain future measured
    values; profile_ids and target_days identify each case. This record keeps
    those four tensors in the same case order.
    """

    histories: torch.Tensor
    targets: torch.Tensor
    profile_ids: torch.Tensor
    target_days: torch.Tensor


@dataclass
class DatasetSplit:
    """Group the data used for fitting, validation, and final scoring.

    Fields: training, validation, and test are ForecastWindows records;
    metadata describes the source and partition rules. Keeping roles explicit
    prevents training code from accidentally receiving test windows.
    """

    training: ForecastWindows
    validation: ForecastWindows
    test: ForecastWindows
    metadata: dict


def require_archive(dataset):
    """Locate the requested load dataset or explain how to obtain it."""
    path = dataset_archive(dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run 'energyfm fetch data {dataset}'."
        )
    return path


def subset_windows(windows, limit):
    """Retain the same first cases across histories, targets and identifiers."""
    return ForecastWindows(
        windows.histories[:limit],
        windows.targets[:limit],
        windows.profile_ids[:limit],
        windows.target_days[:limit],
    )


def load_vea_profiles(profile_count=200):
    """Read complete VEA load profiles from the local archive.

    Inputs: The requested count determines how many complete rows to take from the
        configured VEA archive.
    Outputs: A float32 tensor [profile, time] preserving eligible source-row order for
        stable profile numbering.
    Purpose: Give all VEA experiments the same eligibility rule and measurement layout.
    """
    profiles = []
    with zipfile.ZipFile(require_archive("vea")) as archive:
        with archive.open("load_profiles_tabsep.csv") as binary_source:
            source = io.TextIOWrapper(binary_source, encoding="utf-8-sig")
            rows = csv.reader(source, delimiter="\t")
            next(rows, None)
            for row in rows:
                try:
                    values = []
                    for value in row[1:]:
                        if value:
                            values.append(float(value))
                except ValueError:
                    continue
                if len(values) == 35136:
                    profiles.append(values)
                if len(profiles) == profile_count:
                    break
    if len(profiles) != profile_count:
        raise ValueError(f"Expected {profile_count} complete VEA profiles.")
    return torch.tensor(profiles, dtype=torch.float32)


def make_daily_windows(profiles, first_day, stop_day, first_profile_id=1):
    """Create daily VEA history/forecast pairs for a half-open day range.

    Inputs: The profile tensor supplies consecutive measurements. An inclusive first day
        and exclusive stop day define target dates, and the first profile ID labels the
        rows.
    Outputs: ForecastWindows in day-then-profile order so histories, measured targets
        and identifiers stay aligned.
    Purpose: Apply a common 512-step context and 96-step target horizon to complete
        days.
    """
    histories = []
    targets = []
    owners = []
    days = []
    first_day = max(first_day, math.ceil(CONTEXT / HORIZON))
    stop_day = min(stop_day, profiles.shape[1] // HORIZON)
    for day in range(first_day, stop_day):
        start = day * HORIZON
        histories.append(profiles[:, start - CONTEXT : start])
        targets.append(profiles[:, start : start + HORIZON])
        owners.append(torch.arange(first_profile_id, first_profile_id + len(profiles)))
        days.append(torch.full((len(profiles),), day, dtype=torch.long))
    return ForecastWindows(
        torch.cat(histories), torch.cat(targets), torch.cat(owners), torch.cat(days)
    )


def load_vea_split():
    """Build the fixed VEA training, validation, and main test partitions.

    Inputs: No explicit arguments. The configured VEA archive and fixed profile and date
        ranges supply the cases.
    Outputs: A DatasetSplit with separate profiles for training, validation and test so
        evaluation can reproduce the recorded split.
    Purpose: Centralize the VEA profile holdout, whose test dates overlap the training
        calendar.

    Notes: Training uses profiles 1–80, validation 81–100 and test 112–200.
    """
    profiles = load_vea_profiles()
    training = make_daily_windows(profiles[:80], 6, 306)
    validation = make_daily_windows(profiles[80:100], 306, 336, 81)
    test = make_daily_windows(profiles[111:200], 240, 270, 112)
    metadata = {
        "dataset": "VEA 2016",
        "test": "main evaluated test",
        "split": "Different training, validation and test profiles; half-open target days.",
    }
    return DatasetSplit(training, validation, test, metadata)


def read_uci_frame():
    """Read UCI client measurements and label each interval by its start time.

    Inputs: No explicit arguments. The configured local ZIP provides UCI client
        observations and their interval-end timestamps.
    Outputs: A numeric client table indexed by interval start so windows refer to the
        actual quarter-hour intervals.
    Purpose: Apply the timestamp and numeric parsing rules before any UCI case
        selection.
    """
    with zipfile.ZipFile(require_archive("uci")) as archive:
        for name in archive.namelist():
            if name.endswith("LD2011_2014.txt") and not name.startswith("__MACOSX"):
                with archive.open(name) as source:
                    frame = pd.read_csv(
                        source, sep=";", decimal=",", index_col=0, parse_dates=True
                    )
                break
    # Source marks interval ends. Shifting labels does not modify the readings.
    frame.index = frame.index - pd.Timedelta(minutes=15)
    return frame


def make_uci_windows(frame, clients, start, stop, first_profile_id, limit=None):
    """Create fixed-length daily histories and targets for selected UCI clients.

    Inputs: The client table and chosen columns define candidate series. Date bounds and
        optional day limit choose the cases, and the starting profile ID labels them.
    Outputs: Fixed-size ForecastWindows in day-then-client order, with daylight-saving
        transition windows excluded.
    Purpose: Keep UCI input and target lengths comparable across clients and dates.

    Notes: The start date is inclusive and the stop date is exclusive. The optional
        limit counts retained days.
    """
    transitions = pd.to_datetime(
        [
            "2011-03-27",
            "2011-10-30",
            "2012-03-25",
            "2012-10-28",
            "2013-03-31",
            "2013-10-27",
            "2014-03-30",
            "2014-10-26",
        ]
    )
    histories, targets, owners, target_days = [], [], [], []
    for day in pd.date_range(start, stop, freq="D", inclusive="left"):
        history_start = day - pd.Timedelta(minutes=15 * CONTEXT)
        target_stop = day + pd.Timedelta(days=1)
        # Exclude daylight-saving transition days from the full history and target span.
        crosses_transition = False
        for transition in transitions:
            if (
                history_start <= transition + pd.Timedelta(days=1)
                and target_stop > transition
            ):
                crosses_transition = True
                break
        if crosses_transition:
            continue
        history_stop = day - pd.Timedelta(minutes=15)
        target_end = target_stop - pd.Timedelta(minutes=15)
        history = (
            frame.loc[history_start:history_stop, clients]
            .to_numpy(dtype="float32")
            .T.copy()
        )
        target = frame.loc[day:target_end, clients].to_numpy(dtype="float32").T.copy()
        histories.append(torch.from_numpy(history))
        targets.append(torch.from_numpy(target))
        owners.append(torch.arange(first_profile_id, first_profile_id + len(clients)))
        day_index = (day - pd.Timestamp("2011-01-01")).days
        target_days.append(torch.full((len(clients),), day_index, dtype=torch.long))
        if limit is not None and len(histories) == limit:
            break
    return ForecastWindows(
        torch.cat(histories),
        torch.cat(targets),
        torch.cat(owners),
        torch.cat(target_days),
    )


def load_uci_split():
    """Build fixed UCI client partitions for training, validation, and testing.

    Inputs: No explicit arguments. The UCI archive, historical eligibility filter and
        fixed date ranges determine the client partitions.
    Outputs: A DatasetSplit with disjoint clients and filtering metadata for
        reproduction.
    Purpose: Apply one consistent UCI protocol to experiments and teacher-cache
        preparation.
    """
    frame = read_uci_frame()
    active_before_2012 = frame.loc[:"2011-12-31"].gt(0).any()
    mostly_active = frame.loc["2012-01-01":"2012-11-15"].gt(0).mean() > 0.95
    clients = list(frame.columns[active_before_2012 & mostly_active])[:200]
    training = make_uci_windows(frame, clients[:80], "2012-01-07", "2012-12-01", 1, 300)
    validation = make_uci_windows(
        frame, clients[80:100], "2012-12-01", "2012-12-31", 81
    )
    test = make_uci_windows(frame, clients[100:200], "2013-01-01", "2013-01-31", 101)
    metadata = {
        "dataset": "UCI ElectricityLoadDiagrams20112014",
        "client_columns": clients,
        "day_origin": "2011-01-01",
        "timestamps": "Interval-end labels shifted back 15 minutes; DST windows excluded.",
        "selection": "Positive before 2012 and >95% positive in Jan 1 to Nov 15 2012.",
    }
    return DatasetSplit(training, validation, test, metadata)


def load_dataset(dataset):
    """Load one supported dataset and its fixed partitions.

    Inputs: The key 'vea' or 'uci' selects the corresponding archive reader and split
        protocol.
    Outputs: A DatasetSplit in the common format used by training, evaluation and cache
        code.
    Purpose: Keep dataset selection in one place while preserving each dataset's timing
        rules.
    """
    if dataset == "vea":
        return load_vea_split()
    if dataset == "uci":
        return load_uci_split()
    raise ValueError(f"Unknown dataset: {dataset}")


def make_label_masks(profile_ids, fractions, seed):
    """Create reproducible nested masks for measured training targets.

    Inputs: Profile IDs define groups, fractions specify label coverage and the seed
        fixes the random order within each profile.
    Outputs: One boolean training-case mask per fraction, with smaller selections nested
        inside larger selections.
    Purpose: Vary measured target availability while keeping profile coverage and
        underlying historical inputs comparable.
    """
    generator = torch.Generator().manual_seed(seed)
    # Reuse one random ordering per profile at all fractions. Taking prefixes
    # makes smaller measured-label budgets true subsets of larger ones.
    ordered_indices = []
    for profile_id in torch.unique(profile_ids):
        indices = (profile_ids == profile_id).nonzero().flatten()
        permutation = torch.randperm(len(indices), generator=generator)
        ordered_indices.append(indices[permutation])
    masks = {}
    for fraction in fractions:
        mask = torch.zeros(len(profile_ids), dtype=torch.bool)
        for indices in ordered_indices:
            selected_count = round(len(indices) * fraction)
            mask[indices[:selected_count]] = True
        masks[fraction] = mask
    return masks
