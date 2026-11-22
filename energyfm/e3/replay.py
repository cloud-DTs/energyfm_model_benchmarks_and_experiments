"""Reproduce the multi-profile E3 trigger, local update and retrospective acceptance gate."""

import csv
import json
import math
import statistics
from pathlib import Path

import torch

from energyfm.learning.data import load_vea_profiles, make_daily_windows
from energyfm.learning.files import (
    create_run_directory,
    experiment_provenance,
    hardware_details,
    save_candidate,
    select_device,
    write_json,
)
from energyfm.learning.metrics import validation_mae
from energyfm.learning.models import (
    load_student,
    load_teacher,
    predict_student,
    predict_teacher,
)
from energyfm.learning.settings import CONTEXT, HORIZON, SEEDS, LocalUpdateSettings
from energyfm.learning.training import train_local_update

MONITOR_START = 305
MONITOR_DAYS = 30
TRIGGER_WINDOW = 5
REQUIRED_FAILURES = 3
REFERENCE_DAYS = 30
TRAINING_DAYS = 90
SHADOW_DAYS = 30
MINIMUM_GAIN_PERCENT = 0.1
VEA_DAYS = 366


def add_arguments(parser):
    """Add model, profile limit, seed, device and planning options to the parser.

    Inputs: The replay subparser receives the model, profile limit, seed list, device
        and plan mode.
    Outputs: None. Registers options so the launcher can validate the replay
        configuration.
    Purpose: Support the same retrospective protocol for Toto and Chronos 2 Small.
    """
    parser.add_argument("--max-profiles", type=int, default=10)
    parser.add_argument("--model", choices=["toto", "chronos-small"], default="toto")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--smoke", action="store_true", help="Screen only VEA profile 101."
    )
    modes.add_argument(
        "--schedule", type=Path, help="CSV with profile_id and update_day columns."
    )
    parser.add_argument("--plan", action="store_true")


def find_triggers(model_name, model, monitored):
    """Find profiles whose forecasts repeatedly lose to previous-day persistence.

    Inputs: The model name selects the inference path. The unadapted model and
        chronological VEA measurements provide forecasts, actuals and the previous-day
        baseline.
    Outputs: The first trigger per qualifying profile, in profile order, with both the
        observation day and next update day.
    Purpose: Choose when to attempt a local update using only the failures observed by
        the trigger day.

    Notes: A day counts as a miss when the forecast median's MAE is strictly higher than
        the previous day's persistence MAE. A profile triggers after at least three
        misses in a complete five-day window.
    """
    # Profile IDs are one-based positions in the loaded archive order.
    failures = []
    triggers = [None] * len(monitored)
    for day in range(MONITOR_START, MONITOR_START + MONITOR_DAYS):
        start = day * HORIZON
        histories = monitored[:, start - CONTEXT : start]
        actual = monitored[:, start : start + HORIZON]
        previous_day = monitored[:, start - HORIZON : start]
        forecasts = predict_student(model_name, model, histories)
        # Forecasts use the common quantile order, with the median at index 4.
        model_error = (forecasts[4] - actual).abs().mean(dim=1)
        persistence_error = (previous_day - actual).abs().mean(dim=1)
        failures.append(model_error > persistence_error)
        # Wait until all five daily comparisons are available.
        if len(failures) < TRIGGER_WINDOW:
            continue
        failure_counts = torch.stack(failures[-TRIGGER_WINDOW:]).sum(dim=0)
        for index, failure_count in enumerate(failure_counts):
            if triggers[index] is None and int(failure_count) >= REQUIRED_FAILURES:
                triggers[index] = {
                    "profile_id": index + 101,
                    "observed_day": day,
                    # The forecast error becomes available after this day ends.
                    "update_day": day + 1,
                    "failure_count": int(failure_count),
                }
    return [trigger for trigger in triggers if trigger is not None]


def acceptance_decision(
    base_reference, candidate_reference, base_shadow, candidate_shadow
):
    """Apply the replay's shadow-improvement and reference-safety gates.

    Inputs: The two reference MAEs check for regression on older dates. The two shadow
        MAEs measure relative improvement on dates after the trigger.
    Outputs: An acceptance flag and percentage shadow gain, for applying and reporting
        both acceptance conditions.
    Purpose: Require a useful shadow improvement and protect the reference score before
        marking a candidate accepted.

    Notes: The candidate must improve shadow MAE by at least 0.1 percent and have
        reference MAE no higher than the base. A zero base shadow MAE gives zero
        relative gain.
    """
    # Zero base error leaves no room for relative improvement.
    gain = 0.0
    if base_shadow:
        gain = 100.0 * (base_shadow - candidate_shadow) / base_shadow
    accepted = gain >= MINIMUM_GAIN_PERCENT and candidate_reference <= base_reference
    return accepted, gain


def summarize_updates(rows):
    """Summarize candidate acceptance separately for each update scope.

    Inputs: Candidate rows carry scope, profile, seed, acceptance and shadow gain.
    Outputs: Per-scope candidate and acceptance counts, distinct accepted profile IDs
        and mean shadow gain.
    Purpose: Keep repeated seeds visible without counting each seed as a separate
        profile.
    """
    summary = {}
    for scope in ["head", "full"]:
        candidates = [row for row in rows if row["scope"] == scope]
        gains = [row["shadow_improvement_percent"] for row in candidates]
        accepted_count = sum(row["accepted"] for row in candidates)
        accepted_profiles = set()
        for row in candidates:
            if row["accepted"]:
                accepted_profiles.add(row["profile_id"])
        summary[scope] = {
            "candidates": len(candidates),
            "accepted": accepted_count,
            "profiles_with_any_acceptance": sorted(accepted_profiles),
            "mean_shadow_improvement_percent": (
                statistics.mean(gains) if gains else None
            ),
        }
    return summary


def evaluate_profile(model_name, base, teacher, profile, trigger, seeds, output):
    """Fit and score local updates for one triggered profile.

    Inputs: A base student and teacher provide forecasts. The measured profile, trigger
        day, seeds and output path specify local data, repetitions and artifact storage.
    Outputs: Score records for each scope and seed. Checkpoints are written for review,
        including when a candidate fails.
    Purpose: Compare head-only and full updates starting from the same base and using
        distinct training and evaluation target periods.
    """
    update_day = trigger["update_day"]
    training_start = update_day - TRAINING_DAYS
    reference_start = training_start - REFERENCE_DAYS
    profile_id = trigger["profile_id"]
    # The window builder expects a batch, so add a one-profile batch dimension.
    values = profile.unsqueeze(0)
    # These half-open target ranges are disjoint. A day's input history may
    # cross a boundary because it contains only measurements before that day.
    training = make_daily_windows(values, training_start, update_day, profile_id)
    reference = make_daily_windows(values, reference_start, training_start, profile_id)
    shadow = make_daily_windows(
        values, update_day, update_day + SHADOW_DAYS, profile_id
    )
    # The teacher supplies local pseudo-labels only for the adaptation window.
    targets = predict_teacher(teacher, training.histories, batch_size=8)
    base_reference = validation_mae(model_name, base, reference)
    base_shadow = validation_mae(model_name, base, shadow)
    rows = []
    for scope in ["head", "full"]:
        settings = LocalUpdateSettings(scope)
        for seed in seeds:
            candidate, training_log = train_local_update(
                base, training, targets, settings, seed, model_name=model_name
            )
            candidate_reference = validation_mae(model_name, candidate, reference)
            candidate_shadow = validation_mae(model_name, candidate, shadow)
            accepted, gain = acceptance_decision(
                base_reference, candidate_reference, base_shadow, candidate_shadow
            )
            row = {
                "profile_id": profile_id,
                "update_day": update_day,
                "scope": scope,
                "seed": seed,
                "base_reference_mae": base_reference,
                "candidate_reference_mae": candidate_reference,
                "base_shadow_mae": base_shadow,
                "candidate_shadow_mae": candidate_shadow,
                "accepted": accepted,
                "shadow_improvement_percent": gain,
                "training": training_log,
            }
            # Keep the checkpoint for review; this replay never activates it.
            name = f"profile{profile_id}_day{update_day}_{scope}_seed{seed}.pt"
            save_candidate(
                output / "checkpoints" / name,
                candidate,
                model_name,
                settings,
                seed,
                training_log,
            )
            rows.append(row)
            del candidate
    return rows


def load_schedule(path):
    """Read a fixed adaptation schedule to replay both models on the same cases."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"profile_id", "update_day"} <= set(
            reader.fieldnames
        ):
            raise ValueError("The schedule requires profile_id and update_day columns.")
        earliest = REFERENCE_DAYS + TRAINING_DAYS + math.ceil(CONTEXT / HORIZON)
        latest = VEA_DAYS - SHADOW_DAYS
        entries = []
        seen = set()
        for row in reader:
            try:
                profile = int(row["profile_id"])
                day = int(row["update_day"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid replay schedule row {reader.line_num}."
                ) from error
            key = (profile, day)
            if (
                not 101 <= profile <= 200
                or not earliest <= day <= latest
                or key in seen
            ):
                raise ValueError(f"Invalid or duplicate replay schedule entry: {key}")
            seen.add(key)
            entries.append({"profile_id": profile, "update_day": day})
    if not entries:
        raise ValueError("The replay schedule must contain at least one case.")
    return entries


def run(args):
    """Print the replay plan or run the retrospective update evaluation.

    Inputs: Parsed model, profile limit, seed, device and plan options define the replay
        workload.
    Outputs: None. Prints the plan or saves selected triggers, candidate checkpoints and
        acceptance summaries.
    Purpose: Coordinate screening, local training and retrospective evaluation while
        keeping the base model as the comparison point.
    """
    args.seeds = list(dict.fromkeys(args.seeds))
    if args.max_profiles < 1:
        raise ValueError("--max-profiles must be positive.")
    schedule = load_schedule(args.schedule) if args.schedule is not None else None
    last_profile = 101 if args.smoke else 200
    plan = {
        "model": args.model,
        "profile_limit": args.max_profiles,
        "screened_profiles": [101, last_profile] if schedule is None else None,
        "schedule": schedule,
        "seeds": args.seeds,
        "scopes": ["head", "full"],
        "training_steps": 10,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "teacher_weight": 0.5,
        "reference_training_shadow_days": [30, 90, 30],
        "minimum_gain_percent": MINIMUM_GAIN_PERCENT,
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    device = select_device(args.device)
    profiles = load_vea_profiles(last_profile)
    base = load_student(args.model, device)
    # Screen all monitored profiles first. The limit selects candidates in
    # profile order, independent of when or how often they missed the baseline.
    if schedule is None:
        triggers = find_triggers(args.model, base, profiles[100:last_profile])
    else:
        triggers = schedule
    triggers = triggers[: args.max_profiles]
    output = create_run_directory("E3", f"replay_{args.model.replace('-', '_')}")
    result = {
        "model": args.model,
        "plan": plan,
        "hardware": hardware_details(device),
        "provenance": experiment_provenance("vea", [args.model, "teacher"]),
        "triggers": triggers,
        "candidates": [],
        "summary": summarize_updates([]),
        "limitations": "Replay only: no active model changes, no labelled drift events, no independent post-gate test.",
    }
    teacher = load_teacher(device) if triggers else None
    for trigger in triggers:
        profile = profiles[trigger["profile_id"] - 1]
        rows = evaluate_profile(
            args.model, base, teacher, profile, trigger, args.seeds, output
        )
        result["candidates"].extend(rows)
        result["summary"] = summarize_updates(result["candidates"])
        write_json(output / "results.json", result)
    write_json(output / "results.json", result)
    print(f"Replay completed: {output}", flush=True)
