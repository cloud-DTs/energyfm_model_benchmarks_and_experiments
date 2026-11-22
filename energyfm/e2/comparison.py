"""Compare forecasting students under historical or matched training settings."""

import math
import statistics
from dataclasses import asdict, replace

import torch

from energyfm.learning.cache import load_teacher_targets
from energyfm.learning.data import ForecastWindows, load_dataset, subset_windows
from energyfm.learning.files import (
    create_run_directory,
    experiment_provenance,
    hardware_details,
    save_candidate,
    save_predictions,
    save_tensor,
    select_device,
    write_json,
)
from energyfm.learning.metrics import forecast_metrics
from energyfm.learning.models import (
    load_student,
    load_teacher,
    predict_student,
    predict_teacher,
)
from energyfm.learning.settings import SEEDS, comparison_settings
from energyfm.learning.training import train_candidate

OBJECTIVES = ["supervised", "distilled", "hybrid"]
FAIR_LEARNING_RATES = [3e-5, 1e-4]
QUICK_TRAINING_WINDOWS = 3200
SMOKE_WINDOWS = 8


def add_arguments(parser):
    """Add comparison options to the command line parser.

    Inputs: The comparison subparser receives experiment protocol, dataset, models,
        objectives, tuning budget and execution options.
    Outputs: None. Registers the options in place so the top-level parser can dispatch
        the comparison.
    Purpose: Expose the historical comparison, matched budget and short pilot from one
        command.
    """
    parser.add_argument("--protocol", choices=["historical", "fair"], default="fair")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["toto", "chronos-small"],
        default=["toto", "chronos-small"],
    )
    parser.add_argument("--dataset", choices=["vea", "uci"], default="vea")
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=OBJECTIVES
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Default: 42 for fair/quick; 42/43/44 for historical.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Fair comparison rates (default: 3e-5 1e-4; quick: 1e-4).",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick",
        action="store_true",
        help="Pilot comparison: one rate, one seed by default, one epoch, 3,200 training windows.",
    )
    modes.add_argument(
        "--smoke",
        action="store_true",
        help="Run one short epoch on eight cases per split.",
    )
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show budgets without loading data or models.",
    )


def candidate_seeds(args):
    """Return explicit seeds or the full/quick preset's default seeds.

    Inputs: Parsed options provide an explicit seed list or the protocol and quick
        settings used to choose defaults.
    Outputs: The resolved seed list for both workload calculation and candidate
        training.
    Purpose: Keep planned and executed repetitions consistent.

    Notes: The default is seed 42 in fair and quick mode, or 42, 43 and 44 in a full
        historical run.
    """
    if args.seeds is not None:
        return args.seeds
    return SEEDS[:1] if args.quick or args.smoke or args.protocol == "fair" else SEEDS


def candidate_rates(args):
    """Resolve positive fair search rates, preserving historical model rates.

    Inputs: Parsed options supply the protocol, optional user rates and pilot flag.
    Outputs: Positive unique rates for the fair search, or a None sentinel that lets the
        historical protocol use architecture defaults.
    Purpose: Resolve and validate rates once for both planning and execution.
    """
    if args.protocol == "historical":
        if args.learning_rates is not None:
            raise ValueError("--learning-rates requires --protocol fair.")
        return [None]
    if args.learning_rates is not None:
        if any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
            raise ValueError("Learning rates must be finite and positive.")
        return list(dict.fromkeys(args.learning_rates))
    return [1e-4] if args.quick or args.smoke else FAIR_LEARNING_RATES


def candidate_settings(args, model_name, rate):
    """Apply the pilot epoch budget to the selected protocol's training policy.

    Inputs: The protocol and quick flag specify the budget. Model name selects
        historical defaults, and rate sets the fair trial's learning rate.
    Outputs: A training settings record for the planned and executed candidate.
    Purpose: Centralize the one-epoch pilot override without changing the underlying
        protocol.
    """
    settings = comparison_settings(args.protocol, model_name, rate)
    if args.quick or args.smoke:
        return replace(settings, epochs=1)
    return settings


def quick_training_subset(split, teacher_targets):
    """Select shared reproducible training cases and aligned cached teacher targets.

    Inputs: The validated split and aligned teacher quantiles define the available
        cases. A fixed internal seed controls the subset.
    Outputs: A split with reduced training cases, the matching teacher targets and
        original indices for auditing. Validation and test stay complete.
    Purpose: Give every quick candidate the same training windows while preserving the
        original teacher alignment.
    """
    generator = torch.Generator().manual_seed(0)
    count = min(QUICK_TRAINING_WINDOWS, len(split.training.histories))
    indices = torch.randperm(len(split.training.histories), generator=generator)[:count]
    indices = indices.sort().values
    training = ForecastWindows(
        split.training.histories[indices],
        split.training.targets[indices],
        split.training.profile_ids[indices],
        split.training.target_days[indices],
    )
    # Teacher targets put the case axis second, after quantile. Use the same
    # sorted indices as the observed windows to preserve case correspondence.
    return replace(split, training=training), teacher_targets[:, indices], indices


def build_plan(args):
    """Describe the requested workload without loading data or models.

    Inputs: Parsed arguments define the protocol, device-independent budget and
        candidate combinations.
    Outputs: A list of settings, objective and seed choices, case counts and update
        counts so the full cost is visible before execution.
    Purpose: Describe work without loading datasets or models and reject unsupported
        historical dataset combinations.
    """
    if args.protocol == "historical" and args.dataset != "vea":
        raise ValueError(
            "The historical objective comparison uses VEA. Use fair for a new UCI study."
        )
    mode = "full"
    training_windows = 24000
    if args.smoke:
        mode = "smoke"
        training_windows = SMOKE_WINDOWS
    elif args.quick:
        mode = "quick"
        training_windows = QUICK_TRAINING_WINDOWS
    plan = []
    for model_name in args.models:
        rates = candidate_rates(args)
        for rate in rates:
            settings = candidate_settings(args, model_name, rate)
            seeds = candidate_seeds(args)
            plan.append(
                {
                    "model": model_name,
                    "settings": asdict(settings),
                    "mode": mode,
                    "training_windows": training_windows,
                    "seeds": seeds,
                    "objectives": args.objectives,
                    "updates_per_candidate": settings.epochs
                    * math.ceil(training_windows / settings.batch_size),
                    "candidate_count": len(seeds) * len(args.objectives),
                }
            )
    return plan


def select_learning_rate(trials):
    """Select one learning rate using validation scores only.

    Inputs: Trial records for one model and objective provide validation MAE, learning
        rate and seed evidence.
    Outputs: The rate with the lowest mean best-validation MAE across seeds, for
        selecting test checkpoints.
    Purpose: Keep test scores out of hyperparameter selection.
    """
    grouped = {}
    for trial in trials:
        rate = trial["settings"]["learning_rate"]
        # Pool seeds at each rate so selection uses one rate for the whole objective.
        grouped.setdefault(rate, []).append(trial["training"]["best_validation_mae"])
    averages = {}
    for rate, values in grouped.items():
        averages[rate] = statistics.mean(values)
    return min(averages, key=averages.get)


def fit_trials(
    base, model_name, objective, split, teacher_targets, args, output, result
):
    """Fit and save every rate/seed candidate for one model and objective.

    Inputs: The base model, model name, objective, aligned split and teacher targets
        define training. CLI options define the rates and seeds. The output directory
        and result record collect evidence.
    Outputs: The fitted trial records with settings, validation history and checkpoint
        paths for rate selection.
    Purpose: Keep evidence for every attempted rate and seed, including rates that will
        not be scored on test.
    """
    rates = candidate_rates(args)
    trials = []
    for rate in rates:
        settings = candidate_settings(args, model_name, rate)
        for seed in candidate_seeds(args):
            key = f"{model_name}_{objective}_lr{settings.learning_rate}_seed{seed}"
            print(f"Training {key}", flush=True)
            model, training = train_candidate(
                base,
                model_name,
                split.training,
                split.validation,
                teacher_targets,
                objective,
                seed,
                settings,
            )
            checkpoint = output / "checkpoints" / f"{key}.pt"
            save_candidate(checkpoint, model, model_name, settings, seed, training)
            trial = {
                "key": key,
                "model": model_name,
                "objective": objective,
                "seed": seed,
                "settings": asdict(settings),
                "training": training,
                "checkpoint": str(checkpoint.relative_to(output)),
            }
            trials.append(trial)
            result["trials"].append(trial)
            write_json(output / "results.json", result)
            del model
    return trials


def score_selected_trials(base, model_name, objective, trials, split, output):
    """Score only checkpoints at the validation-selected learning rate.

    Inputs: The unmodified base, trial records, model and objective identify the
        selected checkpoints. The split and output location supply cases and a
        destination for test predictions.
    Outputs: One test score record per selected seed, with predictions also saved for
        inspection.
    Purpose: Use validation-selected weights and reconstruct every candidate from an
        independent copy of the same base.
    """
    chosen_rate = select_learning_rate(trials)
    selected = []
    for trial in trials:
        if trial["settings"]["learning_rate"] != chosen_rate:
            continue
        checkpoint = torch.load(
            output / trial["checkpoint"], map_location="cpu", weights_only=True
        )
        # Rebuild each candidate from the same base so frozen weights cannot
        # carry over from another checkpoint.
        import copy

        model = copy.deepcopy(base)
        # Checkpoints store directly trained parameters. With a partial update,
        # the frozen parameters are supplied by the copied base model.
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        forecasts = predict_student(model_name, model, split.test.histories)
        prediction_path = output / "predictions" / f"{trial['key']}.pt"
        save_predictions(prediction_path, split.test, forecasts)
        selected.append(
            {
                "model": model_name,
                "objective": objective,
                "seed": trial["seed"],
                "selected_learning_rate": chosen_rate,
                "metrics": forecast_metrics(split.test, forecasts),
            }
        )
        del model
    return selected


def summarize_scores(rows):
    """Summarize selected test metrics by model and objective.

    Inputs: Per-seed test results identify the model, training objective and MAE for
        each observation.
    Outputs: Groups with mean MAE, population standard deviation and individual seed
        scores for comparison.
    Purpose: Summarize accuracy and variation while retaining the individual
        observations.
    """
    groups = {}
    for row in rows:
        key = f"{row['model']}/{row['objective']}"
        groups.setdefault(key, []).append(row["metrics"]["mae"])
    summary = {}
    for name, values in groups.items():
        summary[name] = {
            "mean_mae": statistics.mean(values),
            "std_population": statistics.pstdev(values),
            "seed_mae": values,
        }
    return summary


def run(args):
    """Print the trial plan or run the requested model comparison.

    Inputs: Parsed options specify the dataset, models, objectives, protocol and
        optional plan or quick mode.
    Outputs: None. Prints a workload plan or writes provenance, checkpoint trials, test
        forecasts and summary scores.
    Purpose: Coordinate the E2 objective comparison from data and cache checks to
        validation selection and test scoring.
    """
    args.models = list(dict.fromkeys(args.models))
    args.objectives = list(dict.fromkeys(args.objectives))
    if args.seeds is not None:
        args.seeds = list(dict.fromkeys(args.seeds))
    plan = build_plan(args)
    if args.plan:
        import json

        print(json.dumps(plan, indent=2))
        return
    device = select_device(args.device)
    print(f"Device: {device}; loading {args.dataset.upper()} windows.", flush=True)
    split = load_dataset(args.dataset)
    mode = plan[0]["mode"]
    if args.smoke:
        split = replace(
            split,
            training=subset_windows(split.training, SMOKE_WINDOWS),
            validation=subset_windows(split.validation, SMOKE_WINDOWS),
            test=subset_windows(split.test, SMOKE_WINDOWS),
        )
        teacher = load_teacher(device)
        teacher_targets = predict_teacher(teacher, split.training.histories)
        teacher_test = predict_teacher(teacher, split.test.histories)
        del teacher
    else:
        teacher_targets, teacher_test = load_teacher_targets(args.dataset, split)
    mode_suffix = "" if mode == "full" else f"_{mode}"
    output = create_run_directory(
        "E2", f"comparison_{args.protocol}_{args.dataset}{mode_suffix}"
    )
    if args.quick:
        split, teacher_targets, indices = quick_training_subset(split, teacher_targets)
        save_tensor(output / "training_indices.pt", indices)
        print(
            f"Quick pilot: {len(split.training.histories):,} training windows, "
            f"one epoch, {len(candidate_rates(args))} learning rate(s). "
            "Validation and test remain complete.",
            flush=True,
        )
    candidates = sum(item["candidate_count"] for item in plan)
    updates = sum(
        item["candidate_count"] * item["updates_per_candidate"] for item in plan
    )
    print(
        f"Budget: {candidates} candidates, {updates:,} optimizer updates total.",
        flush=True,
    )
    result = {
        "protocol": args.protocol,
        "mode": mode,
        "dataset": split.metadata,
        "plan": plan,
        "hardware": hardware_details(device),
        "provenance": experiment_provenance(args.dataset, args.models + ["teacher"]),
        "teacher": forecast_metrics(split.test, teacher_test),
        "bases": {},
        "trials": [],
        "selected_test_scores": [],
        "limitations": [
            "The VEA test has already been evaluated; it is not a new untouched holdout.",
            "Fair means equal examples, updates, optimizer settings and learning-rate search budget.",
            "Native architectures, normalization, parameter counts and compute per update differ.",
            "No measured energy saving or universal architecture ranking is inferred.",
        ],
    }
    if len(candidate_seeds(args)) == 1:
        result["limitations"].append(
            "One training seed: scores do not measure variability across random starts."
        )
    if args.smoke:
        result["limitations"].append("Smoke run on eight windows per split.")
    if args.quick:
        result["limitations"].append(
            "Quick pilot: reduced training data and epochs, one learning rate by default, "
            "and one seed by default. Not a full-budget study or a measure of seed variability. "
            "Original training indices are stored in training_indices.pt (sampling seed 0)."
        )
    print(f"Output: {output}", flush=True)
    for model_name in args.models:
        base = load_student(model_name, device)
        forecasts = predict_student(model_name, base, split.test.histories)
        result["bases"][model_name] = forecast_metrics(split.test, forecasts)
        save_predictions(
            output / "predictions" / f"{model_name}_base.pt", split.test, forecasts
        )
        for objective in args.objectives:
            trials = fit_trials(
                base,
                model_name,
                objective,
                split,
                teacher_targets,
                args,
                output,
                result,
            )
            scores = score_selected_trials(
                base, model_name, objective, trials, split, output
            )
            result["selected_test_scores"].extend(scores)
            result["summary"] = summarize_scores(result["selected_test_scores"])
            write_json(output / "results.json", result)
        del base
    print(f"Completed comparison: {output}", flush=True)
