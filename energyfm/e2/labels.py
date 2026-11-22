"""Measure point pseudo-label performance at several observed-label budgets."""

import json
from dataclasses import asdict

from energyfm.learning.cache import load_teacher_targets
from energyfm.learning.data import load_dataset, make_label_masks
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
from energyfm.learning.models import load_student, predict_student
from energyfm.learning.settings import SEEDS, TrainingSettings
from energyfm.learning.training import train_candidate

FRACTIONS = [1.0, 0.25, 0.1, 0.01, 0.0]


def add_arguments(parser):
    """Add model, dataset, seed, device and planning options to the command parser.

    Inputs: The label-study subparser receives dataset and student choices, seeds,
        device and plan mode.
    Outputs: None. Registers options in place for the top-level command.
    Purpose: Make it possible to repeat the same label-budget protocol for either
        student on either dataset.
    """
    parser.add_argument(
        "--datasets", nargs="+", choices=["vea", "uci"], default=["vea", "uci"]
    )
    parser.add_argument(
        "--models", nargs="+", choices=["toto", "chronos-small"], default=["toto"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--plan", action="store_true")


def target_jobs(fraction):
    """Choose the distinct training conditions for one measured-label fraction.

    Inputs: The measured fraction determines which training conditions are distinct and
        feasible.
    Outputs: Job names for routing and, at intermediate budgets, the measured-only
        control.
    Purpose: Skip a duplicate control at full coverage and an infeasible measured-only
        run at zero coverage.
    """
    if 0 < fraction < 1:
        return ["routed", "supervised-only"]
    return ["routed"]


def run_dataset(dataset, model_name, args, device):
    """Fit and evaluate one student's label-budget candidates on one dataset.

    Inputs: The dataset identifies cases and teacher medians. The model name selects the
        student, CLI options provide seeds, and the device controls placement.
    Outputs: None. Writes masks, checkpoints, predictions and metrics for auditing each
        budget and control.
    Purpose: Measure the effect of replacing missing measured training targets while
        retaining the same validation and test cases.
    """
    split = load_dataset(dataset)
    teacher_points, teacher_test = load_teacher_targets(dataset, split, "point")
    settings = TrainingSettings(dropout_mode="all", gradient_clip=None)
    output = create_run_directory(
        "E2", f"point_labels_{model_name.replace('-', '_')}_{dataset}"
    )
    base = load_student(model_name, device)
    base_forecasts = predict_student(model_name, base, split.test.histories)
    result = {
        "dataset": split.metadata,
        "settings": asdict(settings),
        "hardware": hardware_details(device),
        "model": model_name,
        "provenance": experiment_provenance(dataset, [model_name, "teacher"]),
        "teacher": forecast_metrics(split.test, teacher_test),
        "base": forecast_metrics(split.test, base_forecasts),
        "runs": [],
        "limitations": "Historical inputs, validation labels and test labels remain observed. Fractions mask training targets only.",
    }
    # Masks affect future training labels only. Every variant keeps the same
    # observed histories and the same measured validation and test targets.
    for seed in args.seeds:
        masks = make_label_masks(split.training.profile_ids, FRACTIONS, seed)
        save_tensor(output / f"masks_seed{seed}.pt", masks)
        for fraction in FRACTIONS:
            mask = masks[fraction]
            # Replace unobserved future targets with teacher medians, case by case.
            routed = teacher_points.clone()
            routed[mask] = split.training.targets[mask]
            for objective in target_jobs(fraction):
                key = f"{objective}_{fraction:g}_seed{seed}"
                print(f"Training {key}", flush=True)
                control_mask = None
                targets = routed
                if objective == "supervised-only":
                    # Repeating only labelled cases matches the routed run's update count.
                    control_mask = mask
                    targets = split.training.targets
                model, training = train_candidate(
                    base,
                    model_name,
                    split.training,
                    split.validation,
                    None,
                    "supervised",
                    seed,
                    settings,
                    measured_mask=control_mask,
                    point_targets=targets,
                )
                forecasts = predict_student(model_name, model, split.test.histories)
                save_candidate(
                    output / "checkpoints" / f"{key}.pt",
                    model,
                    model_name,
                    settings,
                    seed,
                    training,
                )
                save_predictions(
                    output / "predictions" / f"{key}.pt", split.test, forecasts
                )
                result["runs"].append(
                    {
                        "fraction": fraction,
                        "seed": seed,
                        "objective": objective,
                        "measured_windows": int(mask.sum()),
                        "training": training,
                        "metrics": forecast_metrics(split.test, forecasts),
                    }
                )
                write_json(output / "results.json", result)
                del model
    print(f"Completed label study: {output}", flush=True)


def run(args):
    """Print candidate counts or run each selected model and dataset combination.

    Inputs: Parsed dataset, student and seed choices specify the requested workload.
        Device and plan options control execution.
    Outputs: None. Prints candidate counts or writes one run directory per model and
        dataset combination.
    Purpose: Coordinate label-budget runs without loading data or weights in plan mode.
    """
    args.models = list(dict.fromkeys(args.models))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.seeds = list(dict.fromkeys(args.seeds))
    if args.plan:
        plan = {
            "datasets": args.datasets,
            "models": args.models,
            "seeds": args.seeds,
            "fractions": FRACTIONS,
            "candidates_per_model_and_dataset": 8 * len(args.seeds),
            "updates_per_candidate": 6000,
            "epochs": 4,
            "batch_size": 16,
        }
        print(json.dumps(plan, indent=2))
        return
    device = select_device(args.device)
    for model_name in args.models:
        for dataset in args.datasets:
            run_dataset(dataset, model_name, args, device)
