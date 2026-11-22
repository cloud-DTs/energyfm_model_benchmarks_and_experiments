"""Compare four unadapted forecasting models on the same load windows."""

import json

import numpy as np
import torch

from energyfm.assets import MODELS
from energyfm.e1.chronicle_forecast import ChronicleForecaster
from energyfm.learning.data import load_dataset, subset_windows
from energyfm.learning.files import (
    create_run_directory,
    experiment_provenance,
    hardware_details,
    save_tensor,
    select_device,
    write_json,
)
from energyfm.learning.models import (
    load_student,
    load_teacher,
    predict_student,
    predict_teacher,
)
from energyfm.learning.settings import CONTEXT, HORIZON

MODEL_NAMES = ("chronos", "chronicle", "toto", "chronos-small")


def add_arguments(parser):
    """Register dataset, model and execution options for the zero-shot benchmark."""
    parser.add_argument(
        "--datasets", nargs="+", choices=["vea", "uci"], default=["vea", "uci"]
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument(
        "--limit", type=int, help="Use only the first N test windows per dataset."
    )
    parser.add_argument("--plan", action="store_true")


def load_model(name, device):
    """Load the pinned base weights for one of the four forecast paths."""
    if name == "chronicle":
        return ChronicleForecaster(device)
    if name == "chronos":
        return load_teacher(device)
    return load_student(name, device)


def predict_median(name, model, histories):
    """Return the point forecast for each history in the original window order."""
    if name == "chronicle":
        tokens = model.encode("")
        batch_size = 16 if model.device == "cuda" else 2
        batches = []
        for batch in histories.split(batch_size):
            prediction = model.predict_batch(
                batch.numpy(), HORIZON, [tokens] * len(batch)
            )
            batches.append(prediction)
        return torch.from_numpy(np.concatenate(batches, axis=0))
    if name == "chronos":
        return predict_teacher(model, histories)[4]
    return predict_student(name, model, histories)[4]


def metrics(actual, predicted):
    """Compute MAE and RMSE on the same quarter-hour targets."""
    if predicted.shape != actual.shape or not torch.isfinite(predicted).all():
        raise ValueError(
            "Forecasts must match the test shape and contain only finite values."
        )
    errors = (actual - predicted).to(torch.float64)
    return {
        "mae_kw": float(errors.abs().mean()),
        "rmse_kw": float(errors.square().mean().sqrt()),
    }


def run(args):
    """Run the zero-shot comparison on matched test windows.

    Inputs: Dataset and model selections, device and optional case limit.
    Outputs: Forecasts, measured values and per-dataset metrics.
    Purpose: Compare the four base models against each other and previous-day persistence.
    """
    datasets = list(dict.fromkeys(args.datasets))
    models = list(dict.fromkeys(args.models))
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    configuration = {
        "datasets": datasets,
        "models": models,
        "history_steps": CONTEXT,
        "forecast_steps": HORIZON,
        "case_limit": args.limit,
    }
    if args.plan:
        print(json.dumps(configuration, indent=2))
        return

    device = select_device(args.device)
    torch.set_num_threads(4)
    output = create_run_directory("benchmark", "zero_shot")
    test_sets = {}
    results = {}
    for dataset in datasets:
        split = load_dataset(dataset)
        test = subset_windows(split.test, args.limit)
        test_sets[dataset] = test
        persistence = test.histories[:, -HORIZON:]
        results[dataset] = {
            "cases": len(test.targets),
            "dataset_version": experiment_provenance(dataset, [])["dataset_version"],
            "scores": {"previous_day": metrics(test.targets, persistence)},
        }
        save_tensor(
            output / dataset / "cases.pt",
            {
                "actual": test.targets,
                "profile_ids": test.profile_ids,
                "target_days": test.target_days,
                "previous_day": persistence,
            },
        )

    for name in models:
        model = load_model(name, device)
        for dataset, test in test_sets.items():
            print(
                f"Forecasting {len(test.targets)} {dataset.upper()} windows with {name}",
                flush=True,
            )
            forecasts = predict_median(name, model, test.histories)
            results[dataset]["scores"][name] = metrics(test.targets, forecasts)
            save_tensor(output / dataset / f"{name}.pt", forecasts)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    result = {
        "configuration": configuration,
        "hardware": hardware_details(device),
        "model_revisions": {name: MODELS[name]["revision"] for name in models},
        "results": results,
    }
    write_json(output / "results.json", result)
    print(output)
