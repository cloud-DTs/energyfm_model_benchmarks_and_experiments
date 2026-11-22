"""Compare history-only forecasts with forecasts that also receive archived weather."""

import json

import numpy as np
import pandas as pd
import torch

from energyfm.assets import ASSETS, MODELS
from energyfm.e1.cases import load_cases
from energyfm.e1.chronicle_forecast import (
    ChronicleForecaster,
    context_text,
    forecast_cases,
)
from energyfm.e1.common import (
    choose_device,
    load_chronos,
    paired_block_ci,
    predict_median,
)
from energyfm.e1.inputs import EXPERIMENTS
from energyfm.learning.files import create_run_directory, hardware_details, write_json


def add_arguments(parser):
    """Register the benchmark suite, models, device and optional case limit."""
    parser.add_argument(
        "--experiment", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS)
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["chronos", "chronicle"],
        default=["chronos", "chronicle"],
    )
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument(
        "--limit", type=int, help="Use only the first N cases per experiment."
    )
    parser.add_argument("--plan", action="store_true")


def chronos_predictions(model, cases, experiment):
    """Run Chronos with and without the same hourly weather channels supplied to Chronicle."""
    bare = []
    enriched = []
    for case in cases:
        bare.append({"target": case.history})
        enriched.append(
            {
                "target": case.history,
                "past_covariates": case.past,
                "future_covariates": case.future,
            }
        )
    predictions = []
    for requests in (bare, enriched):
        result = predict_median(
            model,
            requests,
            experiment.history,
            experiment.horizon,
            batch_size=experiment.chronos_batch_size,
        )
        predictions.append(result)
    return predictions


def chronicle_predictions(model, cases):
    """Run the same case windows with empty or weather-enriched text prompts."""
    plain_tokens = model.encode("")
    enriched_tokens = [model.encode(context_text(case)) for case in cases]
    history_batch_size = 16 if model.device == "cuda" else 1
    baseline, _ = forecast_cases(
        model,
        cases,
        [plain_tokens] * len(cases),
        history_batch_size,
        "Chronicle history only",
    )
    enriched, _ = forecast_cases(
        model, cases, enriched_tokens, 1, "Chronicle with weather"
    )
    return baseline, enriched


def score_predictions(number, model_name, cases, predictions, output):
    """Score and save paired forecasts for one E1 experiment.

    Inputs: The paired model outputs, fixed cases and experiment configuration.
    Outputs: Per-period score rows and a CSV with the hourly forecasts.
    Purpose: Weight forecast origins equally and apply a paired bootstrap interval.
    """
    experiment = EXPERIMENTS[number]
    baseline = predictions[0][:, -24:]
    enriched = predictions[1][:, -24:]
    if experiment.clip:
        baseline = np.maximum(baseline, 0)
        enriched = np.maximum(enriched, 0)
    if not np.isfinite(baseline).all() or not np.isfinite(enriched).all():
        raise ValueError("The model returned non-finite forecasts.")

    errors = []
    forecast_rows = []
    hours = range(experiment.horizon - 23, experiment.horizon + 1)
    for case, without, with_weather in zip(cases, baseline, enriched, strict=True):
        issue = case.issue_time
        target_year = (issue + pd.Timedelta("1h")).year
        if number == "08-single" and target_year not in (2025, 2026):
            continue
        errors.append(
            (
                issue,
                np.mean(np.abs(case.actual - without)),
                np.mean(np.abs(case.actual - with_weather)),
            )
        )
        for lead, actual, raw, enriched_value in zip(
            hours, case.actual, without, with_weather, strict=True
        ):
            forecast_rows.append(
                (issue, case.entity, lead, actual, raw, enriched_value)
            )

    columns = [
        "issue_time_utc",
        "entity",
        "lead_hour",
        "actual",
        "without_context",
        "with_context",
    ]
    pd.DataFrame(forecast_rows, columns=columns).to_csv(
        output / "predictions.csv", index=False
    )
    by_issue = pd.DataFrame(
        errors, columns=["issue_time_utc", "without_context", "with_context"]
    )
    by_issue = by_issue.groupby("issue_time_utc").mean()
    groups = {"all": by_issue}
    if number == "08-single":
        groups = {}
        target_years = (by_issue.index + pd.Timedelta("1h")).year
        for year in (2025, 2026):
            group = by_issue[target_years == year]
            if not group.empty:
                groups[str(year)] = group

    rows = []
    for period, errors in groups.items():
        if errors.empty:
            continue
        base_mae = float(errors.without_context.mean())
        enriched_mae = float(errors.with_context.mean())
        gain = None
        if base_mae > 0:
            gain = 100 * (1 - enriched_mae / base_mae)
        low, high = paired_block_ci(
            errors.without_context,
            errors.with_context,
            block_size=experiment.block_size,
            seed=experiment.bootstrap_seed,
        )
        interval = [low, high] if np.isfinite([low, high]).all() else None
        rows.append(
            {
                "experiment": number,
                "model": model_name,
                "period": period,
                "unit": experiment.unit,
                "origins": len(errors),
                "without_context_mae": base_mae,
                "with_context_mae": enriched_mae,
                "mae_reduction_percent": gain,
                "confidence_interval": interval,
            }
        )
    return rows


def run(args):
    """Run the selected E1 history-only and weather comparisons.

    Inputs: The selected experiment IDs, backends, device and optional case limit.
    Outputs: A run directory with hourly forecasts and paired metrics.
    Purpose: Measure the effect of the archived weather in each forecast model.
    """
    numbers = list(dict.fromkeys(args.experiment))
    models = list(dict.fromkeys(args.models))
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.plan:
        print(
            json.dumps(
                {"experiments": numbers, "models": models, "case_limit": args.limit},
                indent=2,
            )
        )
        return

    device = choose_device(args.device)
    torch.set_num_threads(4)
    cases = {number: load_cases(number, args.limit) for number in numbers}
    output = create_run_directory("e1", "context")
    rows = []
    for model_name in models:
        if model_name == "chronos":
            model = load_chronos(device)
        else:
            model = ChronicleForecaster(device)
        for number, experiment_cases in cases.items():
            print(
                f"E1 {number}: {model_name}, {len(experiment_cases)} cases", flush=True
            )
            if model_name == "chronos":
                forecasts = chronos_predictions(
                    model, experiment_cases, EXPERIMENTS[number]
                )
            else:
                forecasts = chronicle_predictions(model, experiment_cases)

            destination = output / number / model_name
            destination.mkdir(parents=True)
            rows.extend(
                score_predictions(
                    number, model_name, experiment_cases, forecasts, destination
                )
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_json(
        output / "results.json",
        {
            "hardware": hardware_details(device),
            "model_revisions": {name: MODELS[name]["revision"] for name in models},
            "case_limit": args.limit,
            "input_version": ASSETS["e1"]["version"],
            "metrics": rows,
        },
    )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    print(output)
