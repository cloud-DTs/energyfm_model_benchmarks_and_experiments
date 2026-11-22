"""Compute forecast accuracy metrics from paired predictions and observations."""

import torch


def forecast_metrics(windows, forecasts):
    """Calculate accuracy for paired forecast cases.

    Inputs: The forecast windows supply actuals and profile IDs. Quantiles are ordered
        [quantile, case, horizon] so index 4 selects the median.
    Outputs: Overall MAE and RMSE, case count and per-profile MAEs for aggregate and
        profile-level evaluation.
    Purpose: Use the same point forecast and case alignment throughout E2.
    """
    error = forecasts[4] - windows.targets
    profile_mae = {}
    for profile_id in torch.unique(windows.profile_ids):
        selected = windows.profile_ids == profile_id
        profile_mae[str(int(profile_id))] = float(error[selected].abs().mean())
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "forecast_count": len(windows.targets),
        "per_profile_mae": profile_mae,
    }


def validation_mae(model_name, model, windows, batch_size=16):
    """Compute the mean absolute error of median forecasts on supplied windows.

    Inputs: The model name selects inference, the loaded model provides forecasts and
        the windows provide histories and measurements. Batch size limits inference
        memory.
    Outputs: A scalar MAE of the median forecast in the dataset's units for direct
        comparison.
    Purpose: Reuse one scoring rule for E2 checkpoint selection and E3 reference and
        shadow evaluation.
    """
    from energyfm.learning.models import predict_student

    forecasts = predict_student(model_name, model, windows.histories, batch_size)
    return float((forecasts[4] - windows.targets).abs().mean())
