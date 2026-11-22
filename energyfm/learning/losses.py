"""Define the measured-target and teacher-target objectives in each model's native scale."""

import torch

from energyfm.learning.settings import HORIZON, QUANTILES


def pinball_loss(prediction, actual, location, scale, use_arcsinh=True):
    """Fit forecast quantiles to measured values or point pseudo-labels.

    Inputs: Student quantiles and measured or point pseudo-targets determine forecast
        errors. Location, scale and the transform flag put targets in the student's
        coordinates.
    Outputs: One differentiable scalar for supervised updates or the measured component
        of a hybrid update.
    Purpose: Use the quantile-specific asymmetric penalty for either point target
        source.
    """
    actual = actual.to(prediction.device)
    normalized = (actual - location) / scale
    if use_arcsinh:
        normalized = normalized.asinh()
    error = normalized.unsqueeze(0) - prediction
    levels = prediction.new_tensor(QUANTILES)[:, None, None]
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def quantile_distillation_loss(
    prediction, teacher, location, scale, use_arcsinh=True, weighted=False
):
    """Match the student's quantiles to all teacher quantiles.

    Inputs: Student and teacher quantiles provide matched forecast levels. Location,
        scale and the transform align coordinates, and optional horizon weights change
        the contribution of each lead.
    Outputs: One differentiable mean squared error for distilled or hybrid training.
    Purpose: Transfer the teacher's distribution information across all forecast
        quantiles.
    """
    teacher = teacher.to(prediction.device)
    # The teacher supplies raw-unit forecasts. Match them in the student's
    # history-derived coordinates before taking squared errors.
    normalized = (teacher - location.unsqueeze(0)) / scale.unsqueeze(0)
    if use_arcsinh:
        normalized = normalized.asinh()
    errors = (prediction - normalized).square()
    if weighted:
        weights = torch.linspace(
            0.5, 1.5, HORIZON, dtype=prediction.dtype, device=prediction.device
        )
        weights = weights / weights.mean()
        errors = errors * weights[None, None, :]
    return errors.mean()


def objective_loss(
    objective,
    prediction,
    actual,
    teacher,
    location,
    scale,
    teacher_weight,
    use_arcsinh=True,
    weighted=False,
):
    """Calculate supervised, distilled, or blended training loss.

    Inputs: The objective and teacher weight choose the active loss terms. Student
        forecasts, measured values, teacher quantiles and normalization parameters
        supply their inputs. Optional transform and horizon flags follow the experiment
        settings.
    Outputs: A differentiable scalar for backpropagation, using pinball, distillation
        MSE or their weighted sum.
    Purpose: Share an objective definition between model adapters while keeping the
        experiment's weighting explicit.
    """
    if objective == "supervised":
        return pinball_loss(prediction, actual, location, scale, use_arcsinh)
    distilled = quantile_distillation_loss(
        prediction, teacher, location, scale, use_arcsinh, weighted
    )
    if objective == "distilled":
        return distilled
    supervised = pinball_loss(prediction, actual, location, scale, use_arcsinh)
    return (1.0 - teacher_weight) * supervised + teacher_weight * distilled
