"""Train both student architectures with explicit, inspectable budgets and selection rules."""

import copy
import math
import time

import torch

from energyfm.learning.losses import objective_loss, pinball_loss
from energyfm.learning.metrics import validation_mae
from energyfm.learning.models import (
    configure_training,
    enable_training_dropout,
    scaled_forward,
)


def snapshot_trainable_weights(model):
    """Copy the current trainable parameters to independent CPU tensors.

    Inputs: The model supplies named parameters and their trainable flags.
    Outputs: Independent CPU copies of the trainable parameters so subsequent updates
        cannot mutate the snapshot.
    Purpose: Preserve the best validation epoch without copying frozen parameter
        tensors.
    """
    state = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().cpu().clone()
    return state


def candidate_order(case_count, label_mask=None, generator=None):
    """Build one epoch's case order, optionally repeating labelled cases.

    Inputs: The case count specifies the epoch length. An optional label mask restricts
        a measured-only control, and the generator controls reproducible ordering.
    Outputs: An index tensor of the original epoch length, repeating labelled indices
        when the control has fewer cases.
    Purpose: Keep the control's optimizer-update budget comparable at each measured-
        label fraction.
    """
    # Draw the full permutation in both branches to keep seeded runs aligned.
    order = torch.randperm(case_count, generator=generator)
    if label_mask is None:
        return order
    labelled = label_mask.nonzero().flatten()
    if len(labelled) == case_count:
        return order
    if not len(labelled):
        raise ValueError(
            "The measured-only control requires at least one labelled case."
        )
    repetitions = math.ceil(case_count / len(labelled))
    shuffled_subsets = []
    for _ in range(repetitions):
        permutation = torch.randperm(len(labelled), generator=generator)
        shuffled_subsets.append(labelled[permutation])
    return torch.cat(shuffled_subsets)[:case_count]


def train_candidate(
    base,
    model_name,
    training,
    validation,
    teacher_targets,
    objective,
    seed,
    settings,
    measured_mask=None,
    point_targets=None,
    horizon_weighted=False,
):
    """Train one candidate and restore its best validation checkpoint.

    Inputs: The base, model key and case splits define the fit. Teacher or point
        targets, the objective and optional mask specify what supervision each case
        uses. Seed, settings and horizon weights define training conditions.
    Outputs: A fitted model restored to its best validation epoch and a log with loss
        history, MAE, update count and runtime.
    Purpose: Train E2 candidates consistently with validation-based checkpoint selection
        before scoring on test.
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    parameters = configure_training(model_name, model, settings)
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    best_mae = float("inf")
    best_state = None
    best_epoch = None
    history = []
    started = time.perf_counter()
    shuffle_generator = None
    if settings.independent_shuffle:
        shuffle_generator = torch.Generator().manual_seed(seed)
    use_arcsinh = True
    if model_name == "chronos-small":
        use_arcsinh = model.instance_norm.use_arcsinh
    for epoch in range(1, settings.epochs + 1):
        enable_training_dropout(model_name, model, settings)
        order = candidate_order(
            len(training.histories), measured_mask, shuffle_generator
        )
        loss_sum = 0.0
        for indices in order.split(settings.batch_size):
            prediction, location, scale = scaled_forward(
                model_name, model, training.histories[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            if point_targets is not None:
                loss = pinball_loss(
                    prediction, point_targets[indices], location, scale, use_arcsinh
                )
            else:
                # Both adapters return [quantile, case, horizon], so one objective
                # implementation can train Toto and Chronos Small alike.
                loss = objective_loss(
                    objective,
                    prediction,
                    training.targets[indices],
                    teacher_targets[:, indices],
                    location,
                    scale,
                    settings.teacher_weight,
                    use_arcsinh,
                    horizon_weighted,
                )
            # Keep one numerical guard: never save silently diverged training.
            if not torch.isfinite(loss):
                raise RuntimeError("Training diverged: the loss is not finite.")
            loss.backward()
            if settings.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        score = validation_mae(
            model_name, model, validation, settings.evaluation_batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / len(training.histories),
                "validation_mae": score,
            }
        )
        if score < best_mae:
            best_mae = score
            best_epoch = epoch
            best_state = snapshot_trainable_weights(model)
        print(f"Epoch {epoch}: validation MAE {score:.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("No finite validation-selected checkpoint was produced.")
    # The snapshot contains trainable tensors only. Frozen parameters stay as
    # initialized in the cloned base model when the best epoch is restored.
    model.load_state_dict(best_state, strict=False)
    model.eval()
    information = {
        "best_validation_mae": best_mae,
        "best_epoch": best_epoch,
        "updates": settings.epochs
        * math.ceil(len(training.histories) / settings.batch_size),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }
    return model, information


def train_local_update(
    base, windows, teacher_targets, settings, seed, model_name="toto"
):
    """Fit a short local update with matched measurements and teacher forecasts.

    Inputs: The base model, aligned windows, teacher forecasts, update settings and seed.
    Outputs: An updated model in evaluation mode and the training statistics.
    Purpose: Fit each continual-learning candidate independently from the same base.
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    parameters = configure_training(model_name, model, settings)
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    use_arcsinh = True
    if model_name == "chronos-small":
        use_arcsinh = model.instance_norm.use_arcsinh

    # Head-only and full-model updates use the same sampled batches for a given seed.
    started = time.perf_counter()
    model.train()
    for _ in range(settings.steps):
        indices = torch.randint(
            len(windows.histories), (settings.batch_size,), generator=generator
        )
        prediction, location, scale = scaled_forward(
            model_name, model, windows.histories[indices]
        )
        optimizer.zero_grad(set_to_none=True)
        loss = objective_loss(
            "hybrid",
            prediction,
            windows.targets[indices],
            teacher_targets[:, indices],
            location,
            scale,
            settings.teacher_weight,
            use_arcsinh,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("The local update returned a non-finite loss.")
        loss.backward()
        optimizer.step()
    model.eval()
    return model, {
        "steps": settings.steps,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "teacher_weight": settings.teacher_weight,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "training_seconds": time.perf_counter() - started,
    }
