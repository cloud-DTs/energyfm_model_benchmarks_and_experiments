"""Keep architecture-specific model operations separate from experiment logic."""

import torch

from energyfm.assets import require_model
from energyfm.learning.settings import CONTEXT, HORIZON, QUANTILES


def load_student(model_name, device):
    """Load one student architecture from its pinned local model files.

    Inputs: The model key chooses the pinned Toto or Chronos 2 Small weights, and the
        device controls placement.
    Outputs: An evaluation-mode model ready for independent scoring or copying into a
        training candidate.
    Purpose: Load the project snapshots consistently and offline.
    """
    if model_name == "toto":
        from toto2 import Toto2Model

        model = Toto2Model.from_pretrained(
            str(require_model(model_name)), local_files_only=True
        )
        return model.to(device).eval()
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        str(require_model(model_name)), device_map=device, local_files_only=True
    )
    return pipeline.model.eval()


def load_teacher(device, small=False):
    """Load the large or small Chronos forecast pipeline from local files.

    Inputs: The device controls placement, and small selects the Small pipeline instead
        of the large Chronos 2 teacher for benchmarking.
    Outputs: A local Chronos pipeline ready for quantile inference.
    Purpose: Use one loader for teacher forecasts and the additional Small benchmark
        path.
    """
    from chronos import Chronos2Pipeline

    model_name = "chronos-small" if small else "chronos"
    return Chronos2Pipeline.from_pretrained(
        str(require_model(model_name)), device_map=device, local_files_only=True
    )


@torch.inference_mode()
def predict_teacher(pipeline, histories, batch_size=16):
    """Generate nine teacher forecast quantiles for each input history.

    Inputs: The Chronos pipeline and [case, context] histories specify predictions, and
        batch size bounds inference memory.
    Outputs: CPU quantiles shaped [9, case, horizon] for caching or direct use as
        training targets.
    Purpose: Expose a consistent quantile ordering for both experiments and both student
        architectures.
    """
    forecasts = []
    for batch in histories.split(batch_size):
        predictions, _ = pipeline.predict_quantiles(
            batch.unsqueeze(1), prediction_length=HORIZON, quantile_levels=QUANTILES
        )
        examples = []
        for prediction in predictions:
            examples.append(prediction[0])
        quantiles = torch.stack(examples).permute(2, 0, 1)
        forecasts.append(quantiles.cpu())
    return torch.cat(forecasts, dim=1)


def make_toto_inputs(histories, model):
    """Convert history tensors to the input mapping expected by Toto.

    Inputs: Histories provide observed values. The model identifies the device required
        for the inputs.
    Outputs: The target, validity mask and series IDs that Toto's public forecast API
        expects.
    Purpose: Handle single-series input layout in one place for standard Toto inference.
    """
    device = next(model.parameters()).device
    target = histories.to(device).unsqueeze(1)
    return {
        "target": target,
        "target_mask": torch.ones_like(target, dtype=torch.bool),
        "series_ids": torch.zeros(target.shape[:2], dtype=torch.long, device=device),
    }


def toto_scaled_forward(model, histories):
    """Run Toto's differentiable forward pass for student training.

    Inputs: The Toto model and [case, context] histories define the local forecast.
        Internal future placeholders reserve output positions.
    Outputs: Differentiable quantiles [9, case, horizon] and location/scale tensors for
        aligning targets to model coordinates.
    Purpose: Train through the native Toto path while masking future placeholders out of
        the observed context.
    """
    device = next(model.parameters()).device
    histories = histories.to(device)
    placeholders = torch.zeros(
        (len(histories), HORIZON), dtype=histories.dtype, device=device
    )
    target = torch.cat([histories, placeholders], dim=1).unsqueeze(1)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    # The forward pass needs future token positions to produce output patches.
    # Mark those positions outside the observed context before passing them in.
    context_mask = target_mask.clone()
    context_mask[..., CONTEXT:] = False
    series_ids = torch.zeros((len(histories), 1), dtype=torch.long, device=device)
    result = model(
        target=target,
        target_mask=target_mask,
        cpm_mask=context_mask,
        series_ids=series_ids,
    )
    future_patch_count = HORIZON // model.config.patch_size
    # Toto predicts one patch ahead, so select the patches after the last context patch.
    future_quantiles = result.quantiles[..., -(future_patch_count + 1) : -1, :]
    prediction = future_quantiles.flatten(-2).squeeze(2)
    location = result.loc[..., -HORIZON:].squeeze(1)
    scale = result.scale[..., -HORIZON:].squeeze(1)
    return prediction, location, scale


def small_quantile_indices(model):
    """Find Small's output positions for the project's nine quantile levels.

    Inputs: The loaded model supplies the available quantile levels to match against the
        project's ordered levels.
    Outputs: The corresponding output indices in the project's requested quantile order.
    Purpose: Align Small's output columns to common teacher and metric conventions.
    """
    indices = []
    for wanted in QUANTILES:
        for index, available in enumerate(model.chronos_config.quantiles):
            if abs(available - wanted) < 1e-6:
                indices.append(index)
                break
    return indices


def small_scaled_forward(model, histories):
    """Run Chronos Small's native differentiable encoder and prediction head.

    Inputs: The Small model supplies the encoder and head. The histories provide
        independent forecast contexts.
    Outputs: Differentiable quantiles [9, case, horizon], with location and scale for
        target normalization or denormalizing predictions.
    Purpose: Use the native Small forward path for training while keeping cases
        independent.
    """
    device = next(model.parameters()).device
    histories = histories.to(device)
    patch_size = model.chronos_config.output_patch_size
    patch_count = (HORIZON + patch_size - 1) // patch_size
    # Unique group IDs keep examples independent in Small's grouped encoder.
    group_ids = torch.arange(len(histories), dtype=torch.long, device=device)
    encoded, location_scale, _, _ = model.encode(
        context=histories, group_ids=group_ids, num_output_patches=patch_count
    )
    future_states = encoded.last_hidden_state[:, -patch_count:]
    output = model.output_patch_embedding(future_states)
    output = output.reshape(
        len(histories), patch_count, model.num_quantiles, patch_size
    )
    output = output.permute(0, 2, 1, 3)
    output = output.reshape(
        len(histories), model.num_quantiles, patch_count * patch_size
    )
    prediction = output[:, small_quantile_indices(model), :HORIZON].permute(1, 0, 2)
    location, scale = location_scale
    return prediction, location, scale


def scaled_forward(model_name, model, histories):
    """Choose the architecture-specific training forward pass.

    Inputs: Model name chooses the architecture-specific adapter, the model supplies
        parameters and histories specify the forecast cases.
    Outputs: Quantiles and location/scale in a common layout for the shared training
        objectives.
    Purpose: Keep model-specific tensor conversions out of experiment loops.
    """
    if model_name == "toto":
        return toto_scaled_forward(model, histories)
    return small_scaled_forward(model, histories)


@torch.inference_mode()
def predict_student(model_name, model, histories, batch_size=16):
    """Generate evaluation forecasts from a fitted student model.

    Inputs: Model name and loaded weights choose the inference path. Histories select
        the cases, and batch size bounds inference work.
    Outputs: CPU quantiles [9, case, horizon] in original units for evaluation and saved
        predictions.
    Purpose: Use the same output units and axes for both students' metrics.
    """
    model.eval()
    forecasts = []
    for batch in histories.split(batch_size):
        if model_name == "toto":
            result = model.forecast(
                make_toto_inputs(batch, model),
                horizon=HORIZON,
                decode_block_size=768,
                has_missing_values=False,
            )
            quantiles = result[:, :, 0, :]
        else:
            scaled, location, scale = small_scaled_forward(model, batch)
            # Keep the historical batch-first unscaling operations for Small.
            scaled = scaled.permute(1, 0, 2)
            if model.instance_norm.use_arcsinh:
                scaled = scaled.sinh()
            real_values = scaled * scale[:, None, :] + location[:, None, :]
            quantiles = real_values.permute(1, 0, 2)
        forecasts.append(quantiles.float().cpu())
    return torch.cat(forecasts, dim=1)


def trainable_modules(model_name, model, scope):
    """Select the model components that an adaptation scope may update.

    Inputs: The model name and requested adaptation scope identify which existing
        submodules of the model can change.
    Outputs: A list of selected modules for enabling gradients or setting their training
        mode.
    Purpose: Keep model-specific definitions of head-only, last-block and full
        adaptation in one place.
    """
    if scope == "full":
        return [model]
    if model_name == "toto":
        modules = [model.output_head]
        if scope == "last-block":
            modules.append(model.transformer.layers[-1])
        return modules
    if scope == "head":
        return [model.output_patch_embedding]
    return [
        model.encoder.block[-1],
        model.encoder.final_layer_norm,
        model.output_patch_embedding,
    ]


def configure_training(model_name, model, settings):
    """Freeze the model and enable gradients for the selected scope.

    Inputs: The model and architecture key identify available modules, and settings
        specify the scope to unfreeze.
    Outputs: The trainable parameter objects to pass to the optimizer.
    Purpose: Ensure each candidate updates exactly the selected scope from its base
        model.
    """
    model.requires_grad_(False)
    for module in trainable_modules(model_name, model, settings.scope):
        module.requires_grad_(True)
    parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameters.append(parameter)
    return parameters


def enable_training_dropout(model_name, model, settings):
    """Set module training state according to the recorded dropout policy.

    Inputs: The model and architecture identify modules, and the settings choose how
        broadly to enable training mode.
    Outputs: None. Updates module modes in place for the next training pass.
    Purpose: Apply dropout policy independently of parameter freezing.
    """
    if settings.dropout_mode == "all":
        model.train()
        return
    model.eval()
    if settings.dropout_mode == "disabled":
        # Evaluation mode disables dropout without disabling gradient tracking.
        return
    for module in trainable_modules(model_name, model, settings.scope):
        module.train()
