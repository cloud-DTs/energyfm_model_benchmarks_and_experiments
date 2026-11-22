"""Read and save experiment artifacts without replacing previous experiment runs."""

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from energyfm.assets import DATASETS, MODELS
from energyfm.paths import run_root


def write_json(path, value):
    """Write a JSON result using an atomic file replacement.

    Inputs: The destination and JSON-compatible value specify the result artifact.
    Outputs: None. Replaces the destination atomically after writing a complete new
        file.
    Purpose: Keep a readable result on disk when a run is interrupted before a
        replacement completes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(path)


def save_tensor(path, value, *, replace=True):
    """Save a PyTorch payload without exposing a partially written file.

    Inputs: The destination path and PyTorch-serializable value specify the checkpoint,
        predictions or cache data to store.
    Outputs: None. Writes to a temporary file and replaces the destination after
        successful serialization.
    Purpose: Avoid exposing partially serialized PyTorch artifacts to readers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        torch.save(value, temporary)
        if replace:
            temporary.replace(path)
        else:
            path.hardlink_to(temporary)


def create_run_directory(section, experiment):
    """Create a unique timestamped directory for one new experiment run.

    Inputs: The project section and experiment name locate the parent and provide a
        readable run prefix.
    Outputs: A newly created timestamped directory for this run's artifacts.
    Purpose: Keep new results identifiable without altering prior runs or references.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = run_root() / section.lower() / f"{experiment}_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def select_device(requested):
    """Resolve automatic or explicit CPU/CUDA device selection.

    Inputs: The requested device controls CPU use, explicit CUDA or automatic selection.
    Outputs: A usable device string for model placement. Explicit unavailable CUDA
        raises an error.
    Purpose: Give training and benchmarking consistent device selection.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose --device cpu explicitly.")
    return requested


def hardware_details(device):
    """Collect the runtime details needed to interpret experiment timings.

    Inputs: The selected device tells the function whether to query GPU-specific
        details.
    Outputs: A metadata mapping with runtime versions and optional GPU details for
        interpreting results.
    Purpose: Record the execution environment alongside experimental output.
    """
    details = {
        "device": device,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    if device == "cuda":
        details["gpu"] = torch.cuda.get_device_name(0)
    return details


def experiment_provenance(dataset, model_names):
    """Record the published data version and model revisions used in a run."""
    revisions = {}
    for name in model_names:
        model_name = "chronos" if name == "teacher" else name
        revisions[name] = MODELS[model_name]["revision"]
    return {
        "dataset_version": DATASETS[dataset]["version"],
        "model_revisions": revisions,
    }


def save_candidate(path, model, model_name, settings, seed, training):
    """Save trainable parameters and the settings required to identify a candidate.

    Inputs: The checkpoint path, trained model, model key, settings, seed and training
        log identify the fit to preserve.
    Outputs: None. Writes the trainable parameter tensors and metadata needed to reload
        them on the pinned base.
    Purpose: Preserve exactly the fitted part of each candidate, including all
        parameters for full fine-tuning.
    """
    state = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().cpu().clone()
    payload = {
        "model": model_name,
        "revision": MODELS[model_name]["revision"],
        "settings": asdict(settings),
        "seed": seed,
        "training": training,
        "state_dict": state,
    }
    save_tensor(path, payload)


def save_predictions(path, windows, forecasts):
    """Save forecasts alongside the measured values and case identifiers.

    Inputs: The destination, case records and forecasts identify the outputs and their
        matching measurements.
    Outputs: None. Saves quantiles, actuals, profile IDs and target days so forecasts
        can be rescored.
    Purpose: Keep per-case evidence for validating aggregate scores.
    """
    payload = {
        "actual": windows.targets,
        "profile_ids": windows.profile_ids,
        "target_days": windows.target_days,
        "forecasts": forecasts,
    }
    save_tensor(path, payload)
