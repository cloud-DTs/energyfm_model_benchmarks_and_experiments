"""Generate and reuse the teacher forecasts for the configured case split."""

import torch

from energyfm.assets import DATASETS, MODELS
from energyfm.learning.files import save_tensor
from energyfm.learning.settings import CONTEXT, HORIZON, QUANTILES
from energyfm.paths import data_root


def cache_path(dataset, kind):
    return data_root() / "cache" / "teacher" / f"{dataset}_{kind}.pt"


def cache_metadata(dataset, kind):
    """Identify the dataset version, target type and teacher configuration."""
    return {
        "dataset": dataset,
        "version": DATASETS[dataset]["version"],
        "kind": kind,
        "teacher_revision": MODELS["chronos"]["revision"],
        "context": CONTEXT,
        "horizon": HORIZON,
        "quantiles": QUANTILES,
    }


def case_ids(windows):
    """Return the ordered profile and target-day identifiers for a split."""
    return torch.stack((windows.profile_ids, windows.target_days), dim=1)


def load_teacher_targets(dataset, split, kind="quantile"):
    """Read cached targets after checking the configured teacher and ordered case IDs."""
    path = cache_path(dataset, kind)
    if not path.is_file():
        raise FileNotFoundError(
            "Prepare the teacher cache first with "
            f"'energyfm e2 prepare-targets --dataset {dataset} --kind {kind}'."
        )
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache.get("metadata") != cache_metadata(dataset, kind):
        raise ValueError(
            f"Teacher cache metadata does not match the current settings: {path}"
        )

    for name in ("training", "test"):
        saved = cache[f"{name}_cases"]
        current = case_ids(getattr(split, name))
        if not torch.equal(saved, current):
            raise ValueError(f"Teacher cache {name} case IDs do not match: {path}")

    training = cache["training_targets"]
    test = cache["test_quantiles"]
    training_shape = (len(split.training.histories), HORIZON)
    if kind == "quantile":
        training_shape = (len(QUANTILES), *training_shape)
    test_shape = (len(QUANTILES), len(split.test.histories), HORIZON)
    if training.shape != training_shape or test.shape != test_shape:
        raise ValueError(f"Unexpected teacher target shapes in {path}")
    return training, test


def prepare_teacher_targets(dataset, split, kind, device):
    """Save Chronos 2 forecasts for the training and test histories."""
    from energyfm.learning.models import load_teacher, predict_teacher

    path = cache_path(dataset, kind)
    if path.exists():
        raise FileExistsError(f"Teacher cache already exists: {path}")
    teacher = load_teacher(device)
    training = predict_teacher(teacher, split.training.histories)
    if kind == "point":
        training = training[QUANTILES.index(0.5)].clone()
    test = predict_teacher(teacher, split.test.histories)
    cache = {
        "metadata": cache_metadata(dataset, kind),
        "training_cases": case_ids(split.training),
        "test_cases": case_ids(split.test),
        "training_targets": training,
        "test_quantiles": test,
    }
    save_tensor(path, cache, replace=False)
    return path
