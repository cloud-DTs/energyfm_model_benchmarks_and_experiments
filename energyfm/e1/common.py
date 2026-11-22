"""Shared paths, Chronos-2 inference, and paired evaluation helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
from chronos import Chronos2Pipeline

from energyfm.assets import require_model


def choose_device(requested: str = "auto") -> str:
    """Choose the CPU or GPU on which the model will run.

    Inputs: The requested device determines whether to use CPU, require CUDA or pick
        automatically.
    Outputs: A usable device string for placing the model and its inputs. Explicit
        unavailable CUDA raises an error.
    Purpose: Give both forecasting backends consistent device selection.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def load_chronos(device):
    """Load the pinned Chronos 2 checkpoint for local inference."""
    return Chronos2Pipeline.from_pretrained(
        str(require_model("chronos")), device_map=device, local_files_only=True
    )


def predict_median(
    model: Chronos2Pipeline,
    items: list[dict[str, Any]],
    context_length: int,
    horizon: int,
    *,
    batch_size: int = 16,
    submission_size: int | None = None,
    output_dtype: Any = np.float32,
) -> np.ndarray:
    """Predict one future target sequence for each input item using the median.

    Inputs: The model and ordered case dictionaries define the forecasts. Context length
        and horizon specify the windows, batch and submission sizes bound memory use,
        and output_dtype controls numeric storage.
    Outputs: An array of shape [case, horizon] in input order so each forecast can be
        paired with its observations. An empty input returns an empty float32 array.
    Purpose: Apply one median forecast path across E1 experiments and keep model
        batching separate from scoring.

    Notes: When submission_size is omitted, each pipeline call receives at most four
        times batch_size cases.
    """
    if not items:
        return np.empty((0, horizon), dtype=np.float32)
    submission_size = submission_size or batch_size * 4
    rows: list[np.ndarray] = []
    for start in range(0, len(items), submission_size):
        # Send only part of the input list at a time to limit memory use.
        chunk = items[start : start + submission_size]
        # The 0.5 quantile is the median. Use it as the prediction when calculating MAE.
        quantiles, _ = model.predict_quantiles(
            chunk,
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=batch_size,
            context_length=context_length,
        )
        for value in quantiles:
            array = np.asarray(value)
            # Take the single target series and keep input order so each forecast
            # matches its actual values.
            forecast = array.reshape(-1, horizon)[0]
            rows.append(np.asarray(forecast, dtype=output_dtype))
    return np.stack(rows)


def paired_block_ci(
    baseline_errors: Iterable[float],
    candidate_errors: Iterable[float],
    *,
    seed: int = 23,
    replicates: int = 2000,
    block_size: int = 7,
    sampling: str = "vectorized",
    gain_formula: str = "ratio",
) -> tuple[float, float]:
    """Estimate a 95% interval for the percentage decrease in mean absolute error.

    Inputs: Paired error sequences preserve case alignment. The seed, block length and
        replicate count control resampling. Sampling order and gain formula preserve the
        arithmetic of older experiments.
    Outputs: Lower and upper percentage bounds to report uncertainty. Fewer observations
        than one block produce two NaNs.
    Purpose: Account for adjacent forecast errors moving together and use the same
        resampled indices for both variants.

    Notes: The vectorized mode draws all block starts at once. The sequential mode
        preserves the older draw sequence. Both modes use circular blocks. The two gain
        formulas preserve the original floating-point operation order.
    """
    baseline = np.asarray(list(baseline_errors), dtype=float)
    candidate = np.asarray(list(candidate_errors), dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("Paired errors must be equal-length one-dimensional arrays.")
    n = len(baseline)
    # There are not enough error entries for a full block, so return an undefined
    # interval.
    if n < block_size:
        return float("nan"), float("nan")
    if sampling not in {"vectorized", "sequential"}:
        raise ValueError("sampling must be 'vectorized' or 'sequential'.")
    if gain_formula not in {"ratio", "difference"}:
        raise ValueError("gain_formula must be 'ratio' or 'difference'.")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block_size)
    offsets = np.arange(block_size)
    gains = []
    for _ in range(replicates):
        # Nearby errors may move together. Sample neighbouring entries together for both
        # variants.
        if sampling == "vectorized":
            starts = rng.integers(0, n, size=blocks)
            # Join sampled blocks, wrap at the array end, and trim to the original
            # number of errors.
            indices = np.concatenate([(start + offsets) % n for start in starts])[:n]
        else:
            # Preserve the legacy Penmanshiel draw order: draw one start, append its
            # whole block, then draw the next start until the sample is large enough.
            selected = []
            while len(selected) < n:
                start = int(rng.integers(0, n))
                for step in range(block_size):
                    selected.append((start + step) % n)
            indices = np.asarray(selected[:n], dtype=int)
        # Compare the two sampled mean errors. Averaging daily percentages would give a
        # different result.
        denominator = baseline[indices].mean()
        if denominator > 0:
            candidate_mean = candidate[indices].mean()
            if gain_formula == "ratio":
                gain = 100.0 * (1.0 - candidate_mean / denominator)
            else:
                gain = 100.0 * (denominator - candidate_mean) / denominator
            gains.append(gain)
    if not gains:
        return float("nan"), float("nan")
    lower, upper = np.quantile(gains, [0.025, 0.975])
    return float(lower), float(upper)
