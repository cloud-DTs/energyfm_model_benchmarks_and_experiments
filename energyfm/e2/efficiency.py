"""Benchmark the retained public inference paths without confusing them with training gains."""

import json
import random
import statistics
import time
from contextlib import contextmanager

import torch

from energyfm.learning.data import load_vea_profiles
from energyfm.learning.files import (
    create_run_directory,
    hardware_details,
    select_device,
    write_json,
)
from energyfm.learning.models import (
    load_student,
    load_teacher,
    predict_student,
    predict_teacher,
)


def add_arguments(parser):
    """Add timing options to the command parser.

    Inputs: The benchmark subparser receives the protocol, device and plan flag.
    Outputs: None. Updates the parser in place so the launcher's timing command can
        validate its options.
    Purpose: Offer historical paths or the matched benchmark, which adds Chronos 2
        Small.
    """
    parser.add_argument(
        "--protocol", choices=["historical", "matched"], default="matched"
    )
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--plan", action="store_true")


def inference_scale(tensor, forward_scale, backward_scale):
    """Apply only forward scaling during the inference timing path.

    Inputs: The tensor and forward factor determine the inference result. The backward
        factor is accepted to match the replaced operator's signature.
    Outputs: The tensor unchanged for a unit forward factor or multiplied by that
        factor. Raises an error when gradients are enabled.
    Purpose: Bypass the backward scaling wrapper only in the timed inference path.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Inference scaling cannot be used during training.")
    if forward_scale == 1:
        return tensor
    return forward_scale * tensor


@contextmanager
def optimized_toto_scaling(enabled):
    """Temporarily replace gradient-scaling wrappers during Toto inference.

    Inputs: A boolean selects whether to substitute the two scaling wrappers for the
        enclosed code.
    Outputs: Yields control to the caller and restores both original wrappers on exit,
        including on failure.
    Purpose: Isolate the Toto timing optimization from standard forecasts and training.
    """
    if not enabled:
        yield
        return
    import dd_unit_scaling.scale as datadog_scaling
    import unit_scaling.scale as unit_scaling

    original_datadog = datadog_scaling._ScaledGrad.apply
    original_unit = unit_scaling._ScaledGrad.apply
    datadog_scaling._ScaledGrad.apply = inference_scale
    unit_scaling._ScaledGrad.apply = inference_scale
    # The replacement is process-wide while active. Restore both wrappers before
    # any other benchmark path runs, including when a measurement fails.
    try:
        yield
    finally:
        datadog_scaling._ScaledGrad.apply = original_datadog
        unit_scaling._ScaledGrad.apply = original_unit


def forecast_path(name, models, histories):
    """Run one named model path on a batch of histories.

    Inputs: The named path selects a loaded model and inference API. Histories provide a
        shared batch for timing.
    Outputs: CPU forecast quantiles for measuring the path and checking whether
        optimized Toto agrees with standard Toto.
    Purpose: Give all benchmark paths a common callable shape while retaining their
        inference implementations.
    """
    batch_size = len(histories)
    if name == "teacher":
        return predict_teacher(models["teacher"], histories, batch_size)
    if name == "chronos-small":
        return predict_teacher(models["chronos-small"], histories, batch_size)
    return predict_student("toto", models["toto"], histories, batch_size)


def synchronize(device):
    """Wait for queued GPU operations before starting or ending a timer.

    Inputs: The device determines whether queued CUDA operations must finish before
        timing proceeds.
    Outputs: None. Waits on CUDA so the measured interval includes completed work.
    Purpose: Prevent asynchronous GPU dispatch from being mistaken for forecast latency.
    """
    if device == "cuda":
        torch.cuda.synchronize()


def measure_path(name, models, histories, device):
    """Collect warm inference timings for one model path and batch.

    Inputs: The path name, loaded model mapping and batch define the work. The device
        controls synchronization around each timer.
    Outputs: Thirty warm samples in milliseconds per forecast, after ten warmups, for
        pooling across blocks.
    Purpose: Measure steady-state latency while isolating initialization effects from
        the timed repetitions.
    """
    with optimized_toto_scaling(name == "toto-optimized"):
        for _ in range(10):
            forecast_path(name, models, histories)
        samples = []
        for _ in range(30):
            synchronize(device)
            started = time.perf_counter()
            forecast_path(name, models, histories)
            synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000 / len(histories)
            samples.append(elapsed)
    return samples


def summarize_timings(rows):
    """Pool repeated timing samples and calculate summary statistics.

    Inputs: Raw timing rows identify batch size, path and the individual samples to
        pool.
    Outputs: One record per batch size and path with median, reported p95 and sample
        count for the summary table.
    Purpose: Compare paths consistently while keeping the underlying samples available.
    """
    groups = {}
    for row in rows:
        key = (row["batch_size"], row["path"])
        groups.setdefault(key, []).extend(row["samples_ms_per_forecast"])
    summaries = []
    for (batch_size, path), samples in groups.items():
        samples.sort()
        summaries.append(
            {
                "batch_size": batch_size,
                "path": path,
                "median_ms_per_forecast": statistics.median(samples),
                "p95_ms_per_forecast": samples[int(0.95 * len(samples)) - 1],
                "repeats": len(samples),
            }
        )
    return summaries


def run(args):
    """Print the timing plan or run the requested inference benchmark.

    Inputs: Parsed protocol, device and plan options select paths and determine whether
        to load the base models.
    Outputs: None. Prints the plan or saves raw timing samples, summaries, equivalence
        checks and parameter storage.
    Purpose: Measure the inference paths on the same repeated inputs and verify the
        optimized Toto output.
    """
    names = ["teacher", "toto", "toto-optimized"]
    if args.protocol == "matched":
        names.append("chronos-small")
    if args.plan:
        print(
            json.dumps(
                {
                    "paths": names,
                    "batches": [1, 8],
                    "blocks": 5,
                    "warmups_per_block": 10,
                    "measurements_per_block": 30,
                    "weights": "Pinned unadapted base models, not trained-candidate timings.",
                },
                indent=2,
            )
        )
        return
    device = select_device(args.device)
    # Keep the input windows fixed so measured differences reflect the inference path.
    inputs = load_vea_profiles(8)[:, :512]
    models = {"teacher": load_teacher(device), "toto": load_student("toto", device)}
    if args.protocol == "matched":
        models["chronos-small"] = load_teacher(device, small=True)
    raw = []
    equivalence = []
    for batch_size in [1, 8]:
        histories = inputs[:batch_size]
        for block in range(5):
            order = names.copy()
            random.Random(block + 75).shuffle(order)
            for name in order:
                samples = measure_path(name, models, histories, device)
                raw.append(
                    {
                        "batch_size": batch_size,
                        "block": block,
                        "path": name,
                        "samples_ms_per_forecast": samples,
                    }
                )
            standard = forecast_path("toto", models, histories)
            with optimized_toto_scaling(True):
                optimized = forecast_path("toto", models, histories)
            equal = torch.allclose(standard, optimized, rtol=1e-5, atol=1e-5)
            equivalence.append(
                {
                    "batch_size": batch_size,
                    "block": block,
                    "allclose": equal,
                    "max_absolute_difference": float(
                        (standard - optimized).abs().max()
                    ),
                }
            )
            if not equal:
                raise RuntimeError("Optimized Toto failed forecast equivalence.")
    weight_bytes = {}
    for name, loaded in models.items():
        model = loaded if name == "toto" else loaded.model
        weight_bytes[name] = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    output = create_run_directory("E2", f"efficiency_{args.protocol}")
    write_json(
        output / "results.json",
        {
            "protocol": args.protocol,
            "hardware": hardware_details(device),
            "raw": raw,
            "summary": summarize_timings(raw),
            "equivalence": equivalence,
            "parameter_bytes": weight_bytes,
            "limitations": "Warmed repeated inputs; weight memory excludes runtime overhead; timing is not measured electricity use.",
        },
    )
    print(f"Benchmark completed: {output}", flush=True)
