"""Expose data, model and experiment commands under one command line."""

import argparse

from energyfm.assets import DATASETS, MODELS, fetch_dataset, fetch_model, install_e1


def fetch_data(args):
    """Prepare each requested dataset in the workspace."""
    for dataset in dict.fromkeys(args.datasets):
        if dataset == "e1":
            print(install_e1(args.e1_source))
        else:
            print(fetch_dataset(dataset))


def fetch_models(args):
    """Download each requested model revision."""
    for model in dict.fromkeys(args.models):
        print(fetch_model(model))


def prepare_e1(args):
    """Rebuild an E1 target series from a locally available raw source."""
    from energyfm.e1.experiments import (
        dk1_inputs,
        hill_of_towie,
        midtjylland,
        penmanshiel,
    )

    preparers = {
        "01": hill_of_towie.prepare_inputs,
        "02": penmanshiel.prepare_inputs,
        "03": dk1_inputs.prepare_inputs,
        "08": midtjylland.prepare_inputs,
    }
    for number in dict.fromkeys(args.experiments):
        preparers[number]()


def prepare_teacher(args):
    """Create teacher forecasts for the fixed E2 training and test cases."""
    from energyfm.learning.cache import prepare_teacher_targets
    from energyfm.learning.data import load_dataset
    from energyfm.learning.files import select_device

    split = load_dataset(args.dataset)
    device = select_device(args.device)
    print(prepare_teacher_targets(args.dataset, split, args.kind, device))


def prepare_parser():
    """Build the four benchmark entry points and their asset-preparation commands."""
    from energyfm import benchmark
    from energyfm.e1 import benchmark as e1
    from energyfm.e2 import comparison, efficiency, labels
    from energyfm.e3 import replay

    parser = argparse.ArgumentParser(prog="energyfm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="Download benchmark inputs and models.")
    fetch_commands = fetch.add_subparsers(dest="asset", required=True)
    data = fetch_commands.add_parser("data", help="Install versioned benchmark inputs.")
    data.add_argument("datasets", choices=("e1", *DATASETS), nargs="+")
    data.add_argument("--e1-source", help="E1 ZIP path or HTTPS URL.")
    data.set_defaults(handler=fetch_data)
    models = fetch_commands.add_parser("models", help="Download fixed model revisions.")
    models.add_argument("models", choices=MODELS, nargs="+")
    models.set_defaults(handler=fetch_models)

    preparation = commands.add_parser(
        "prepare-e1", help="Rebuild target tables from local raw files."
    )
    preparation.add_argument("experiments", choices=["01", "02", "03", "08"], nargs="+")
    preparation.set_defaults(handler=prepare_e1)

    baseline = commands.add_parser(
        "benchmark", help="Compare the four forecast models zero-shot."
    )
    benchmark.add_arguments(baseline)
    baseline.set_defaults(handler=benchmark.run)

    context = commands.add_parser(
        "e1", help="Measure the effect of archived weather context."
    )
    e1.add_arguments(context)
    context.set_defaults(handler=e1.run)

    distillation = commands.add_parser(
        "e2", help="Run distillation and inference experiments."
    )
    e2_commands = distillation.add_subparsers(dest="experiment", required=True)
    for name, module in (
        ("compare", comparison),
        ("labels", labels),
        ("efficiency", efficiency),
    ):
        command = e2_commands.add_parser(name, help=module.__doc__)
        module.add_arguments(command)
        command.set_defaults(handler=module.run)

    teacher = e2_commands.add_parser(
        "prepare-targets", help="Generate an E2 teacher cache."
    )
    teacher.add_argument("--dataset", choices=["vea", "uci"], default="vea")
    teacher.add_argument("--kind", choices=["quantile", "point"], default="quantile")
    teacher.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    teacher.set_defaults(handler=prepare_teacher)

    evolution = commands.add_parser("e3", help="Run the continual-learning replay.")
    e3_commands = evolution.add_subparsers(dest="experiment", required=True)
    replay_command = e3_commands.add_parser("replay", help=replay.__doc__)
    replay.add_arguments(replay_command)
    replay_command.set_defaults(handler=replay.run)
    return parser


def main():
    """Dispatch the selected command and report invalid arguments or missing inputs."""
    import torch

    parser = prepare_parser()
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        args.handler(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"energyfm: {error}\n")
