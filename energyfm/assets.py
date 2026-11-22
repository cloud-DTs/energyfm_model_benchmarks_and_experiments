"""Fetch dataset snapshots and fixed model revisions for offline runs."""

import json
import tomllib
from importlib.resources import files
from os import environ
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from energyfm.paths import artifact_root, data_root, model_root

with files("energyfm").joinpath("assets.toml").open("rb") as stream:
    ASSETS = tomllib.load(stream)

MODELS = ASSETS["models"]
DATASETS = ASSETS["datasets"]


def dataset_archive(name):
    """Return the local path for the selected upstream archive."""
    return data_root() / "raw" / DATASETS[name]["file"]


def model_path(name):
    """Return the directory for the model's configured revision."""
    revision = MODELS[name]["revision"]
    return model_root() / name / revision


def require_model(name):
    """Resolve a downloaded model and check that its required files are present."""
    path = model_path(name)
    for name_in_snapshot in MODELS[name]["files"]:
        if not (path / name_in_snapshot).is_file():
            raise FileNotFoundError(
                f"Model {name} is missing. Run 'energyfm fetch models {name}'."
            )
    return path


def download_archive(url, destination):
    """Download an archive to a temporary file before installing it.

    Inputs: An upstream URL and the local destination.
    Outputs: The path to the existing or newly downloaded ZIP archive.
    Purpose: Keep interrupted downloads out of the usable dataset directory.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination

    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        request = Request(url, headers={"User-Agent": "energyfm-benchmarks/0.1"})
        with urlopen(request, timeout=120) as response, staged.open("wb") as output:
            copyfileobj(response, output, length=1024 * 1024)
        try:
            with ZipFile(staged) as archive:
                if not archive.infolist():
                    raise ValueError(f"Empty ZIP archive from {url}")
        except BadZipFile as error:
            raise ValueError(f"Invalid ZIP archive from {url}") from error
        if not destination.exists():
            staged.replace(destination)
    return destination


def fetch_dataset(name):
    """Download only the upstream file required by the selected dataset."""
    return download_archive(DATASETS[name]["url"], dataset_archive(name))


def fetch_model(name):
    """Fetch the exact upstream files declared for the selected model."""
    from huggingface_hub import snapshot_download

    model = MODELS[name]
    destination = model_path(name)
    if all((destination / filename).is_file() for filename in model["files"]):
        return destination
    snapshot_download(
        repo_id=model["repository"],
        revision=model["revision"],
        allow_patterns=model["files"],
        local_dir=destination,
    )
    return require_model(name)


def install_e1(source=None):
    """Install the frozen E1 inputs from a local ZIP or an HTTPS download.

    Inputs: An optional ZIP location. The asset configuration supplies the default.
    Outputs: The directory containing the complete E1 input version.
    Purpose: Keep the forecast keys, energy series and weather archive together.
    """
    from energyfm.e1.inputs import bundle_files

    destination = data_root() / "e1"
    required = bundle_files()
    if all((destination / name).is_file() for name in required):
        return destination
    if destination.exists():
        raise FileExistsError(
            f"An incomplete E1 input directory already exists at {destination}."
        )

    bundle = ASSETS["e1"]
    local_zip = artifact_root() / bundle["file"]
    if source is None:
        source = environ.get("ENERGYFM_E1_URL")
        if source is None and local_zip.is_file():
            source = local_zip
        if source is None:
            source = bundle.get("url", local_zip)

    if urlsplit(str(source)).scheme == "https":
        source = download_archive(str(source), local_zip)
    source = Path(source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"E1 bundle not found at {source}. Supply --e1-source with a local ZIP or HTTPS URL."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            ZipFile(source) as archive,
            TemporaryDirectory(dir=destination.parent) as temporary,
        ):
            members = archive.namelist()
            if len(members) != len(set(members)) or set(members) != set(required):
                raise ValueError(
                    "The ZIP does not contain the files for the configured E1 bundle."
                )
            metadata = json.loads(archive.read("sources.json"))
            if (
                not isinstance(metadata, dict)
                or metadata.get("version") != bundle["version"]
            ):
                raise ValueError(
                    "The E1 bundle has a different version from the configuration."
                )

            staged_root = Path(temporary) / destination.name
            for name in required:
                path = staged_root.joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as input_file, path.open("wb") as output:
                    copyfileobj(input_file, output)
            try:
                staged_root.rename(destination)
            except OSError:
                if not all((destination / name).is_file() for name in required):
                    raise
    except BadZipFile as error:
        raise ValueError(f"Invalid E1 ZIP archive at {source}") from error
    return destination
