"""Resolve local benchmark resources relative to the current workspace."""

from os import environ
from pathlib import Path


def workspace():
    """Return the configured workspace directory or the current directory."""
    root = environ.get("ENERGYFM_HOME")
    if root:
        return Path(root).expanduser().resolve()
    return Path.cwd()


def data_root():
    return workspace() / "data"


def model_root():
    return workspace() / "models"


def artifact_root():
    return workspace() / "artifacts"


def run_root():
    return workspace() / "runs"
