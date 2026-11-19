"""codemeta.json must agree with the package metadata in pyproject.toml."""

import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPDX_URL = "https://spdx.org/licenses/"


@pytest.fixture(scope="module")
def project():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


@pytest.fixture(scope="module")
def codemeta():
    return json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))


def test_name_and_description(project, codemeta):
    assert codemeta["name"] == project["name"]
    assert codemeta["description"] == project["description"]


def test_license(project, codemeta):
    assert codemeta["license"] == SPDX_URL + project["license"]


def test_requirements(project, codemeta):
    assert codemeta["softwareRequirements"] == project["dependencies"]


def test_keywords(project, codemeta):
    assert codemeta["keywords"] == project["keywords"]


def test_python_version(project, codemeta):
    minimum = project["requires-python"].removeprefix(">=")
    assert codemeta["runtimePlatform"] == f"Python {minimum}+"


def test_repository(project, codemeta):
    assert codemeta["codeRepository"] == project["urls"]["Source"]
    assert codemeta["issueTracker"] == project["urls"]["Issues"]
