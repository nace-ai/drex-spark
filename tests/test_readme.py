"""The README as PyPI shows it: pyproject.toml's substitutions make every repository path a GitHub URL."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "nace-ai/drex-spark"
#: Every image source (an HTML src or srcset) and link target (an HTML href or a Markdown link)
TARGETS = re.compile(r'(src|srcset|href)="([^"]+)"|\]\(([^)]+)\)')
ABSOLUTE = re.compile(r"https?://|#")


def targets(text: str) -> list[tuple[bool, str]]:
    """Each target, and whether it is an image."""
    return [(attribute in ("src", "srcset"), html or markdown) for attribute, html, markdown in TARGETS.findall(text)]


def pypi_readme() -> str:
    hook = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["metadata"]["hooks"]
    text = (ROOT / "README.md").read_text()
    for substitution in hook["fancy-pypi-readme"]["substitutions"]:
        text = re.sub(substitution["pattern"], substitution["replacement"], text)
    return text


def test_every_repository_path_in_the_readme_exists():
    paths = [target for _, target in targets((ROOT / "README.md").read_text()) if not ABSOLUTE.match(target)]
    assert paths, "the README links no repository files"
    for path in paths:
        assert (ROOT / path).exists(), path


def test_pypi_shows_every_path_as_a_github_url():
    shown = targets(pypi_readme())
    expected = [
        (image, target)
        if ABSOLUTE.match(target)
        else (image, f"https://raw.githubusercontent.com/{REPOSITORY}/main/{target}")
        if image
        else (image, f"https://github.com/{REPOSITORY}/blob/main/{target}")
        for image, target in targets((ROOT / "README.md").read_text())
    ]
    assert shown == expected
