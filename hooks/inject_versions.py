"""MkDocs hook: inject current SLEAP ecosystem versions and the copyright year.

Keeps the docs free of hand-maintained version strings (which silently drift on
every release). Reads the SLEAP version from ``sleap/version.py`` and the
``sleap-io`` / ``sleap-nn`` lower bounds from ``pyproject.toml``, then
substitutes these placeholders anywhere they appear in a page's markdown:

    {{ sleap_version }}     -> e.g. 1.6.3   (sleap.version.__version__)
    {{ sleap_io_version }}  -> e.g. 0.7.1   (sleap-io[all] floor in core deps)
    {{ sleap_nn_version }}  -> e.g. 0.2.0   (sleap-nn[torch] floor in the `nn` extra)

It also sets the footer copyright to the current year on every build, so that
never goes stale either.
"""

from __future__ import annotations

import re
import tomllib
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VERSIONS: dict[str, str] | None = None


def _floor(spec: str) -> str:
    """Return the lower-bound version from a PEP 508 spec like ``pkg>=1.2.3,<2``."""
    m = re.search(r">=\s*([0-9][^,\s]*)", spec)
    if not m:
        raise ValueError(f"no lower bound (>=) found in requirement: {spec!r}")
    return m.group(1)


def _versions() -> dict[str, str]:
    """Read the current ecosystem versions from version.py + pyproject.toml."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    version_py = (_ROOT / "sleap" / "version.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', version_py)
    if match is None:
        raise ValueError("could not find __version__ in sleap/version.py")
    sleap_version = match.group(1)

    sleap_io = next(d for d in project["dependencies"] if re.match(r"sleap-io\b", d))
    sleap_nn = next(
        d for d in project["optional-dependencies"]["nn"] if re.match(r"sleap-nn\b", d)
    )

    return {
        "sleap_version": sleap_version,
        "sleap_io_version": _floor(sleap_io),
        "sleap_nn_version": _floor(sleap_nn),
    }


def on_config(config):
    """Set the footer copyright to the current year on every build."""
    config["copyright"] = f"Copyright &copy; {datetime.now().year} Talmo Lab"
    return config


def _substitute(text: str) -> str:
    """Replace every version placeholder in ``text``."""
    global _VERSIONS
    if _VERSIONS is None:
        _VERSIONS = _versions()
    for key, value in _VERSIONS.items():
        text = text.replace("{{ " + key + " }}", value)
    return text


def on_page_markdown(markdown: str, **kwargs) -> str:  # noqa: ARG001
    """Substitute the version placeholders in each page before rendering."""
    return _substitute(markdown)


def on_post_build(config, **kwargs):  # noqa: ARG001
    """Resolve placeholders in the ``.md`` sources copied by copy_source_markdown.

    That hook copies raw markdown straight from ``docs_dir`` to ``site_dir``, so it
    bypasses ``on_page_markdown``. Re-process the copied files in place so the
    published source mirror shows resolved versions too. (This hook is registered
    after copy_source_markdown, so its ``on_post_build`` has already run.)
    """
    site_dir = Path(config["site_dir"])
    for md_file in site_dir.rglob("*.md"):
        original = md_file.read_text(encoding="utf-8")
        resolved = _substitute(original)
        if resolved != original:
            md_file.write_text(resolved, encoding="utf-8")
