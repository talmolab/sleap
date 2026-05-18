"""System information and startup banner for SLEAP."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from rich import box
from rich.align import Align
from rich.color import Color
from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# tomllib is Python 3.11+, fallback to tomli or skip TOML parsing
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


# SLEAP brand colors (extracted from logo)
SLEAP_COLORS = {
    "purple": (155, 89, 182),
    "teal": (26, 188, 156),
    "blue": (52, 152, 219),
    "orange": (230, 126, 34),
    "pink": (229, 115, 115),
    "lime": (139, 195, 74),
    "red": (231, 76, 60),
    "light_blue": (93, 173, 226),
}

# Gradient order (follows the S-shape in the logo)
SLEAP_GRADIENT = [
    SLEAP_COLORS["purple"],
    SLEAP_COLORS["teal"],
    SLEAP_COLORS["blue"],
    SLEAP_COLORS["orange"],
    SLEAP_COLORS["pink"],
    SLEAP_COLORS["lime"],
    SLEAP_COLORS["red"],
    SLEAP_COLORS["light_blue"],
]

# Hex color constants for doctor output
SLEAP_TEAL_HEX = "#1abc9c"
SLEAP_BLUE_HEX = "#3498db"
SLEAP_PURPLE_HEX = "#9b59b6"
SLEAP_ORANGE_HEX = "#e67e22"
SLEAP_GREEN_HEX = "#2ecc71"
SLEAP_RED_HEX = "#e74c3c"
SLEAP_YELLOW_HEX = "#f1c40f"
DIM = "dim"

# ASCII art logo
SLEAP_ASCII = r"""
 ____  _     _____    _    ____
/ ___|| |   | ____|  / \  |  _ \
\___ \| |   |  _|   / _ \ | |_) |
 ___) | |___| |___ / ___ \|  __/
|____/|_____|_____/_/   \_\_|
"""

# Key packages to check for verbose mode
PACKAGES = [
    "sleap",
    "sleap-io",
    "sleap-nn",
    "numpy",
    "h5py",
    "PySide6",
    "PySide2",
    "opencv-python",
    "opencv-python-headless",
    "torch",
    "imageio",
    "imageio-ffmpeg",
    "av",
]


# =============================================================================
# Data Classes for Diagnostics
# =============================================================================


@dataclass
class PackageInfoData:
    """Information about an installed Python package."""

    name: str
    version: str
    source: str  # pip, editable, git, conda, local
    location: str = ""
    editable: bool = False
    # Git info (for editable/git installs)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    git_remote: Optional[str] = None


@dataclass
class UVInfo:
    """Information about uv installation and configuration."""

    version: str = ""
    path: str = ""
    cache_dir: str = ""
    tool_dir: str = ""
    tool_bin_dir: str = ""
    python_dir: str = ""
    installed_tools: list[str] = field(default_factory=list)
    # Configuration settings
    default_python: str = ""  # From .python-version or UV_PYTHON
    resolved_python: str = ""  # From `uv python find`
    python_preference: str = ""  # managed, system, only-managed, only-system
    resolution_strategy: str = ""  # highest, lowest, lowest-direct
    index_strategy: str = ""  # first-index, unsafe-first-match, unsafe-best-match
    prerelease: str = ""  # if-necessary, allow


@dataclass
class CondaInfo:
    """Information about conda environment."""

    active: bool = False
    environment: str = ""
    prefix: str = ""
    version: str = ""
    auto_activate_base: Optional[bool] = None
    sleap_packages: list[str] = field(default_factory=list)


@dataclass
class BinaryInfo:
    """Information about a CLI binary."""

    name: str
    path: str
    real_path: str  # resolved symlink
    python_path: str = ""
    source: str = ""  # uv-tool, conda, pip, venv


@dataclass
class GPUInfo:
    """Information about GPU."""

    name: str
    memory_total: str
    memory_free: str
    utilization: str


def _interpolate_color(color1: tuple, color2: tuple, t: float) -> tuple:
    """Interpolate between two RGB colors."""
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t),
    )


def _multi_gradient(colors: list, t: float) -> tuple:
    """Get color from multi-color gradient at position t (0-1)."""
    if t <= 0:
        return colors[0]
    if t >= 1:
        return colors[-1]
    segment_size = 1.0 / (len(colors) - 1)
    segment_idx = min(int(t / segment_size), len(colors) - 2)
    local_t = (t - segment_idx * segment_size) / segment_size
    return _interpolate_color(colors[segment_idx], colors[segment_idx + 1], local_t)


def _create_gradient_text(text: str, colors: list) -> Text:
    """Create text with multi-color gradient."""
    result = Text()
    for i, char in enumerate(text):
        if char != " ":
            r, g, b = _multi_gradient(colors, i / max(len(text) - 1, 1))
            result.append(char, style=Style(color=Color.from_rgb(r, g, b), bold=True))
        else:
            result.append(char)
    return result


def _shorten_path(path: str, max_len: int = 40) -> str:
    """Shorten a path for display, keeping the end."""
    if not path:
        return ""
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3) :]


def get_package_info(name: str) -> Dict:
    """Get package version, location, and install source without importing.

    Uses importlib.metadata so we don't have to import heavy packages just
    to check their versions.

    Args:
        name: Package name (e.g., "sleap", "sleap-io", "numpy")

    Returns:
        Dict with version, location, source, and editable fields.
        If package is not installed, version will be None.
    """
    try:
        dist = importlib.metadata.distribution(name)
        version = dist.version

        # Check for editable install and source via direct_url.json
        is_editable = False
        source = "pip"  # Default assumption
        try:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct_url = json.loads(direct_url_text)
                is_editable = direct_url.get("dir_info", {}).get("editable", False)
                if is_editable:
                    source = "editable"
                elif "vcs_info" in direct_url:
                    source = "git"
                elif direct_url.get("url", "").startswith("file://"):
                    source = "local"
        except FileNotFoundError:
            pass

        # Fallback: detect old-style editable installs (.egg-info not in site-packages)
        if not is_editable and hasattr(dist, "_path") and dist._path:
            path_str = str(dist._path)
            # Old-style editable: .egg-info in source dir, not site-packages
            if ".egg-info" in path_str and "site-packages" not in path_str:
                is_editable = True
                source = "editable"

        # Check for conda install via INSTALLER file (only if not already known)
        if source == "pip":
            try:
                installer = dist.read_text("INSTALLER")
                if installer and installer.strip() == "conda":
                    source = "conda"
            except FileNotFoundError:
                pass

        # Get location
        location = ""
        if hasattr(dist, "_path") and dist._path:
            path = dist._path.parent
            if not path.is_absolute():
                path = Path.cwd() / path
            location = str(path)

        return {
            "version": version,
            "location": location,
            "source": source,
            "editable": is_editable,
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "version": None,  # None = not installed
            "location": "",
            "source": "",
            "editable": False,
        }


def get_all_package_info() -> Dict:
    """Get info for all relevant SLEAP packages.

    Returns:
        Dict mapping package names to their info dicts.
        Only includes packages that are installed.
    """
    result = {}
    for pkg in PACKAGES:
        info = get_package_info(pkg)
        # Only include if installed
        if info["version"] is not None:
            result[pkg] = info
    return result


def _get_nvidia_driver_version() -> Optional[str]:
    """Get NVIDIA driver version from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def get_pytorch_info() -> Dict:
    """Get PyTorch version and device information.

    Avoids importing torch at module level for fast startup.

    Returns:
        Dict with:
            - installed: bool - whether PyTorch is installed
            - version: str or None - PyTorch version
            - accelerator: str - "cuda", "mps", or "cpu"
            - cuda_version: str or None - CUDA version if available
            - driver_version: str or None - NVIDIA driver version if available
            - device_name: str or None - GPU name if available
    """
    # First check if torch is installed via metadata (fast, no import)
    torch_info = get_package_info("torch")
    if torch_info["version"] is None:
        return {
            "installed": False,
            "version": None,
            "accelerator": "cpu",
            "cuda_version": None,
            "driver_version": None,
            "device_name": None,
        }

    # torch is installed, now we need to import to get device info
    try:
        import torch

        result = {
            "installed": True,
            "version": torch.__version__,
            "accelerator": "cpu",
            "cuda_version": None,
            "driver_version": None,
            "device_name": None,
        }

        # Check CUDA
        if torch.cuda.is_available():
            result["accelerator"] = "cuda"
            result["cuda_version"] = torch.version.cuda
            result["driver_version"] = _get_nvidia_driver_version()
            # Get first GPU name
            if torch.cuda.device_count() > 0:
                result["device_name"] = torch.cuda.get_device_name(0)

        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["accelerator"] = "mps"

        return result

    except Exception:
        # If import fails for some reason, return basic info
        return {
            "installed": True,
            "version": torch_info["version"],
            "accelerator": "cpu",
            "cuda_version": None,
            "driver_version": None,
            "device_name": None,
        }


def _get_platform_name() -> str:
    """Get a friendly platform name."""
    import platform

    system = platform.system()
    if system == "Windows":
        release = platform.release()  # "10", "11", etc.
        return f"Windows {release}"
    elif system == "Darwin":
        # macOS - get version like "14.0" for Sonoma
        release = platform.mac_ver()[0]
        if release:
            return f"macOS {release}"
        return "macOS"
    elif system == "Linux":
        # Try to get distro info
        try:
            import distro

            name = distro.name(pretty=True)
            if name:
                return name
        except ImportError:
            pass
        return "Linux"
    return system


def _build_version_line() -> Text:
    """Build the package version info line with colors.

    Shows SLEAP, sleap-io, and sleap-nn versions.
    """
    sleap_info = get_package_info("sleap")
    sleap_io_info = get_package_info("sleap-io")
    sleap_nn_info = get_package_info("sleap-nn")

    sleap_version = sleap_info["version"] or "not installed"

    # Build colored version line
    line = Text()

    # SLEAP (primary - teal)
    line.append("SLEAP", style="bold rgb(26,188,156)")
    line.append(f" v{sleap_version}", style="rgb(93,173,226)")

    # sleap-io (if installed)
    if sleap_io_info["version"]:
        line.append(" | ", style="dim")
        line.append("sleap-io", style="rgb(26,188,156)")
        line.append(f" v{sleap_io_info['version']}", style="rgb(93,173,226)")

    # sleap-nn (if installed)
    if sleap_nn_info["version"]:
        line.append(" | ", style="dim")
        line.append("sleap-nn", style="rgb(26,188,156)")
        line.append(f" v{sleap_nn_info['version']}", style="rgb(93,173,226)")

    return line


def _build_system_line() -> Text:
    """Build the system/platform info line with colors.

    Shows platform and Python version.
    """
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    platform_name = _get_platform_name()

    line = Text()

    # Platform (purple from brand colors)
    line.append(platform_name, style="rgb(155,89,182)")

    line.append(" | ", style="dim")

    # Python (orange from brand colors)
    line.append("Python", style="rgb(230,126,34)")
    line.append(f" {python_version}", style="rgb(93,173,226)")

    return line


def _build_pytorch_line() -> Optional[Text]:
    """Build the PyTorch info line with version and device status."""
    pytorch_info = get_pytorch_info()

    if not pytorch_info["installed"]:
        return None

    # Start with PyTorch version
    line = Text()
    line.append(f"PyTorch v{pytorch_info['version']}", style="dim")
    line.append(" | ", style="dim")

    # Add device info based on accelerator type
    if pytorch_info["accelerator"] == "cuda":
        # GPU: "GPU [OK] | CUDA v12.8 | Driver: 570.65"
        line.append("GPU ", style="dim")
        line.append("[OK]", style="green bold")

        if pytorch_info["cuda_version"]:
            line.append(f" | CUDA v{pytorch_info['cuda_version']}", style="dim")

        if pytorch_info["driver_version"]:
            line.append(f" | Driver: {pytorch_info['driver_version']}", style="dim")

    elif pytorch_info["accelerator"] == "mps":
        # Apple Silicon: "MPS [OK]"
        line.append("MPS ", style="dim")
        line.append("[OK]", style="green bold")

    else:
        # CPU only
        line.append("CPU-only", style="dim yellow")

    return line


def print_startup_banner(verbose: bool = False, console: Optional[Console] = None):
    """Print the SLEAP startup banner with version info.

    Displays a colorful ASCII art banner with SLEAP branding, version
    information, and helpful links for documentation and support.

    Args:
        verbose: If True, show detailed package table with versions and locations.
        console: Optional Rich Console instance. If None, creates a new one.
    """
    if console is None:
        console = Console()

    console.print()

    # Build styled ASCII art with gradient
    lines = SLEAP_ASCII.strip("\n").split("\n")
    max_width = max(len(line) for line in lines)
    total_chars = sum(len(line.replace(" ", "")) for line in lines)

    ascii_art = Text()
    char_count = 0
    for i, line in enumerate(lines):
        # Pad line to max width to preserve alignment when centered
        padded_line = line.ljust(max_width)
        for char in padded_line:
            if char != " ":
                t = char_count / total_chars if total_chars > 0 else 0
                r, g, b = _multi_gradient(SLEAP_GRADIENT, t)
                ascii_art.append(
                    char, style=Style(color=Color.from_rgb(r, g, b), bold=True)
                )
                char_count += 1
            else:
                ascii_art.append(char)
        if i < len(lines) - 1:
            ascii_art.append("\n")

    # Tagline
    tagline = Text("Social LEAP Estimates Animal Poses", style="bold rgb(26,188,156)")

    # Version info line (SLEAP, sleap-io, sleap-nn)
    version_text = _build_version_line()

    # System info line (platform, Python)
    system_text = _build_system_line()

    # PyTorch info line (version and device) - only in verbose mode (slow)
    pytorch_text = _build_pytorch_line() if verbose else None

    # Links
    link_docs = Text()
    link_docs.append("Docs: ", style="dim")
    link_docs.append("https://docs.sleap.ai", style="rgb(93,173,226)")

    link_support = Text()
    link_support.append("Support: ", style="dim")
    link_support.append(
        "https://github.com/talmolab/sleap/discussions", style="rgb(93,173,226)"
    )

    # Happy SLEAPing with gradient
    welcome = _create_gradient_text("Happy SLEAPing!", SLEAP_GRADIENT)

    # Combine all content - center everything
    content_parts = [
        Align.center(ascii_art),
        Text(),
        Align.center(tagline),
        Text(),
        Align.center(version_text),
        Align.center(system_text),
    ]

    # Add PyTorch line if available
    if pytorch_text:
        content_parts.append(Align.center(pytorch_text))

    content_parts.extend(
        [
            Text(),
            Align.center(link_docs),
            Align.center(link_support),
            Text(),
            Align.center(welcome),
        ]
    )

    content = Group(*content_parts)

    # Create fitted panel with teal border
    panel = Panel(
        content,
        box=box.ROUNDED,
        border_style="rgb(26,188,156)",
        padding=(1, 3),
        expand=False,
    )

    console.print(panel)

    # Show verbose package table if requested
    if verbose:
        console.print()
        _print_package_table(console)

    console.print()


def _print_package_table(console: Console):
    """Print a table of installed packages with version info."""
    table = Table(
        title="Installed Packages",
        show_header=True,
        header_style="bold rgb(26,188,156)",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("Package", style="cyan")
    table.add_column("Version", style="white")
    table.add_column("Source", style="yellow")
    table.add_column("Location", style="dim", overflow="fold")

    packages = get_all_package_info()
    for pkg, info in packages.items():
        table.add_row(pkg, info["version"], info["source"], info["location"])

    console.print(table)


# =============================================================================
# Diagnostic Data Collection Functions (for sleap doctor)
# =============================================================================


def run_command(cmd: list[str], timeout: int = 5) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return -1, "", ""


def get_git_info(path: str) -> dict:
    """Get git information for a directory."""
    if not path or not Path(path).exists():
        return {}

    # Check if it's a git repo
    rc, _, _ = run_command(["git", "-C", path, "rev-parse", "--git-dir"])
    if rc != 0:
        return {}

    info = {}

    # Get commit hash
    rc, stdout, _ = run_command(["git", "-C", path, "rev-parse", "--short", "HEAD"])
    if rc == 0:
        info["commit"] = stdout

    # Get branch name
    rc, stdout, _ = run_command(
        ["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]
    )
    if rc == 0:
        info["branch"] = stdout

    # Check if dirty
    rc, stdout, _ = run_command(["git", "-C", path, "status", "--porcelain"])
    if rc == 0:
        info["dirty"] = bool(stdout)

    # Get remote URL
    rc, stdout, _ = run_command(["git", "-C", path, "remote", "get-url", "origin"])
    if rc == 0:
        info["remote"] = stdout

    return info


def get_detailed_package_info(name: str) -> Optional[PackageInfoData]:
    """Get detailed package info including git status for editables."""
    try:
        dist = importlib.metadata.distribution(name)
        version = dist.version

        # Determine source and editability
        is_editable = False
        source = "pip"
        location = ""

        # Check direct_url.json for modern pip installs
        try:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct_url = json.loads(direct_url_text)
                is_editable = direct_url.get("dir_info", {}).get("editable", False)
                if is_editable:
                    source = "editable"
                    # Get location from URL
                    url = direct_url.get("url", "")
                    if url.startswith("file://"):
                        location = url[7:]  # Strip file://
                elif "vcs_info" in direct_url:
                    source = "git"
                elif direct_url.get("url", "").startswith("file://"):
                    source = "local"
        except FileNotFoundError:
            pass

        # Fallback: detect old-style editable installs
        if not is_editable and hasattr(dist, "_path") and dist._path:
            path_str = str(dist._path)
            if ".egg-info" in path_str and "site-packages" not in path_str:
                is_editable = True
                source = "editable"
                location = str(dist._path.parent) if dist._path else ""

        # Check for conda install
        if source == "pip":
            try:
                installer = dist.read_text("INSTALLER")
                if installer and installer.strip() == "conda":
                    source = "conda"
            except FileNotFoundError:
                pass

        # Get location if not already set
        if not location and hasattr(dist, "_path") and dist._path:
            path = dist._path.parent
            if not path.is_absolute():
                path = Path.cwd() / path
            location = str(path)

        # Create PackageInfoData
        pkg_info = PackageInfoData(
            name=name,
            version=version,
            source=source,
            location=location,
            editable=is_editable,
        )

        # Get git info for editable installs
        if is_editable and location:
            git_info = get_git_info(location)
            if git_info:
                pkg_info.git_commit = git_info.get("commit")
                pkg_info.git_branch = git_info.get("branch")
                pkg_info.git_dirty = git_info.get("dirty", False)
                pkg_info.git_remote = git_info.get("remote")

        return pkg_info

    except importlib.metadata.PackageNotFoundError:
        return None


def get_uv_config_value(key: str) -> str:
    """Get a uv config value by checking config files and env vars."""
    # Check environment variable first (takes precedence)
    env_key = f"UV_{key.upper().replace('-', '_')}"
    env_val = os.environ.get(env_key, "")
    if env_val:
        return env_val

    # Skip TOML parsing if tomllib not available
    if tomllib is None:
        return ""

    # Check config files (user config at ~/.config/uv/uv.toml)
    # On Windows, use APPDATA; on Unix, use ~/.config
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA", "")
        config_paths = [
            Path(appdata) / "uv" / "uv.toml" if appdata else None,
        ]
    else:
        config_paths = [
            Path.home() / ".config" / "uv" / "uv.toml",
            Path("/etc/uv/uv.toml"),
        ]

    for config_path in config_paths:
        if config_path and config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)
                    # Check top-level and [tool.uv] section
                    if key in config:
                        return str(config[key])
                    if "tool" in config and "uv" in config["tool"]:
                        if key in config["tool"]["uv"]:
                            return str(config["tool"]["uv"][key])
            except (OSError, Exception):
                pass

    return ""


def get_default_python_version() -> str:
    """Get the default Python version from .python-version files or UV_PYTHON."""
    # Check UV_PYTHON env var first
    uv_python = os.environ.get("UV_PYTHON", "")
    if uv_python:
        return f"{uv_python} (UV_PYTHON)"

    # Check .python-version in current directory and parents
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        pv_file = parent / ".python-version"
        if pv_file.exists():
            try:
                version = pv_file.read_text().strip().split("\n")[0]
                if version:
                    return f"{version} ({pv_file})"
            except OSError:
                pass

    # Check user-level .python-version
    user_pv = Path.home() / ".config" / "uv" / ".python-version"
    if user_pv.exists():
        try:
            version = user_pv.read_text().strip().split("\n")[0]
            if version:
                return f"{version} ({user_pv})"
        except OSError:
            pass

    return ""


def get_uv_info_data() -> Optional[UVInfo]:
    """Get comprehensive uv information including config settings."""
    uv_path = shutil.which("uv")
    if not uv_path:
        return None

    info = UVInfo(path=uv_path)

    # Version
    rc, stdout, _ = run_command(["uv", "--version"])
    if rc == 0:
        info.version = stdout

    # Cache directory
    rc, stdout, _ = run_command(["uv", "cache", "dir"])
    if rc == 0:
        info.cache_dir = stdout

    # Tool directory
    rc, stdout, _ = run_command(["uv", "tool", "dir"])
    if rc == 0:
        info.tool_dir = stdout

    # Tool bin directory
    rc, stdout, _ = run_command(["uv", "tool", "dir", "--bin"])
    if rc == 0:
        info.tool_bin_dir = stdout

    # Python directory
    rc, stdout, _ = run_command(["uv", "python", "dir"])
    if rc == 0:
        info.python_dir = stdout

    # Installed tools
    rc, stdout, _ = run_command(["uv", "tool", "list"])
    if rc == 0:
        for line in stdout.split("\n"):
            if line and not line.startswith("-") and not line.startswith(" "):
                # Format: "tool_name vX.Y.Z"
                parts = line.split()
                if parts:
                    info.installed_tools.append(parts[0])

    # Configuration settings
    info.default_python = get_default_python_version()

    # Get resolved Python using `uv python find`
    rc, stdout, _ = run_command(["uv", "python", "find"])
    if rc == 0 and stdout:
        info.resolved_python = stdout

    info.python_preference = get_uv_config_value("python-preference")
    info.resolution_strategy = get_uv_config_value("resolution")
    info.index_strategy = os.environ.get(
        "UV_INDEX_STRATEGY", ""
    ) or get_uv_config_value("index-strategy")
    info.prerelease = get_uv_config_value("prerelease")

    return info


def get_conda_info_data() -> Optional[CondaInfo]:
    """Get conda environment information."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        # Check if conda is installed but not activated
        conda_path = shutil.which("conda")
        if not conda_path:
            return None
        info = CondaInfo(active=False)
    else:
        info = CondaInfo(
            active=True,
            prefix=conda_prefix,
            environment=os.environ.get("CONDA_DEFAULT_ENV", "base"),
        )

    # Get conda version
    rc, stdout, _ = run_command(["conda", "--version"])
    if rc == 0:
        info.version = stdout

    # Check auto_activate_base setting
    rc, stdout, _ = run_command(["conda", "config", "--show", "auto_activate_base"])
    if rc == 0:
        if "True" in stdout:
            info.auto_activate_base = True
        elif "False" in stdout:
            info.auto_activate_base = False

    # Check for sleap packages in conda environment
    if info.active:
        rc, stdout, _ = run_command(["conda", "list", "sleap"])
        if rc == 0:
            for line in stdout.split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split()
                    if parts and "sleap" in parts[0].lower():
                        info.sleap_packages.append(parts[0])

    return info


def get_binary_info(name: str) -> Optional[BinaryInfo]:
    """Get detailed information about a CLI binary."""
    path = shutil.which(name)
    if not path:
        return None

    info = BinaryInfo(name=name, path=path, real_path=path)

    # Resolve symlinks
    try:
        real_path = Path(path).resolve()
        info.real_path = str(real_path)

        # Determine source based on path
        real_str = str(real_path)
        if ".local/share/uv/tools" in real_str:
            info.source = "uv-tool"
        elif "conda" in real_str.lower() or "miniconda" in real_str.lower():
            info.source = "conda"
        elif ".venv" in real_str or "venv" in real_str:
            info.source = "venv"
        else:
            info.source = "pip"

        # Try to get the Python interpreter from script shebang
        if real_path.exists():
            try:
                with open(real_path) as f:
                    first_line = f.readline()
                    if first_line.startswith("#!") and "python" in first_line:
                        info.python_path = first_line[2:].strip()
            except (OSError, UnicodeDecodeError):
                pass

    except OSError:
        pass

    return info


def get_nvidia_info() -> tuple[str, str, list[GPUInfo]]:
    """Get NVIDIA driver version, CUDA version, and GPU info."""
    if not shutil.which("nvidia-smi"):
        return "", "", []

    # Driver version
    driver = ""
    rc, stdout, _ = run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if rc == 0:
        driver = stdout.split("\n")[0]

    # System CUDA version (from nvidia-smi header)
    # nvidia-smi shows "CUDA Version: X.Y" in the header
    cuda_version = ""
    rc, stdout, _ = run_command(["nvidia-smi"])
    if rc == 0:
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", stdout)
        if match:
            cuda_version = match.group(1)

    # GPU info
    gpus = []
    rc, stdout, _ = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc == 0:
        for line in stdout.split("\n"):
            if line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        GPUInfo(
                            name=parts[0],
                            memory_total=f"{parts[1]} MB",
                            memory_free=f"{parts[2]} MB",
                            utilization=f"{parts[3]}%",
                        )
                    )

    return driver, cuda_version, gpus


def get_pytorch_info_detailed() -> tuple[str, str, str]:
    """Get PyTorch version, accelerator, and CUDA version."""
    # Check if torch is installed first (fast check via metadata)
    torch_pkg = get_package_info("torch")
    if torch_pkg["version"] is None:
        return "", "", ""

    try:
        import torch

        version = torch.__version__
        if torch.cuda.is_available():
            accelerator = "cuda"
            cuda_version = torch.version.cuda or ""
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
            cuda_version = ""
        else:
            accelerator = "cpu"
            cuda_version = ""
        return version, accelerator, cuda_version
    except ImportError:
        return "", "", ""


def get_memory_info() -> tuple[str, str, str]:
    """Get RAM usage info: (used, available, total). Cross-platform."""

    def fmt(b):
        if b >= 1024**3:
            return f"{b / 1024**3:.1f} GB"
        elif b >= 1024**2:
            return f"{b / 1024**2:.1f} MB"
        return f"{b / 1024:.1f} KB"

    # Try platform-specific methods
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1]) * 1024  # kB to bytes

            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", 0)
            used = total - available
            return fmt(used), fmt(available), fmt(total)
        except (OSError, KeyError, ValueError):
            pass

    elif system == "Darwin":  # macOS
        try:
            # Use vm_stat for memory info
            rc, stdout, _ = run_command(["vm_stat"])
            if rc == 0:
                # Parse vm_stat output
                stats = {}
                for line in stdout.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        val = val.strip().rstrip(".")
                        try:
                            stats[key.strip()] = int(val)
                        except ValueError:
                            pass

                page_size = 16384  # Default, could parse from header
                # Try to get page size from first line
                if "page size of" in stdout:
                    match = re.search(r"page size of (\d+) bytes", stdout)
                    if match:
                        page_size = int(match.group(1))

                # Get total via sysctl
                rc2, stdout2, _ = run_command(["sysctl", "-n", "hw.memsize"])
                if rc2 == 0:
                    total = int(stdout2.strip())
                    free_pages = stats.get("Pages free", 0)
                    inactive_pages = stats.get("Pages inactive", 0)
                    available = (free_pages + inactive_pages) * page_size
                    used = total - available
                    return fmt(used), fmt(available), fmt(total)
        except (OSError, ValueError):
            pass

    elif system == "Windows":
        try:
            # Use wmic for memory info
            wmic_fields = "TotalVisibleMemorySize,FreePhysicalMemory"
            rc, stdout, _ = run_command(["wmic", "OS", "get", wmic_fields, "/VALUE"])
            if rc == 0:
                values = {}
                for line in stdout.split("\n"):
                    if "=" in line:
                        key, val = line.strip().split("=", 1)
                        try:
                            values[key] = int(val) * 1024  # kB to bytes
                        except ValueError:
                            pass
                total = values.get("TotalVisibleMemorySize", 0)
                available = values.get("FreePhysicalMemory", 0)
                used = total - available
                if total:
                    return fmt(used), fmt(available), fmt(total)
        except (OSError, ValueError):
            pass

    return "", "", ""


def get_disk_info(path: str) -> tuple[str, str, str]:
    """Get disk usage info for path: (used, available, total)."""
    try:
        usage = shutil.disk_usage(path)

        def fmt(b):
            if b >= 1024**4:
                return f"{b / 1024**4:.1f} TB"
            elif b >= 1024**3:
                return f"{b / 1024**3:.1f} GB"
            elif b >= 1024**2:
                return f"{b / 1024**2:.1f} MB"
            return f"{b / 1024:.1f} KB"

        return fmt(usage.used), fmt(usage.free), fmt(usage.total)
    except (OSError, AttributeError):
        return "", "", ""


def get_ffmpeg_info() -> list[BinaryInfo]:
    """Get ffmpeg binary information from PATH and imageio-ffmpeg."""
    binaries = []

    # Check PATH for ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        info = BinaryInfo(name="ffmpeg", path=ffmpeg_path, real_path=ffmpeg_path)
        try:
            real_path = Path(ffmpeg_path).resolve()
            info.real_path = str(real_path)
        except OSError:
            pass
        info.source = "PATH"

        # Get version
        rc, stdout, _ = run_command(["ffmpeg", "-version"])
        if rc == 0:
            # First line is like "ffmpeg version X.Y.Z ..."
            first_line = stdout.split("\n")[0] if stdout else ""
            if "version" in first_line:
                parts = first_line.split()
                for i, p in enumerate(parts):
                    if p == "version" and i + 1 < len(parts):
                        info.python_path = f"v{parts[i + 1]}"  # Reuse field for version
                        break
        binaries.append(info)

    # Check imageio-ffmpeg
    try:
        import imageio_ffmpeg

        imageio_path = imageio_ffmpeg.get_ffmpeg_exe()
        if imageio_path and imageio_path != ffmpeg_path:
            info = BinaryInfo(
                name="ffmpeg (imageio)",
                path=imageio_path,
                real_path=imageio_path,
                source="imageio-ffmpeg",
            )
            try:
                real_path = Path(imageio_path).resolve()
                info.real_path = str(real_path)
            except OSError:
                pass

            # Get version
            rc, stdout, _ = run_command([imageio_path, "-version"])
            if rc == 0:
                first_line = stdout.split("\n")[0] if stdout else ""
                if "version" in first_line:
                    parts = first_line.split()
                    for i, p in enumerate(parts):
                        if p == "version" and i + 1 < len(parts):
                            info.python_path = f"v{parts[i + 1]}"
                            break
            binaries.append(info)
    except (ImportError, Exception):
        pass

    return binaries


def analyze_path() -> tuple[list[str], list[str]]:
    """Analyze PATH for conflicts."""
    path_str = os.environ.get("PATH", "")
    entries = path_str.split(os.pathsep)

    conflicts = []

    # Check for common conflict patterns
    # Normalize path separators for cross-platform matching
    normalized = [p.replace("\\", "/").lower() for p in entries]
    conda_paths = [p for p in normalized if "conda" in p or "miniconda" in p]
    uv_paths = [p for p in normalized if ".local/share/uv" in p or ".local/bin" in p]

    if conda_paths and uv_paths:
        # Check priority - which comes first?
        for i, path in enumerate(normalized):
            if "conda" in path:
                first_conda = i
                break
        else:
            first_conda = len(entries)

        for i, path in enumerate(normalized):
            if ".local/bin" in path:
                first_uv = i
                break
        else:
            first_uv = len(entries)

        if first_conda < first_uv:
            conflicts.append(
                "Conda paths appear before uv paths in PATH - "
                "conda binaries may take precedence"
            )

    return entries, conflicts
