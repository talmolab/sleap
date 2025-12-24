"""System information and startup banner for SLEAP."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
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
]


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
    table.add_column("Location", style="dim", overflow="ellipsis")

    packages = get_all_package_info()
    for pkg, info in packages.items():
        location = _shorten_path(info["location"], 45)
        table.add_row(pkg, info["version"], info["source"], location)

    console.print(table)
