"""SLEAP Command-Line Interface.

This module provides the primary command-line interface for SLEAP using
click and rich-click. The `sleap` command is the main entry point.

Usage:
    sleap                    Launch the GUI
    sleap my_project.slp     Open project in GUI
    sleap label [FILE]       Launch the GUI (explicit)
    sleap doctor             Show system diagnostics
    sleap --help             Show CLI help

Legacy CLIs (sleap-label, sleap-train, etc.) are maintained for backwards
compatibility but the unified `sleap` command is preferred.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Optional

import rich_click as click
from rich_click import RichHelpConfiguration, rich_config
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import sleap


# =============================================================================
# DefaultGroup Implementation
# =============================================================================


class DefaultGroup(click.RichGroup):
    """A Click group that invokes a default subcommand if none is specified.

    Adapted from click-contrib/click-default-group for rich-click.

    Key behaviors:
    - `sleap` with no args -> invokes `label` command
    - `sleap foo.slp` (unrecognized command) -> invokes `label foo.slp`
    - `sleap doctor` -> invokes `doctor` command normally
    - `sleap --help` -> shows group help
    """

    ignore_unknown_options = True

    def __init__(
        self,
        *args: Any,
        default: Optional[str] = None,
        default_if_no_args: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default
        self.default_if_no_args = default_if_no_args

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # If no args and we have a default, insert it
        if not args and self.default_if_no_args and self.default_cmd_name:
            args.insert(0, self.default_cmd_name)
        return super().parse_args(ctx, args)

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        # First try normal command lookup
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        # If command not found, we'll handle it in resolve_command
        return None

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[Optional[str], Optional[click.Command], list[str]]:
        try:
            # Try to resolve normally first
            cmd_name, cmd, remaining = super().resolve_command(ctx, args)

            # If we found a real command, use it
            if cmd is not None:
                return cmd_name, cmd, remaining
        except click.UsageError:
            # No matching command found
            pass

        # No matching command - use the default and treat first arg as an argument
        if self.default_cmd_name:
            default_cmd = super().get_command(ctx, self.default_cmd_name)
            if default_cmd:
                return self.default_cmd_name, default_cmd, args

        # Re-raise if we can't handle it
        raise click.UsageError(
            f"No such command '{args[0]}'." if args else "No command specified."
        )


# =============================================================================
# CLI Configuration
# =============================================================================

# SLEAP brand colors (from sleap.system_info)
SLEAP_TEAL = "#1abc9c"
SLEAP_BLUE = "#3498db"
SLEAP_PURPLE = "#9b59b6"
SLEAP_ORANGE = "#e67e22"

# Configure rich-click with solarized-slim theme
SLEAP_HELP_CONFIG = RichHelpConfiguration(
    theme="solarized-slim",
    header_text=f"[bold {SLEAP_TEAL}]SLEAP[/] - Social LEAP Estimates Animal Poses",
    footer_text=(
        "[dim]Docs: https://docs.sleap.ai | "
        "Support: https://github.com/talmolab/sleap/discussions[/]"
    ),
    text_markup="rich",
    show_arguments=True,
)


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group(
    cls=DefaultGroup,
    default="label",
    default_if_no_args=True,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@rich_config(help_config=SLEAP_HELP_CONFIG)
@click.version_option(version=sleap.__version__, prog_name="sleap")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """SLEAP: A deep learning framework for multi-animal pose tracking.

    Run [bold cyan]sleap[/] without arguments to launch the GUI.

    [dim]Examples:[/]
      sleap                    Launch the GUI
      sleap my_project.slp     Open project in GUI
      sleap doctor             Show system diagnostics
    """
    pass


# =============================================================================
# Label Command (GUI Launcher)
# =============================================================================


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@rich_config(help_config=SLEAP_HELP_CONFIG)
@click.argument(
    "labels_path",
    required=False,
    type=click.Path(exists=False),
    metavar="[LABELS.slp]",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed startup information including GPU status.",
)
@click.option(
    "--reset",
    is_flag=True,
    help="Reset GUI preferences to defaults.",
)
@click.option(
    "--no-usage-data",
    is_flag=True,
    help="Disable anonymous usage data collection.",
)
@click.option(
    "--nonnative",
    is_flag=True,
    help="Use non-native file dialogs.",
)
@click.option(
    "--profiling",
    is_flag=True,
    help="Enable performance profiling.",
)
def label(
    labels_path: Optional[str],
    verbose: bool,
    reset: bool,
    no_usage_data: bool,
    nonnative: bool,
    profiling: bool,
) -> None:
    """Launch the SLEAP labeling GUI.

    Optionally open a labels file (.slp) directly.

    [dim]Examples:[/]
      sleap label                      Launch empty GUI
      sleap label my_project.slp       Open existing project
      sleap my_project.slp             Same as above (shorthand)
    """
    # Build args list for the existing GUI main function
    args = []

    if labels_path:
        args.append(labels_path)
    if verbose:
        args.append("--verbose")
    if reset:
        args.append("--reset")
    if no_usage_data:
        args.append("--no-usage-data")
    if nonnative:
        args.append("--nonnative")
    if profiling:
        args.append("--profiling")

    # Import and call the existing GUI launcher
    from sleap.gui.app import main as gui_main

    gui_main(args=args if args else None)


# =============================================================================
# Doctor Command (System Diagnostics)
# =============================================================================


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@rich_config(help_config=SLEAP_HELP_CONFIG)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output diagnostics as JSON for programmatic use.",
)
def doctor(output_json: bool) -> None:
    """Show system diagnostics for troubleshooting.

    Displays detailed information about your system configuration,
    including Python environment, GPU status, and package versions.

    This output is designed to be copy-pasted when reporting issues.

    [dim]Examples:[/]
      sleap doctor           Show diagnostics
      sleap doctor --json    Output as JSON
    """
    from sleap.system_info import (
        get_all_package_info,
        get_pytorch_info,
        _get_nvidia_driver_version,
    )

    if output_json:
        _doctor_json()
        return

    console = Console()

    console.print()
    console.print(
        Panel(
            f"[bold {SLEAP_TEAL}]SLEAP System Diagnostics[/]",
            border_style=SLEAP_TEAL,
            padding=(0, 2),
        )
    )
    console.print()

    # -------------------------------------------------------------------------
    # Platform Information
    # -------------------------------------------------------------------------
    platform_table = Table(
        title="Platform",
        show_header=False,
        box=box.SIMPLE,
        title_style=f"bold {SLEAP_BLUE}",
        border_style="dim",
        padding=(0, 2),
    )
    platform_table.add_column("Key", style="cyan")
    platform_table.add_column("Value")

    platform_table.add_row("OS", f"{platform.system()} {platform.release()}")
    platform_table.add_row("Platform", platform.platform())
    platform_table.add_row("Machine", platform.machine())
    processor = platform.processor()
    platform_table.add_row("Processor", processor if processor else "N/A")

    console.print(platform_table)
    console.print()

    # -------------------------------------------------------------------------
    # Python Information
    # -------------------------------------------------------------------------
    python_table = Table(
        title="Python",
        show_header=False,
        box=box.SIMPLE,
        title_style=f"bold {SLEAP_BLUE}",
        border_style="dim",
        padding=(0, 2),
    )
    python_table.add_column("Key", style="cyan")
    python_table.add_column("Value")

    python_table.add_row("Version", sys.version.split()[0])
    python_table.add_row("Executable", sys.executable)
    python_table.add_row("Prefix", sys.prefix)

    # Virtual environment detection
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        python_table.add_row("Virtual Env", venv)

    console.print(python_table)
    console.print()

    # -------------------------------------------------------------------------
    # Conda / UV Information
    # -------------------------------------------------------------------------
    conda_info = _get_conda_info()
    if conda_info:
        conda_table = Table(
            title="Conda",
            show_header=False,
            box=box.SIMPLE,
            title_style=f"bold {SLEAP_BLUE}",
            border_style="dim",
            padding=(0, 2),
        )
        conda_table.add_column("Key", style="cyan")
        conda_table.add_column("Value")
        conda_table.add_row("Environment", conda_info["environment"])
        conda_table.add_row("Prefix", conda_info["prefix"])
        console.print(conda_table)
        console.print()

    uv_info = _get_uv_info()
    if uv_info:
        uv_table = Table(
            title="UV",
            show_header=False,
            box=box.SIMPLE,
            title_style=f"bold {SLEAP_BLUE}",
            border_style="dim",
            padding=(0, 2),
        )
        uv_table.add_column("Key", style="cyan")
        uv_table.add_column("Value")
        uv_table.add_row("Version", uv_info["version"])
        uv_table.add_row("Path", uv_info["path"])
        console.print(uv_table)
        console.print()

    # -------------------------------------------------------------------------
    # GPU / CUDA Information
    # -------------------------------------------------------------------------
    gpu_table = Table(
        title="GPU / CUDA",
        show_header=False,
        box=box.SIMPLE,
        title_style=f"bold {SLEAP_BLUE}",
        border_style="dim",
        padding=(0, 2),
    )
    gpu_table.add_column("Key", style="cyan")
    gpu_table.add_column("Value")

    nvidia_driver = _get_nvidia_driver_version()
    if nvidia_driver:
        gpu_table.add_row("NVIDIA Driver", nvidia_driver)

        gpus = _get_nvidia_gpu_info()
        for i, gpu in enumerate(gpus):
            gpu_table.add_row(
                f"GPU {i}",
                f"{gpu['name']} ({gpu['memory_free']} free / {gpu['memory_total']})",
            )
    else:
        gpu_table.add_row("NVIDIA Driver", "[dim]Not detected[/]")

    # Get PyTorch info
    pytorch_info = get_pytorch_info()
    if pytorch_info["installed"]:
        pytorch_str = f"v{pytorch_info['version']}"
        if pytorch_info["accelerator"] == "cuda":
            pytorch_str += f" (CUDA {pytorch_info['cuda_version']})"
        elif pytorch_info["accelerator"] == "mps":
            pytorch_str += " (MPS)"
        else:
            pytorch_str += " (CPU)"
        gpu_table.add_row("PyTorch", pytorch_str)
    else:
        gpu_table.add_row("PyTorch", "[dim]Not installed[/]")

    console.print(gpu_table)
    console.print()

    # -------------------------------------------------------------------------
    # Package Versions
    # -------------------------------------------------------------------------
    packages_table = Table(
        title="Packages",
        show_header=True,
        header_style=f"bold {SLEAP_TEAL}",
        box=box.SIMPLE,
        title_style=f"bold {SLEAP_BLUE}",
        border_style="dim",
        padding=(0, 2),
    )
    packages_table.add_column("Package", style="cyan")
    packages_table.add_column("Version")
    packages_table.add_column("Source", style="dim")

    packages = get_all_package_info()
    for pkg_name, info in packages.items():
        packages_table.add_row(pkg_name, info["version"], info["source"])

    console.print(packages_table)
    console.print()

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    console.print(
        Panel(
            "[dim]Copy this output when reporting issues at:\n"
            "https://github.com/talmolab/sleap/issues[/]",
            border_style="dim",
            padding=(0, 2),
        )
    )
    console.print()


def _doctor_json() -> None:
    """Output diagnostics as JSON."""
    import json

    from sleap.system_info import (
        get_all_package_info,
        get_pytorch_info,
        _get_nvidia_driver_version,
    )

    data = {
        "sleap_version": sleap.__version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": sys.prefix,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "conda": _get_conda_info(),
        "uv": _get_uv_info(),
        "gpu": {
            "nvidia_driver": _get_nvidia_driver_version(),
            "gpus": _get_nvidia_gpu_info(),
        },
        "pytorch": get_pytorch_info(),
        "packages": get_all_package_info(),
    }

    print(json.dumps(data, indent=2))


def _get_conda_info() -> Optional[dict[str, str]]:
    """Get conda environment information."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_prefix:
        return {
            "environment": conda_env or "base",
            "prefix": conda_prefix,
        }
    return None


def _get_uv_info() -> Optional[dict[str, str]]:
    """Get uv information if available."""
    uv_path = shutil.which("uv")
    if not uv_path:
        return None
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return {
                "version": result.stdout.strip(),
                "path": uv_path,
            }
    except Exception:
        pass
    return None


def _get_nvidia_gpu_info() -> list[dict[str, str]]:
    """Get NVIDIA GPU information."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append(
                            {
                                "name": parts[0],
                                "memory_total": f"{parts[1]} MB",
                                "memory_free": f"{parts[2]} MB",
                                "utilization": f"{parts[3]}%",
                            }
                        )
            return gpus
    except Exception:
        pass
    return []


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    cli()
