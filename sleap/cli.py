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
import sys
from typing import Any, Optional

import rich_click as click
from rich_click import RichHelpConfiguration, rich_config
from rich.console import Console

import sleap

# Import sleap-io CLI commands for integration (requires sleap-io>=0.6.1)
from sleap_io.io.cli import (
    show as sio_show,
    convert as sio_convert,
    split as sio_split,
    unsplit as sio_unsplit,
    merge as sio_merge,
    filenames as sio_filenames,
    render as sio_render,
    fix as sio_fix,
    embed as sio_embed,
    unembed as sio_unembed,
    trim as sio_trim,
    reencode as sio_reencode,
    transform as sio_transform,
)


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
SLEAP_GREEN = "#2ecc71"
SLEAP_RED = "#e74c3c"
SLEAP_YELLOW = "#f1c40f"
SLEAP_CYAN = "#00bcd4"  # Paths/filenames - visible on both light and dark terminals

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

# Command organization for help display
click.rich_click.COMMAND_GROUPS = {
    "sleap": [
        {"name": "Application", "commands": ["label", "doctor"]},
        {"name": "Data Inspection", "commands": ["show", "filenames"]},
        {
            "name": "Data Transformation",
            "commands": ["convert", "split", "unsplit", "merge", "trim", "fix"],
        },
        {"name": "Frame Management", "commands": ["embed", "unembed"]},
        {"name": "Video Processing", "commands": ["reencode", "transform", "render"]},
    ]
}


def wrap_sio_command(sio_cmd: click.Command) -> click.Command:
    """Wrap a sleap-io CLI command with SLEAP branding.

    This creates a new command that:
    1. Has the same parameters as the original command
    2. Uses SLEAP's rich-click configuration for help formatting
    3. Replaces 'sio' with 'sleap' in help text examples

    Args:
        sio_cmd: A Click Command object from sleap-io.

    Returns:
        A new Command object with SLEAP branding applied.
    """
    import copy

    # Deep copy to avoid modifying the original
    new_cmd = copy.copy(sio_cmd)

    # Replace examples in help text
    if new_cmd.help:
        new_cmd.help = new_cmd.help.replace("$ sio ", "$ sleap ")
        # Also replace any "sio" command references in the docs
        new_cmd.help = new_cmd.help.replace("[bold]sio[/]", "[bold]sleap[/]")

    # Apply SLEAP's rich-click configuration
    # RichCommand stores config in _rich_config attribute
    new_cmd._rich_config = SLEAP_HELP_CONFIG

    return new_cmd


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

# Field width constants for aligned output (right-aligned field names)
_DOCTOR_WIDTHS = {
    "platform": 9,  # "Processor"
    "python": 11,  # "Virtual Env"
    "uv": 15,  # "Installed Tools"
    "uv_config": 17,  # "Python Preference"
    "conda": 18,  # "auto_activate_base"
    "gpu": 13,  # "NVIDIA Driver"
    "binaries": 9,  # "Real Path"
}


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@rich_config(help_config=SLEAP_HELP_CONFIG)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output diagnostics as JSON for programmatic use.",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    default=None,
    help="Save output to file. Use '-o auto' for auto-timestamped filename.",
)
def doctor(output_json: bool, output_file: Optional[str]) -> None:
    """Show system diagnostics for troubleshooting.

    Displays detailed information about your system configuration,
    including Python environment, GPU status, package versions,
    UV/conda configuration, and more.

    This output is designed to be copy-pasted when reporting issues.

    [dim]Examples:[/]
      sleap doctor           Show diagnostics
      sleap doctor --json    Output as JSON
      sleap doctor -o        Save to auto-timestamped file
      sleap doctor -o out.txt   Save to specific file
    """
    from datetime import datetime
    from pathlib import Path

    from sleap.system_info import (
        PACKAGES,
        DIM,
        get_detailed_package_info,
        get_uv_info_data,
        get_conda_info_data,
        get_binary_info,
        get_nvidia_info,
        get_pytorch_info_detailed,
        get_memory_info,
        get_disk_info,
        get_ffmpeg_info,
        analyze_path,
    )

    if output_json:
        _doctor_json()
        return

    console = Console()
    all_data = {}

    # Print header
    console.print()
    console.print(f"[bold {SLEAP_TEAL}]SLEAP System Diagnostics[/]")
    console.print(f"[{SLEAP_TEAL}]{'=' * 24}[/]")

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[{DIM}]Generated:[/] {timestamp}")
    console.print()
    all_data["timestamp"] = timestamp

    # -------------------------------------------------------------------------
    # Platform Information
    # -------------------------------------------------------------------------
    ram_used, ram_avail, ram_total = get_memory_info()
    venv_path = os.environ.get("VIRTUAL_ENV", "") or sys.prefix
    disk_used, disk_avail, disk_total = get_disk_info(venv_path)

    all_data["platform"] = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "platform_full": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "ram_used": ram_used,
        "ram_total": ram_total,
        "disk_used": disk_used,
        "disk_total": disk_total,
    }

    console.print("[Platform]", style=f"bold {SLEAP_BLUE}")
    w = _DOCTOR_WIDTHS["platform"] + 1  # +1 for colon
    console.print(f"  [{DIM}]{'OS:':<{w}}[/] {platform.system()} {platform.release()}")
    console.print(f"  [{DIM}]{'Platform:':<{w}}[/] {platform.platform()}")
    console.print(f"  [{DIM}]{'Machine:':<{w}}[/] {platform.machine()}")
    processor = platform.processor()
    if processor:
        console.print(f"  [{DIM}]{'Processor:':<{w}}[/] {processor}")
    if ram_total:
        console.print(f"  [{DIM}]{'RAM:':<{w}}[/] {ram_used} / {ram_total}")
    if disk_total:
        console.print(f"  [{DIM}]{'Disk:':<{w}}[/] {disk_used} / {disk_total}")
    console.print()

    # -------------------------------------------------------------------------
    # Python Information
    # -------------------------------------------------------------------------
    all_data["python"] = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
    }

    console.print("[Python]", style=f"bold {SLEAP_BLUE}")
    w = _DOCTOR_WIDTHS["python"] + 1  # +1 for colon
    py_ver = sys.version.split()[0]
    console.print(f"  [{DIM}]{'Version:':<{w}}[/] [{SLEAP_GREEN}]{py_ver}[/]")
    console.print(f"  [{DIM}]{'Executable:':<{w}}[/] [{SLEAP_CYAN}]{sys.executable}[/]")
    console.print(f"  [{DIM}]{'Prefix:':<{w}}[/] [{SLEAP_CYAN}]{sys.prefix}[/]")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        console.print(f"  [{DIM}]{'Virtual Env:':<{w}}[/] [{SLEAP_CYAN}]{venv}[/]")
    console.print()

    # -------------------------------------------------------------------------
    # UV Information
    # -------------------------------------------------------------------------
    with console.status(f"[{DIM}]Checking UV...[/]", spinner="dots"):
        uv_info = get_uv_info_data()
    all_data["uv"] = uv_info

    if uv_info:
        console.print("[UV]", style=f"bold {SLEAP_BLUE}")
        w = _DOCTOR_WIDTHS["uv"] + 1  # +1 for colon
        uv_ver = uv_info.version
        console.print(f"  [{DIM}]{'Version:':<{w}}[/] [{SLEAP_GREEN}]{uv_ver}[/]")
        console.print(f"  [{DIM}]{'Path:':<{w}}[/] [{SLEAP_CYAN}]{uv_info.path}[/]")
        uv_cache = uv_info.cache_dir
        console.print(f"  [{DIM}]{'Cache Dir:':<{w}}[/] [{SLEAP_CYAN}]{uv_cache}[/]")
        uv_tool = uv_info.tool_dir
        console.print(f"  [{DIM}]{'Tool Dir:':<{w}}[/] [{SLEAP_CYAN}]{uv_tool}[/]")
        uv_tool_bin = uv_info.tool_bin_dir
        console.print(
            f"  [{DIM}]{'Tool Bin Dir:':<{w}}[/] [{SLEAP_CYAN}]{uv_tool_bin}[/]"
        )
        uv_py_dir = uv_info.python_dir
        console.print(f"  [{DIM}]{'Python Dir:':<{w}}[/] [{SLEAP_CYAN}]{uv_py_dir}[/]")
        if uv_info.installed_tools:
            tools_str = ", ".join(uv_info.installed_tools)
            console.print(f"  [{DIM}]{'Installed Tools:':<{w}}[/] {tools_str}")
        console.print()

        # UV Config
        console.print("[UV Config]", style=f"bold {SLEAP_BLUE}")
        w = _DOCTOR_WIDTHS["uv_config"] + 1  # +1 for colon
        if uv_info.default_python:
            default_py = uv_info.default_python
            console.print(
                f"  [{DIM}]{'Default Python:':<{w}}[/] [{SLEAP_CYAN}]{default_py}[/]"
            )
        else:
            console.print(
                f"  [{DIM}]{'Default Python:':<{w}}[/] [{DIM}](not configured)[/]"
            )
        if uv_info.resolved_python:
            resolved_py = uv_info.resolved_python
            console.print(
                f"  [{DIM}]{'Resolved Python:':<{w}}[/] [{SLEAP_CYAN}]{resolved_py}[/]"
            )

        pref = uv_info.python_preference or "managed"
        is_default = not uv_info.python_preference
        pref_display = f"{pref} [{DIM}](default)[/]" if is_default else pref
        console.print(f"  [{DIM}]{'Python Preference:':<{w}}[/] {pref_display}")

        res = uv_info.resolution_strategy or "highest"
        is_default = not uv_info.resolution_strategy
        res_display = f"{res} [{DIM}](default)[/]" if is_default else res
        console.print(f"  [{DIM}]{'Resolution:':<{w}}[/] {res_display}")

        idx = uv_info.index_strategy or "first-index"
        is_default = not uv_info.index_strategy
        idx_display = f"{idx} [{DIM}](default)[/]" if is_default else idx
        console.print(f"  [{DIM}]{'Index Strategy:':<{w}}[/] {idx_display}")

        pre = uv_info.prerelease or "if-necessary"
        is_default = not uv_info.prerelease
        pre_display = f"{pre} [{DIM}](default)[/]" if is_default else pre
        console.print(f"  [{DIM}]{'Prerelease:':<{w}}[/] {pre_display}")
        console.print()

    # -------------------------------------------------------------------------
    # Conda Information
    # -------------------------------------------------------------------------
    with console.status(f"[{DIM}]Checking conda...[/]", spinner="dots"):
        conda_info = get_conda_info_data()
    all_data["conda"] = conda_info

    if conda_info:
        console.print("[Conda]", style=f"bold {SLEAP_BLUE}")
        w = _DOCTOR_WIDTHS["conda"] + 1  # +1 for colon
        if conda_info.active:
            console.print(f"  [{DIM}]{'Status:':<{w}}[/] [{SLEAP_YELLOW}]ACTIVE[/]")
            console.print(f"  [{DIM}]{'Environment:':<{w}}[/] {conda_info.environment}")
            conda_prefix = conda_info.prefix
            console.print(
                f"  [{DIM}]{'Prefix:':<{w}}[/] [{SLEAP_CYAN}]{conda_prefix}[/]"
            )
        else:
            console.print(f"  [{DIM}]{'Status:':<{w}}[/] installed but not activated")
        if conda_info.version:
            console.print(f"  [{DIM}]{'Version:':<{w}}[/] {conda_info.version}")
        if conda_info.auto_activate_base is not None:
            status = "True" if conda_info.auto_activate_base else "False"
            color = SLEAP_RED if conda_info.auto_activate_base else SLEAP_GREEN
            console.print(
                f"  [{DIM}]{'auto_activate_base:':<{w}}[/] [{color}]{status}[/]"
            )
            if conda_info.auto_activate_base:
                console.print(
                    f"  [{SLEAP_YELLOW}]WARNING: auto_activate_base=True "
                    f"may interfere with uv[/]"
                )
                console.print(
                    f"  [{DIM}]Suggestion: "
                    f"conda config --set auto_activate_base false[/]"
                )
        if conda_info.sleap_packages:
            pkgs_str = ", ".join(conda_info.sleap_packages)
            console.print(
                f"  [{DIM}]{'SLEAP in conda:':<{w}}[/] [{SLEAP_RED}]{pkgs_str}[/]"
            )
            console.print(
                f"  [{SLEAP_YELLOW}]WARNING: Conda SLEAP packages "
                f"may conflict with uv/pip[/]"
            )
        console.print()

    # -------------------------------------------------------------------------
    # GPU / CUDA Information
    # -------------------------------------------------------------------------
    with console.status(f"[{DIM}]Checking GPU...[/]", spinner="dots"):
        nvidia_driver, system_cuda, gpus = get_nvidia_info()
    all_data["nvidia_driver"] = nvidia_driver
    all_data["system_cuda"] = system_cuda
    all_data["gpus"] = gpus

    with console.status(f"[{DIM}]Checking PyTorch...[/]", spinner="dots"):
        pytorch_version, pytorch_accelerator, pytorch_cuda = get_pytorch_info_detailed()
    all_data["pytorch_version"] = pytorch_version
    all_data["pytorch_accelerator"] = pytorch_accelerator
    all_data["pytorch_cuda"] = pytorch_cuda

    console.print("[GPU / CUDA]", style=f"bold {SLEAP_BLUE}")
    w = _DOCTOR_WIDTHS["gpu"] + 1  # +1 for colon
    if nvidia_driver:
        driver_str = nvidia_driver
        if system_cuda:
            driver_str += f" (CUDA {system_cuda})"
        console.print(
            f"  [{DIM}]{'NVIDIA Driver:':<{w}}[/] [{SLEAP_GREEN}]{driver_str}[/]"
        )
        for i, gpu in enumerate(gpus):
            console.print(
                f"  [{DIM}]{f'GPU {i}:':<{w}}[/] [{SLEAP_TEAL}]{gpu.name}[/] "
                f"([{SLEAP_GREEN}]{gpu.memory_free}[/] free / {gpu.memory_total})"
            )
    else:
        console.print(f"  [{DIM}]{'NVIDIA Driver:':<{w}}[/] Not detected")
    if pytorch_version:
        pt_str = f"v{pytorch_version}"
        if pytorch_accelerator == "cuda":
            pt_str += f" ([{SLEAP_GREEN}]CUDA {pytorch_cuda}[/])"
        elif pytorch_accelerator == "mps":
            pt_str += f" ([{SLEAP_GREEN}]MPS[/])"
        else:
            pt_str += f" ([{SLEAP_YELLOW}]CPU[/])"
        console.print(f"  [{DIM}]{'PyTorch:':<{w}}[/] [{SLEAP_TEAL}]{pt_str}[/]")
    else:
        console.print(f"  [{DIM}]{'PyTorch:':<{w}}[/] Not installed")
    console.print()

    # -------------------------------------------------------------------------
    # Package Versions
    # -------------------------------------------------------------------------
    with console.status(f"[{DIM}]Checking packages...[/]", spinner="dots"):
        packages = []
        for pkg_name in PACKAGES:
            pkg_info = get_detailed_package_info(pkg_name)
            if pkg_info:
                packages.append(pkg_info)
    all_data["packages"] = packages

    console.print("[Packages]", style=f"bold {SLEAP_BLUE}")
    w = max(len(pkg.name) for pkg in packages) + 1 if packages else 10  # +1 for colon
    for pkg in packages:
        source_color = (
            SLEAP_PURPLE
            if pkg.source == "editable"
            else SLEAP_ORANGE
            if pkg.source == "conda"
            else DIM
        )
        pkg_line = (
            f"  [{SLEAP_TEAL}]{(pkg.name + ':'):<{w}}[/] "
            f"[{SLEAP_GREEN}]v{pkg.version}[/] ([{source_color}]{pkg.source}[/])"
        )
        if pkg.editable and pkg.git_commit:
            git_info = f"git:{pkg.git_branch or 'HEAD'}@{pkg.git_commit}"
            if pkg.git_dirty:
                git_info += "*"
            pkg_line += f" [[{SLEAP_PURPLE}]{git_info}[/]]"
        console.print(pkg_line)
        if pkg.editable:
            console.print(f"  {'':<{w}} Location: [{SLEAP_CYAN}]{pkg.location}[/]")
    console.print()

    # -------------------------------------------------------------------------
    # CLI Binaries
    # -------------------------------------------------------------------------
    with console.status(f"[{DIM}]Checking CLI binaries...[/]", spinner="dots"):
        binaries = []
        bin_names = ["sleap", "sleap-nn", "sleap-nn-track", "sio"]
        for bin_name in bin_names:
            bin_info = get_binary_info(bin_name)
            if bin_info:
                binaries.append(bin_info)
        # Add ffmpeg binaries
        binaries.extend(get_ffmpeg_info())
    all_data["binaries"] = binaries

    if binaries:
        console.print("[CLI Binaries]", style=f"bold {SLEAP_BLUE}")
        w = _DOCTOR_WIDTHS["binaries"] + 1  # +1 for colon
        for binary in binaries:
            source_color = (
                SLEAP_TEAL
                if binary.source == "venv"
                else SLEAP_PURPLE
                if binary.source == "uv-tool"
                else SLEAP_ORANGE
            )
            console.print(f"  [{SLEAP_TEAL}]{binary.name}[/]:")
            bin_path = binary.path
            console.print(f"    [{DIM}]{'Path:':<{w}}[/] [{SLEAP_CYAN}]{bin_path}[/]")
            if binary.real_path != binary.path:
                real_path = binary.real_path
                console.print(
                    f"    [{DIM}]{'Real Path:':<{w}}[/] [{SLEAP_CYAN}]{real_path}[/]"
                )
            bin_src = binary.source
            console.print(
                f"    [{DIM}]{'Source:':<{w}}[/] [{source_color}]{bin_src}[/]"
            )
            if binary.python_path:
                py_path = binary.python_path
                console.print(
                    f"    [{DIM}]{'Python:':<{w}}[/] [{SLEAP_CYAN}]{py_path}[/]"
                )
        console.print()

    # -------------------------------------------------------------------------
    # PATH Analysis
    # -------------------------------------------------------------------------
    path_entries, path_conflicts = analyze_path()
    all_data["path_entries"] = path_entries
    all_data["path_conflicts"] = path_conflicts

    if path_conflicts:
        console.print("[PATH Conflicts]", style=f"bold {SLEAP_RED}")
        for conflict in path_conflicts:
            console.print(f"  [{SLEAP_YELLOW}]WARNING: {conflict}[/]")
        console.print()

    console.print("[PATH (relevant entries)]", style=f"bold {SLEAP_BLUE}")
    relevant_keywords = [
        "conda",
        "miniconda",
        "uv",
        ".local",
        "sleap",
        "python",
        "venv",
    ]
    for path in path_entries:
        if any(kw in path.lower() for kw in relevant_keywords):
            console.print(f"  [{SLEAP_CYAN}]{path}[/]")
    console.print()

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    output_path = None
    if output_file:
        if output_file == "auto":
            file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = Path(f"sleap-doctor-{file_timestamp}.txt")
        else:
            output_path = Path(output_file)

        output_text = _format_doctor_plain(all_data)
        output_path.write_text(output_text)

    console.print(f"[{DIM}]Copy this output when reporting issues at:[/]")
    console.print(f"[{SLEAP_BLUE}]https://github.com/talmolab/sleap/issues[/]")
    console.print()
    if output_path:
        console.print(f"[{SLEAP_GREEN}]Saved to:[/] [{SLEAP_TEAL}]{output_path}[/]")
    else:
        console.print(
            f"[bold {SLEAP_TEAL}]Tip:[/] [{DIM}]Use[/] "
            f"[{SLEAP_TEAL}]sleap doctor -o[/] "
            f"[{DIM}]to save diagnostics to a file[/]"
        )
    console.print()


def _doctor_json() -> None:
    """Output diagnostics as JSON."""
    import dataclasses
    import json

    from sleap.system_info import (
        PACKAGES,
        get_detailed_package_info,
        get_uv_info_data,
        get_conda_info_data,
        get_binary_info,
        get_nvidia_info,
        get_pytorch_info_detailed,
        get_memory_info,
        get_disk_info,
        get_ffmpeg_info,
        analyze_path,
    )

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        else:
            return obj

    ram_used, ram_avail, ram_total = get_memory_info()
    venv_path = os.environ.get("VIRTUAL_ENV", "") or sys.prefix
    disk_used, disk_avail, disk_total = get_disk_info(venv_path)

    nvidia_driver, system_cuda, gpus = get_nvidia_info()
    pytorch_version, pytorch_accelerator, pytorch_cuda = get_pytorch_info_detailed()

    packages = []
    for pkg_name in PACKAGES:
        pkg_info = get_detailed_package_info(pkg_name)
        if pkg_info:
            packages.append(pkg_info)

    binaries = []
    for bin_name in ["sleap", "sleap-nn", "sleap-nn-track", "sio"]:
        bin_info = get_binary_info(bin_name)
        if bin_info:
            binaries.append(bin_info)
    binaries.extend(get_ffmpeg_info())

    path_entries, path_conflicts = analyze_path()

    data = {
        "sleap_version": sleap.__version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "ram_used": ram_used,
            "ram_total": ram_total,
            "disk_used": disk_used,
            "disk_total": disk_total,
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": sys.prefix,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "uv": to_dict(get_uv_info_data()),
        "conda": to_dict(get_conda_info_data()),
        "gpu": {
            "nvidia_driver": nvidia_driver,
            "system_cuda": system_cuda,
            "gpus": to_dict(gpus),
        },
        "pytorch": {
            "version": pytorch_version,
            "accelerator": pytorch_accelerator,
            "cuda_version": pytorch_cuda,
        },
        "packages": to_dict(packages),
        "binaries": to_dict(binaries),
        "path_entries": path_entries,
        "path_conflicts": path_conflicts,
    }

    print(json.dumps(data, indent=2))


def _format_doctor_plain(data: dict) -> str:
    """Format diagnostic data as plain text (for file output)."""
    lines = []

    # Header
    lines.append("SLEAP System Diagnostics")
    lines.append("=" * 24)

    # Timestamp
    lines.append(f"Generated: {data.get('timestamp', 'N/A')}")
    lines.append("")

    # Platform
    p = data.get("platform", {})
    w = _DOCTOR_WIDTHS["platform"] + 1  # +1 for colon
    lines.append("[Platform]")
    lines.append(f"  {'OS:':<{w}} {p.get('os_name', '')} {p.get('os_release', '')}")
    lines.append(f"  {'Platform:':<{w}} {p.get('platform_full', '')}")
    lines.append(f"  {'Machine:':<{w}} {p.get('machine', '')}")
    if p.get("processor"):
        lines.append(f"  {'Processor:':<{w}} {p['processor']}")
    if p.get("ram_total"):
        lines.append(f"  {'RAM:':<{w}} {p['ram_used']} / {p['ram_total']}")
    if p.get("disk_total"):
        lines.append(f"  {'Disk:':<{w}} {p['disk_used']} / {p['disk_total']}")
    lines.append("")

    # Python
    py = data.get("python", {})
    w = _DOCTOR_WIDTHS["python"] + 1  # +1 for colon
    lines.append("[Python]")
    lines.append(f"  {'Version:':<{w}} {py.get('version', '')}")
    lines.append(f"  {'Executable:':<{w}} {py.get('executable', '')}")
    lines.append(f"  {'Prefix:':<{w}} {py.get('prefix', '')}")
    if py.get("virtual_env"):
        lines.append(f"  {'Virtual Env:':<{w}} {py['virtual_env']}")
    lines.append("")

    # UV
    uv = data.get("uv")
    if uv:
        w = _DOCTOR_WIDTHS["uv"] + 1  # +1 for colon
        lines.append("[UV]")
        lines.append(f"  {'Version:':<{w}} {uv.version}")
        lines.append(f"  {'Path:':<{w}} {uv.path}")
        lines.append(f"  {'Cache Dir:':<{w}} {uv.cache_dir}")
        lines.append(f"  {'Tool Dir:':<{w}} {uv.tool_dir}")
        lines.append(f"  {'Tool Bin Dir:':<{w}} {uv.tool_bin_dir}")
        lines.append(f"  {'Python Dir:':<{w}} {uv.python_dir}")
        if uv.installed_tools:
            lines.append(f"  {'Installed Tools:':<{w}} {', '.join(uv.installed_tools)}")
        lines.append("")
        # UV Config
        w = _DOCTOR_WIDTHS["uv_config"] + 1  # +1 for colon
        lines.append("[UV Config]")
        if uv.default_python:
            lines.append(f"  {'Default Python:':<{w}} {uv.default_python}")
        else:
            lines.append(f"  {'Default Python:':<{w}} (not configured)")
        if uv.resolved_python:
            lines.append(f"  {'Resolved Python:':<{w}} {uv.resolved_python}")
        pref = uv.python_preference or "managed"
        lines.append(
            f"  {'Python Preference:':<{w}} {pref}"
            f"{' (default)' if not uv.python_preference else ''}"
        )
        res = uv.resolution_strategy or "highest"
        lines.append(
            f"  {'Resolution:':<{w}} {res}"
            f"{' (default)' if not uv.resolution_strategy else ''}"
        )
        idx = uv.index_strategy or "first-index"
        lines.append(
            f"  {'Index Strategy:':<{w}} {idx}"
            f"{' (default)' if not uv.index_strategy else ''}"
        )
        pre = uv.prerelease or "if-necessary"
        lines.append(
            f"  {'Prerelease:':<{w}} {pre}{' (default)' if not uv.prerelease else ''}"
        )
        lines.append("")

    # Conda
    conda = data.get("conda")
    if conda:
        w = _DOCTOR_WIDTHS["conda"] + 1  # +1 for colon
        lines.append("[Conda]")
        if conda.active:
            lines.append(f"  {'Status:':<{w}} ACTIVE")
            lines.append(f"  {'Environment:':<{w}} {conda.environment}")
            lines.append(f"  {'Prefix:':<{w}} {conda.prefix}")
        else:
            lines.append(f"  {'Status:':<{w}} installed but not activated")
        if conda.version:
            lines.append(f"  {'Version:':<{w}} {conda.version}")
        if conda.auto_activate_base is not None:
            status = "True" if conda.auto_activate_base else "False"
            lines.append(f"  {'auto_activate_base:':<{w}} {status}")
            if conda.auto_activate_base:
                lines.append("  WARNING: auto_activate_base=True may interfere with uv")
                lines.append(
                    "  Suggestion: conda config --set auto_activate_base false"
                )
        if conda.sleap_packages:
            lines.append(
                f"  {'SLEAP in conda:':<{w}} {', '.join(conda.sleap_packages)}"
            )
            lines.append("  WARNING: Conda SLEAP packages may conflict with uv/pip")
        lines.append("")

    # GPU/CUDA
    w = _DOCTOR_WIDTHS["gpu"] + 1  # +1 for colon
    lines.append("[GPU / CUDA]")
    nvidia_driver = data.get("nvidia_driver", "")
    system_cuda = data.get("system_cuda", "")
    gpus = data.get("gpus", [])
    if nvidia_driver:
        driver_str = nvidia_driver
        if system_cuda:
            driver_str += f" (CUDA {system_cuda})"
        lines.append(f"  {'NVIDIA Driver:':<{w}} {driver_str}")
        for i, gpu in enumerate(gpus):
            gpu_label = f"GPU {i}:"
            gpu_mem = f"{gpu.memory_free} free / {gpu.memory_total}"
            lines.append(f"  {gpu_label:<{w}} {gpu.name} ({gpu_mem})")
    else:
        lines.append(f"  {'NVIDIA Driver:':<{w}} Not detected")
    pytorch_version = data.get("pytorch_version", "")
    pytorch_accelerator = data.get("pytorch_accelerator", "")
    pytorch_cuda = data.get("pytorch_cuda", "")
    if pytorch_version:
        pt_str = f"v{pytorch_version}"
        if pytorch_accelerator == "cuda":
            pt_str += f" (CUDA {pytorch_cuda})"
        elif pytorch_accelerator == "mps":
            pt_str += " (MPS)"
        else:
            pt_str += " (CPU)"
        lines.append(f"  {'PyTorch:':<{w}} {pt_str}")
    else:
        lines.append(f"  {'PyTorch:':<{w}} Not installed")
    lines.append("")

    # Packages
    packages = data.get("packages", [])
    lines.append("[Packages]")
    w = max(len(pkg.name) for pkg in packages) + 1 if packages else 10  # +1 for colon
    for pkg in packages:
        pkg_line = f"  {(pkg.name + ':'):<{w}} v{pkg.version} ({pkg.source})"
        if pkg.editable:
            if pkg.git_commit:
                git_info = f"git:{pkg.git_branch or 'HEAD'}@{pkg.git_commit}"
                if pkg.git_dirty:
                    git_info += "*"
                pkg_line += f" [{git_info}]"
            pkg_line += f"\n  {'':<{w}} Location: {pkg.location}"
        lines.append(pkg_line)
    lines.append("")

    # Binaries
    binaries = data.get("binaries", [])
    if binaries:
        w = _DOCTOR_WIDTHS["binaries"] + 1  # +1 for colon
        lines.append("[CLI Binaries]")
        for binary in binaries:
            lines.append(f"  {binary.name}:")
            lines.append(f"    {'Path:':<{w}} {binary.path}")
            if binary.real_path != binary.path:
                lines.append(f"    {'Real Path:':<{w}} {binary.real_path}")
            lines.append(f"    {'Source:':<{w}} {binary.source}")
            if binary.python_path:
                lines.append(f"    {'Python:':<{w}} {binary.python_path}")
        lines.append("")

    # PATH
    path_conflicts = data.get("path_conflicts", [])
    path_entries = data.get("path_entries", [])
    if path_conflicts:
        lines.append("[PATH Conflicts]")
        for conflict in path_conflicts:
            lines.append(f"  WARNING: {conflict}")
        lines.append("")

    lines.append("[PATH (relevant entries)]")
    relevant_keywords = [
        "conda",
        "miniconda",
        "uv",
        ".local",
        "sleap",
        "python",
        "venv",
    ]
    for path in path_entries:
        if any(kw in path.lower() for kw in relevant_keywords):
            lines.append(f"  {path}")

    return "\n".join(lines)


# =============================================================================
# sleap-io Commands (Inherited)
# =============================================================================

# Add wrapped sleap-io commands to the CLI group
cli.add_command(wrap_sio_command(sio_show), name="show")
cli.add_command(wrap_sio_command(sio_convert), name="convert")
cli.add_command(wrap_sio_command(sio_split), name="split")
cli.add_command(wrap_sio_command(sio_unsplit), name="unsplit")
cli.add_command(wrap_sio_command(sio_merge), name="merge")
cli.add_command(wrap_sio_command(sio_filenames), name="filenames")
cli.add_command(wrap_sio_command(sio_render), name="render")
cli.add_command(wrap_sio_command(sio_fix), name="fix")
cli.add_command(wrap_sio_command(sio_embed), name="embed")
cli.add_command(wrap_sio_command(sio_unembed), name="unembed")
cli.add_command(wrap_sio_command(sio_trim), name="trim")
cli.add_command(wrap_sio_command(sio_reencode), name="reencode")
cli.add_command(wrap_sio_command(sio_transform), name="transform")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    cli()
