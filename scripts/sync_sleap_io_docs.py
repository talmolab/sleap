#!/usr/bin/env python3
"""Sync and transform sleap-io CLI docs for SLEAP documentation.

This script:
1. Reads CLI documentation from sleap-io
2. Transforms `sio` -> `sleap` in examples
3. Extracts individual command sections
4. Writes to docs/cli/*.md with SLEAP branding

Run during docs build via gen-files plugin or as a pre-build step.

Usage:
    python scripts/sync_sleap_io_docs.py

    # Or specify paths explicitly:
    python scripts/sync_sleap_io_docs.py --input ../sleap-io/docs/cli.md --output docs/cli/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def get_sleap_io_version() -> str:
    """Get the installed sleap-io version."""
    try:
        import sleap_io

        return sleap_io.__version__
    except ImportError:
        return "unknown"


def transform_sio_to_sleap(content: str) -> str:
    """Transform sleap-io examples to use sleap command."""
    # Replace command invocations in code blocks and inline
    replacements = [
        (r"\$ sio ", "$ sleap "),
        (r"`sio ", "`sleap "),
        (r"sio show", "sleap show"),
        (r"sio convert", "sleap convert"),
        (r"sio split", "sleap split"),
        (r"sio filenames", "sleap filenames"),
        (r"sio render", "sleap render"),
        (r"sio --", "sleap --"),
        # Replace uvx sleap-io with sleap
        (r"uvx sleap-io ", "sleap "),
        # Replace standalone sio in examples
        (r"^sio ", "sleap ", re.MULTILINE),
    ]

    for pattern, replacement, *flags in replacements:
        flag = flags[0] if flags else 0
        content = re.sub(pattern, replacement, content, flags=flag)

    return content


def extract_command_section(content: str, command: str) -> str | None:
    """Extract a specific command section from the full CLI docs.

    Looks for sections like:
        ### `sio show` - Inspect Labels and Video Files
        ...content...
        ### `sio convert` - ...  (or end of file)
    """
    # Pattern to match the command header and capture until the next command header
    # The header format is: ### `sio <command>` - <description>
    pattern = rf"(### `sio {command}` - [^\n]+\n)(.*?)(?=### `sio |\Z)"

    match = re.search(pattern, content, re.DOTALL)
    if match:
        header = match.group(1)
        body = match.group(2).strip()

        # Transform the header to use sleap
        header = header.replace(f"`sio {command}`", f"`sleap {command}`")

        return header + "\n" + body

    return None


def create_command_doc(command: str, section: str, version: str) -> str:
    """Create a full documentation page for a command."""
    # Create the page title from the command name
    title = f"sleap {command}"

    header = f"""# {title}

!!! info "Powered by sleap-io"
    This command is provided by [sleap-io](https://io.sleap.ai) v{version}.
    For the most detailed and up-to-date documentation, see [io.sleap.ai/cli](https://io.sleap.ai/cli/).

"""

    # Remove the ### header from the section since we have our own title
    section = re.sub(r"^### `sleap \w+` - [^\n]+\n", "", section)

    return header + section


def sync_docs(
    sleap_io_docs_path: Path,
    output_dir: Path,
    commands: list[str] | None = None,
) -> None:
    """Main sync function.

    Args:
        sleap_io_docs_path: Path to sleap-io docs directory or cli.md file
        output_dir: Output directory for generated docs
        commands: List of commands to sync (default: all)
    """
    # Handle both directory and file paths
    if sleap_io_docs_path.is_dir():
        cli_md = sleap_io_docs_path / "cli.md"
    else:
        cli_md = sleap_io_docs_path

    if not cli_md.exists():
        print(f"Warning: {cli_md} not found, skipping sleap-io doc sync")
        return

    print(f"Reading: {cli_md}")
    content = cli_md.read_text()

    # Get version for header
    version = get_sleap_io_version()
    print(f"sleap-io version: {version}")

    # Default commands to sync
    if commands is None:
        commands = ["show", "convert", "split", "filenames", "render"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for cmd in commands:
        # Extract section BEFORE transformation (pattern uses 'sio')
        section = extract_command_section(content, cmd)
        if section:
            # Transform sio -> sleap AFTER extraction
            section = transform_sio_to_sleap(section)
            doc = create_command_doc(cmd, section, version)
            output_file = output_dir / f"{cmd}.md"
            output_file.write_text(doc)
            print(f"Generated: {output_file}")
        else:
            print(f"Warning: Could not extract section for '{cmd}'")


def main():
    parser = argparse.ArgumentParser(
        description="Sync sleap-io CLI docs to SLEAP documentation"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("../sleap-io/docs"),
        help="Path to sleap-io docs directory or cli.md file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("docs/cli"),
        help="Output directory for generated docs",
    )
    parser.add_argument(
        "--commands",
        "-c",
        nargs="+",
        default=None,
        help="Commands to sync (default: show, convert, split, filenames, render)",
    )

    args = parser.parse_args()
    sync_docs(args.input, args.output, args.commands)


if __name__ == "__main__":
    main()
