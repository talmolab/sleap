#!/usr/bin/env python3
"""Map changed files from a PR to their corresponding docs preview URLs.

This script is used by the PR preview workflow to generate direct links to
the specific documentation pages that changed.

Usage:
    python scripts/get_changed_docs_urls.py <base_url> [file1] [file2] ...

Example:
    python scripts/get_changed_docs_urls.py <base_url> docs/install.md

Output:
    JSON object with categorized changes::

        {"pages": [...], "api_affected": true, "config_changed": true}
"""

import json
import sys
from pathlib import Path


def file_to_docs_url(file_path: str, base_url: str) -> dict | None:
    """Convert a docs file path to its preview URL.

    Args:
        file_path: Path to the changed file (e.g., "docs/guides/overview.md")
        base_url: Base URL for the preview (e.g., "https://docs.sleap.ai/pr/123/")

    Returns:
        Dict with file, url, and title if the file is a docs page, else None.
    """
    path = Path(file_path)

    # Only process docs markdown files
    if not file_path.startswith("docs/") or path.suffix != ".md":
        return None

    # Get the relative path within docs/
    rel_path = path.relative_to("docs")

    # Convert to URL path following MkDocs conventions:
    # - index.md -> /
    # - foo.md -> /foo/
    # - foo/bar.md -> /foo/bar/
    # - foo/index.md -> /foo/
    if rel_path.name == "index.md":
        # index.md files map to their parent directory
        url_path = str(rel_path.parent)
        if url_path == ".":
            url_path = ""
    else:
        # Other .md files map to a directory with their stem
        url_path = str(rel_path.with_suffix(""))

    # Ensure base_url ends with /
    if not base_url.endswith("/"):
        base_url += "/"

    # Build the full URL
    if url_path:
        full_url = f"{base_url}{url_path}/"
    else:
        full_url = base_url

    # Generate a friendly title from the file name
    title = rel_path.name

    return {"file": file_path, "url": full_url, "title": title}


def categorize_changes(files: list[str], base_url: str) -> dict:
    """Categorize changed files into docs pages and other changes.

    Args:
        files: List of changed file paths
        base_url: Base URL for the preview

    Returns:
        Dict with pages, api_affected, and config_changed keys.
    """
    pages = []
    api_affected = False
    config_changed = False

    for file_path in files:
        # Check for API-affecting changes (sleap package)
        if file_path.startswith("sleap/"):
            api_affected = True
            continue

        # Check for config changes
        if file_path in ("mkdocs.yml", ".github/workflows/pr-preview.yml"):
            config_changed = True
            continue

        # Try to map to a docs URL
        page_info = file_to_docs_url(file_path, base_url)
        if page_info:
            pages.append(page_info)

    return {
        "pages": pages,
        "api_affected": api_affected,
        "config_changed": config_changed,
    }


def format_markdown_links(result: dict, base_url: str) -> str:
    """Format the categorized changes as markdown links.

    Args:
        result: Output from categorize_changes()
        base_url: Base URL for the preview

    Returns:
        Markdown formatted string with links
    """
    lines = []

    # Add page links
    if result["pages"]:
        lines.append("**Changed pages:**")
        for page in result["pages"]:
            lines.append(f"- [{page['title']}]({page['url']})")

    # Add API reference note
    if result["api_affected"]:
        lines.append("")
        lines.append(
            f"**API changes detected** - [View API Reference]({base_url}api/)"
        )

    # Add config change note
    if result["config_changed"]:
        lines.append("")
        lines.append("_Configuration files changed - full rebuild performed._")

    # If no specific pages but config/API changed, just note it
    if not result["pages"] and not result["api_affected"] and result["config_changed"]:
        lines.append("_Only configuration files changed._")

    return "\n".join(lines) if lines else "_No documentation pages directly changed._"


def main():
    """CLI entry point for the script."""
    if len(sys.argv) < 2:
        print(
            "Usage: get_changed_docs_urls.py <base_url> [file1] [file2] ...",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = sys.argv[1]
    files = sys.argv[2:] if len(sys.argv) > 2 else []

    result = categorize_changes(files, base_url)

    # Output as JSON for workflow parsing
    output = {
        **result,
        "markdown": format_markdown_links(result, base_url),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
