"""MkDocs hook to publish markdown sources alongside HTML pages.

This hook does two things:
1. Copies markdown source files to the build output for direct access via URLs like:
       https://docs.sleap.ai/{version}/guides/overview.md
2. Injects a "View page source" link at the top of each page pointing to the local
   markdown file (useful for LLM context).

The hook runs during the build process to modify pages and copy source files.
"""

import shutil
from pathlib import Path


def on_page_content(html: str, page, config, files) -> str:  # noqa: ARG001
    """Inject a link to the markdown source at the top of the page.

    Args:
        html: The rendered HTML content of the page.
        page: The MkDocs page object with source file info.
        config: MkDocs config object.
        files: Collection of files in the docs.

    Returns:
        Modified HTML with the source link injected.
    """
    # Get the source file path relative to docs_dir
    src_path = Path(page.file.src_path)  # e.g., "guides/overview.md" or "index.md"

    # Calculate a relative URL to the markdown source
    # MkDocs generates directory-style URLs:
    #   - install.md -> /install/index.html
    #   - guides/overview.md -> /guides/overview/index.html
    #   - guides/index.md -> /guides/index.html
    #   - index.md -> /index.html
    # The markdown files are copied to the same relative path in site_dir:
    #   - install.md -> /install.md
    #   - guides/overview.md -> /guides/overview.md
    # So we use relative paths that work from the page's URL:
    if src_path.name == "index.md":
        # index.md pages: source is in the same directory
        md_url = "index.md"
    else:
        # Non-index pages: source is one level up (../filename.md)
        md_url = f"../{src_path.name}"

    # Create a subtle link block at the top of the content
    # SVG is a document icon from Material Design Icons
    svg_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 '
        '2-2V8l-6-6m4 18H6V4h7v5h5v11M13 9V3.5L18.5 9H13Z"/></svg>'
    )
    source_link = (
        f'<div class="md-source-file">'
        f'<a href="{md_url}" title="View markdown source" class="md-icon">'
        f"{svg_icon}</a></div>\n"
    )

    return source_link + html


def on_post_build(config, **kwargs):  # noqa: ARG001
    """Copy markdown source files to the build output after site generation.

    Args:
        config: MkDocs config object containing docs_dir and site_dir paths.
        **kwargs: Additional keyword arguments passed by MkDocs (unused).
    """
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])

    # Track how many files were copied for logging
    copied_count = 0

    # Find all markdown files in the docs directory
    for md_file in docs_dir.rglob("*.md"):
        # Get the relative path from docs_dir
        rel_path = md_file.relative_to(docs_dir)

        # Determine destination path in site_dir
        dest_path = site_dir / rel_path

        # Ensure the destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the markdown file
        shutil.copy2(md_file, dest_path)
        copied_count += 1

    print(f"Copied {copied_count} markdown source files to output directory")
