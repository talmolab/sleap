"""Utilities for WandB integration in SLEAP GUI."""

import os
from pathlib import Path
from typing import Optional, Tuple


def check_wandb_login_status() -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if wandb is logged in and return status info.

    This function checks for cached credentials without calling wandb.login(),
    which would be slow and print messages to stdout.

    Returns:
        Tuple of (is_logged_in, auth_source, username).
        auth_source describes how the user is authenticated (env var, netrc, etc.)
        username is the logged-in username if available, else None.
    """
    # Check environment variable first (fastest)
    if os.environ.get("WANDB_API_KEY"):
        return True, "WANDB_API_KEY environment variable", None

    # Check netrc file for cached credentials (wandb stores keys here)
    try:
        import netrc

        # Try both .netrc and _netrc (Windows can use either)
        for netrc_name in (".netrc", "_netrc"):
            netrc_path = Path.home() / netrc_name
            if netrc_path.exists():
                nrc = netrc.netrc(str(netrc_path))
                auth = nrc.authenticators("api.wandb.ai")
                if auth and auth[2]:  # (login, account, password) - password is API key
                    username = auth[0] if auth[0] else None
                    return True, "cached credentials", username
    except Exception:
        pass

    return False, None, None


def get_wandb_api_key_help_text(is_logged_in: bool, auth_source: Optional[str]) -> str:
    """Generate help text for the WandB API key field based on login status.

    Args:
        is_logged_in: Whether wandb is currently authenticated.
        auth_source: Description of how the user is authenticated.

    Returns:
        Help text string for the API key field.
    """
    base_help = (
        "WandB API Key. From https://wandb.ai/authorize. "
        "You could also set it in your terminal by exporting the WANDB_API_KEY "
        "environment variable or `wandb login` in your shell."
    )

    if is_logged_in and auth_source:
        return (
            f"{base_help} (Already authenticated via {auth_source} - can leave blank)"
        )

    return base_help
