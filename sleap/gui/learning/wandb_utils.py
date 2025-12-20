"""Utilities for WandB integration in SLEAP GUI."""

import os
from typing import Optional, Tuple


def check_wandb_login_status() -> Tuple[bool, Optional[str]]:
    """Check if wandb is logged in and return status info.

    Returns:
        Tuple of (is_logged_in, status_message).
        status_message describes how the user is authenticated (env var, cached, etc.)
    """
    # Check environment variable first
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return True, "WANDB_API_KEY environment variable"

    # Try to check wandb cached credentials
    try:
        import wandb

        # wandb.login(verify=True) returns True if already logged in
        # This checks cached credentials without prompting
        if wandb.login(verify=True):
            return True, "cached wandb credentials"
    except Exception:
        pass

    return False, None


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
        return f"{base_help} (Already authenticated via {auth_source} - can leave blank)"

    return base_help
