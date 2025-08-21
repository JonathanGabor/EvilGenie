"""
Utilities for constructing hardened environments for subprocesses.

The goal is to pass only the minimal, allow-listed environment variables
needed by child processes, reducing the risk of secret leakage.
"""

from __future__ import annotations

import os
from typing import Iterable, Dict, Optional


DEFAULT_LOCALE = "C.UTF-8"


def build_subprocess_env(required_vars: Optional[Iterable[str]] = None,
                         extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create a minimal environment for subprocesses.

    - Preserves essential runtime vars: PATH, HOME (if present), locale settings
    - Includes common proxy vars if present (HTTP_PROXY/HTTPS_PROXY/NO_PROXY)
    - Optionally includes an allow-list of additional variables (e.g., API keys)

    Args:
        required_vars: Iterable of env var names to copy from the current env
        extra_env: Additional explicit env variables to set

    Returns:
        Dict suitable for passing to subprocess env=...
    """
    env: Dict[str, str] = {}

    # Essentials
    env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]

    # Locale to avoid encoding issues
    env["LANG"] = os.environ.get("LANG", DEFAULT_LOCALE)
    env["LC_ALL"] = os.environ.get("LC_ALL", env["LANG"])

    # Make Python output unbuffered for more predictable logs
    env["PYTHONUNBUFFERED"] = "1"

    # Respect proxy configuration if present
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        if proxy_var in os.environ:
            env[proxy_var] = os.environ[proxy_var]

    # Allow-list variables explicitly requested by callers
    if required_vars:
        for key in required_vars:
            if key in os.environ:
                env[key] = os.environ[key]

    # Explicit overrides/additions
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v is not None})

    return env


def provider_env_keys(provider: str) -> list[str]:
    """Return known environment vars needed for a given provider/CLI.

    This is intentionally conservative to avoid leaking unrelated secrets.
    """
    p = (provider or "").lower()
    if p in ("openai", "codex"):
        return ["OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_BASE_URL", "OPENAI_PROJECT"]
    if p in ("anthropic", "claude"):
        return ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
    if p in ("google", "gemini"):
        return ["GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"]
    return []

