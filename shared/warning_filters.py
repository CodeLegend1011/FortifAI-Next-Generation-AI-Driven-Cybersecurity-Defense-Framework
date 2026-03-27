"""Suppress noisy third-party FutureWarnings from Google client libraries."""

from __future__ import annotations

import warnings


def silence_google_sdk_future_warnings() -> None:
    """
    Call before ``import google.generativeai`` or other Google SDK entry points.

    Quiets:
    - google.api_core Python version / EOL notices
    - Deprecated ``google.generativeai`` package notice (migration to google.genai)
    """
    # Issued from inside google.api_core (module name may vary by version)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.api_core.*")
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"(?s).*which Google will stop supporting.*google\.api_core.*",
    )
    # Often attributed to the caller file; match message body
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"(?s).*All support for the `google\.generativeai` package.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"(?s).*switch to the `google\.genai` package.*",
    )
