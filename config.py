"""Configuration settings for FinSight.

Centralizes API credentials, supported Gemini models, and system defaults.
"""

from __future__ import annotations

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Supported Gemini models in FinSight
SUPPORTED_GEMINI_MODELS: List[str] = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

DEFAULT_PRIMARY_MODEL: str = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-004"

# API Key
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Primary model resolution with safe fallback
_env_model = os.getenv("GEMINI_MODEL", DEFAULT_PRIMARY_MODEL)
GEMINI_MODEL: str = (
    _env_model if _env_model in SUPPORTED_GEMINI_MODELS else DEFAULT_PRIMARY_MODEL
)

# Fallback model resolution with safe fallback
_env_fallback = os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
GEMINI_FALLBACK_MODEL: str = (
    _env_fallback if _env_fallback in SUPPORTED_GEMINI_MODELS else DEFAULT_FALLBACK_MODEL
)

# Embedding model
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_effective_fallback_model(primary_model: str) -> str:
    """Return a valid fallback model distinct from the active primary model.

    Args:
        primary_model: The currently selected primary Gemini model name.

    Returns:
        A supported fallback model identifier distinct from primary_model.
    """
    if primary_model == DEFAULT_FALLBACK_MODEL:
        # If the primary is already gemini-3.5-flash-lite, fallback to gemini-2.5-flash
        return "gemini-2.5-flash"

    if GEMINI_FALLBACK_MODEL != primary_model and GEMINI_FALLBACK_MODEL in SUPPORTED_GEMINI_MODELS:
        return GEMINI_FALLBACK_MODEL

    for model in SUPPORTED_GEMINI_MODELS:
        if model != primary_model:
            return model

    return DEFAULT_FALLBACK_MODEL


# Reference for self-referential helper calls if needed
import sys
config_module = sys.modules.get(__name__, sys.modules.get("config"))
