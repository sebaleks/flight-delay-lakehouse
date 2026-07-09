"""Shared config for ingestion jobs: env loading and GCP identifiers.

All GCP identifiers come from env vars (loaded from the repo-root .env when
present) per CLAUDE.md — never hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Required env var {name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
