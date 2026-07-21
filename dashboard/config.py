"""Dashboard config: GCP identifiers from env (never hardcoded, per CLAUDE.md §2).

Reuses ``ingestion.config`` for env loading so the whole repo resolves project /
dataset the same way. Auth is ADC — no key files.
"""

from __future__ import annotations

from ingestion.config import require_env


def gcp_project() -> str:
    return require_env("GCP_PROJECT_ID")


def gold_dataset() -> str:
    return require_env("BQ_GOLD_DATASET")


def fq_view(view: str) -> str:
    """Fully-qualified `project.dataset.view` for a gold dashboard view."""
    return f"`{gcp_project()}.{gold_dataset()}.{view}`"
