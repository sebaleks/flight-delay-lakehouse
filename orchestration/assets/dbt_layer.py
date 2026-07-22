"""dbt assets: every seed, model, and test of the existing dbt project,
loaded from its manifest via dagster-dbt — nothing reimplemented. dbt tests
surface as Dagster asset checks (dagster-dbt default), so a failing test
fails its asset and blocks downstream materialization in the run.
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module —
# PEP 563 string annotations break dagster-dbt's inspection of the `context`
# parameter annotation on @dbt_assets functions (identity check against the
# real AssetExecutionContext class fails on the string form).
import os
from pathlib import Path

from dagster import AssetExecutionContext
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

REPO_ROOT = Path(__file__).resolve().parents[2]

dbt_project = DbtProject(
    project_dir=REPO_ROOT / "dbt",
    profiles_dir=REPO_ROOT / "dbt",
)
# `dagster dev` re-prepares the manifest itself on load. Every OTHER context
# (definitions validate, job execute, asset materialize, a deployed image)
# re-parses UNCONDITIONALLY: an exists-only check would silently load a STALE
# manifest after dbt project edits, and the runtime `dbt build` would then
# emit result events for nodes the definition-time manifest has never seen
# (KeyError mid-stream) or leave phantom assets. A parse costs seconds.
dbt_project.prepare_if_dev()
if not os.getenv("DAGSTER_IS_DEV_CLI"):
    dbt_project.preparer.prepare(dbt_project)


@dbt_assets(manifest=dbt_project.manifest_path)
def flight_delays_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    """`dbt build` over the whole project: seeds -> silver -> gold (star,
    marts, dashboard views, ml_flight_features), with tests inline."""
    yield from dbt.cli(["build"], context=context).stream()
