# orchestration/ (Dagster) — added last

Dagster code location `flight_delays`: the end-to-end DAG, wrapping the
EXISTING standalone entry points — nothing reimplemented.

```
bts_bronze ──► bts_external_table ──► [dbt: seeds → silver → gold] ──► ml_training
  (ingestion.bts.run_ingestion)       (dagster-dbt `dbt build` over        (ml.train.run_training —
                                       the existing project; dbt            the real path: audit gate,
                                       tests = asset checks)                is_training_row split,
                                                                            determinism sort, artifact
                                                                            contract)
```

- Source→asset lineage is wired via `meta.dagster.asset_key` on the
  `bts_on_time_performance` source in `_bronze__sources.yml`.
- **Checks**: a blocking asset check on `bts_bronze` (every window month must
  have CSV + manifest — failure blocks everything downstream), dbt tests
  surface as asset checks on their models, and an artifact-contract check on
  `ml_training`.
- **Schedule** `monthly_refresh_schedule` on job `monthly_refresh`
  (cron `0 6 10 * *`, stopped by default):
  BTS publishes ~2–3 months in arrears; monthly on the 10th picks up newly
  published months. With the fixed 2022–2024 window the bronze step is an
  idempotent skip and the run refreshes dbt + retrains.
- Group `reference` (`airports_seed_csv`, `holidays_seed_csv`) regenerates
  the dbt seed CSVs — deliberately outside the monthly job; refresh manually
  and commit the deterministic diff.
- Assets import `ingestion`/`ml` lazily inside their bodies, so Definitions
  load is light and credential-free — CI runs `dagster definitions validate`
  with placeholder env values.

## Local development

```
uv sync --all-extras
uv run --all-extras dagster dev                 # UI at http://localhost:3000
uv run --all-extras dagster definitions validate
uv run --all-extras dagster job execute -j monthly_refresh -m orchestration.definitions
```

## Cheapest GCP deployment (outline — not deployed)

The cadence is one run per month, which makes a standing webserver wasteful:

1. **Recommended: Cloud Run Job + Cloud Scheduler (~$0/month idle).** Build
   the repo into a container (uv image, `--all-extras`); a Cloud Run **Job**
   runs `dagster job execute -j monthly_refresh -m orchestration.definitions`
   (in-process, no daemon needed — Cloud Scheduler replaces the Dagster
   schedule daemon; set the Job's service account and drop the local-ADC
   `.env` for env vars on the Job). Costs pennies per run, nothing idle.
   Definitions load re-parses the dbt manifest on start (seconds), so a
   baked image can never run against a stale manifest; alternatively bake
   one at build time with `dagster-dbt project prepare-and-package`.
   The Dagster UI is then dev-only (`dagster dev` locally against the same
   code), which is the honest tradeoff at this budget.
2. **Alternative: one e2-small VM (~$12–15/month)** running
   `dagster-webserver` + `dagster-daemon` under systemd with `DAGSTER_HOME`
   on the boot disk — persistent UI, run history, and the in-product
   schedule, at the cost of an always-on instance.

Both use ADC via the attached service account — no key files, consistent
with CLAUDE.md §2.
