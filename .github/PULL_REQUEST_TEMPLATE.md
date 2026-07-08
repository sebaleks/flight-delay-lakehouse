## Summary

<!-- What does this PR do, and why? One or two sentences. -->

## Changes

-

## Checklist

- [ ] No credentials, keys, `.env`, or data files added (data lives in GCS/BigQuery, never git)
- [ ] Config flows through env vars — no hardcoded GCP project/bucket/dataset ids
- [ ] dbt changes: `uv run dbt parse` passes and models stay in their layer (silver cleans, gold serves)
- [ ] ML/feature changes: every new predictor is knowable **before departure** (CLAUDE.md §9)
- [ ] CLAUDE.md updated if an architectural decision changed
