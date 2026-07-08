# seeds/

Small static reference CSVs loaded into the `bronze` dataset by `dbt seed`.

**Decision (2026-07): `airports` and `holidays` are dbt seeds** — not bronze
CSV + external tables. They were briefly wired as both; the source declarations
have been removed. Rationale: they are tiny, static, and belong in version
control where changes are reviewable in PRs; `dbt seed` + `{{ ref(...) }}`
keeps lineage inside dbt and lets schema tests run on them; and the GCS
year/month partition layout doesn't fit non-temporal reference data anyway.
They must **never** be redeclared as sources in `_bronze__sources.yml` —
a seed refresh over a same-named external table would clobber it, and
`source()` over a dbt-managed table breaks lineage.

Planned seeds:

| Seed CSV       | Content                                                      | Referenced as           |
|----------------|--------------------------------------------------------------|-------------------------|
| `airports.csv` | Airport coords + timezone (trimmed: US airports, needed columns only) | `{{ ref('airports') }}` |
| `holidays.csv` | Generated US holiday calendar 2022–2024 (`holidays` library) | `{{ ref('holidays') }}` |

Keep seeds small (thousands of rows, not tens of thousands — `dbt seed` loads
row-by-row, and the full OurAirports dump is too big; trim it). If a reference
file ever outgrows that or starts changing regularly, move that entity to the
GCS bronze + external-table path and switch `ref()` → `source()` in its
staging model.

Seed CSVs here are intentionally **not** git-ignored (see `.gitignore`).
Large/raw inputs do not belong here — those land as bronze CSV in GCS.
