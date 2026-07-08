# ingestion/

Python extract/load into the **bronze** layer. Bronze is immutable raw **CSV in
GCS**, partitioned by year/month:

```
gs://$GCS_BUCKET/bronze/<source>/year=<YYYY>/month=<MM>/*.csv
```

Config/auth: all GCP identifiers from env vars (`.env`); auth via ADC
(`gcloud auth application-default login`). Not implemented yet — planned modules:

| Module (planned)   | Source                                   | Destination                          |
|--------------------|------------------------------------------|--------------------------------------|
| `bts.py`           | BTS On-Time Performance 2022–2024        | bronze CSV in GCS (partitioned)      |
| `airports.py`      | Airport coordinates + timezone reference | `dbt/seeds/airports.csv` (dbt seed)  |
| `holidays_cal.py`  | Generated US holiday calendar (`holidays`)| `dbt/seeds/holidays.csv` (dbt seed)  |
| `noaa.py`          | NOAA GSOD                                 | *(read in place from public data)*   |

After bronze lands, an external table in the `bronze` dataset exposes the BTS
CSVs to BigQuery for dbt. NOAA GSOD is read directly from
`bigquery-public-data.noaa_gsod` and is **not** copied into bronze. Airports
and holidays are generated/trimmed into `dbt/seeds/` and loaded by `dbt seed`
(decision in `dbt/seeds/README.md`), so their modules write seed CSVs, not GCS.

Install deps: `uv sync --extra ingestion`.
