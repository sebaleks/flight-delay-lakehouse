"""Load the gold ML feature mart from BigQuery — the same gold layer the
dashboard consumes; nothing is recomputed or duplicated here.

Downcasts to float32/int8/category so the full 20.2M-row mart fits
comfortably in memory. NaNs in hist_* (entities new in the test window) and
weather (missing prior-day observation) are preserved — XGBoost consumes them
natively; the logistic-regression pipeline imputes with TRAIN-set medians
only (never test statistics).
"""

from __future__ import annotations

import logging

import pandas as pd
from google.cloud import bigquery

from ingestion.config import require_env
from ml import features as f

log = logging.getLogger("ml.data")

MART_TABLE = "ml_flight_features"


def load_mart() -> tuple[pd.DataFrame, bigquery.Client, str]:
    project = require_env("GCP_PROJECT_ID")
    dataset = require_env("BQ_GOLD_DATASET")
    bq = bigquery.Client(project=project)

    cols = ", ".join([*f.FEATURES, "flight_date", f.SPLIT_COL, *f.LABELS])
    log.info("loading %s.%s (%d columns) ...", dataset, MART_TABLE, len(f.FEATURES) + 4)
    df = bq.query(f"select {cols} from `{project}.{dataset}.{MART_TABLE}`").to_dataframe(
        create_bqstorage_client=True
    )
    log.info("loaded %s rows", f"{len(df):,}")

    for c in f.CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    for c in f.NUMERIC_FEATURES:
        if str(df[c].dtype) == "boolean":
            # nullable pandas boolean -> float32 keeps any NA representable
            df[c] = df[c].astype("Float32").astype("float32")
        else:
            df[c] = pd.to_numeric(df[c]).astype("float32")
    df[f.SPLIT_COL] = df[f.SPLIT_COL].astype(bool)
    df["label_arr_del15"] = df["label_arr_del15"].astype("int8")
    df["label_arr_delay_minutes"] = df["label_arr_delay_minutes"].astype("float32")
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df, bq, dataset
