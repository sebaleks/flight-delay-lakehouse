"""Build the dbt airports seed from OpenFlights airports.dat.

Downloads the public OpenFlights airport database (ODbL license), trims it to
US airports (incl. territories) that have an IATA code, and writes
``dbt/seeds/airports.csv`` deterministically sorted so re-runs produce stable
git diffs. Loaded into the bronze dataset by ``dbt seed`` and referenced via
``{{ ref('airports') }}`` — decision and tradeoff in dbt/seeds/README.md.

Run:
    uv run --extra ingestion python -m ingestion.airports
"""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path

from ingestion.config import REPO_ROOT
from ingestion.util import download_with_retries, setup_logging

log = logging.getLogger("ingestion.airports")

OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
SEED_PATH = REPO_ROOT / "dbt" / "seeds" / "airports.csv"

# airports.dat has no header; documented field order:
OPENFLIGHTS_COLUMNS = [
    "airport_id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "latitude",
    "longitude",
    "altitude_ft",
    "utc_offset",
    "dst",
    "tz",
    "type",
    "source",
]

# BTS covers US states plus territories; OpenFlights lists territories as
# their own "country".
US_COUNTRIES = {
    "United States",
    "Puerto Rico",
    "Guam",
    "Virgin Islands",
    "American Samoa",
    "Northern Mariana Islands",
}

OUTPUT_COLUMNS = [
    "iata",
    "icao",
    "name",
    "city",
    "country",
    "latitude",
    "longitude",
    "elevation_ft",
    "tz",
]

# Upstream rows whose OpenFlights tz is \N but the airport is BTS-active —
# a NULL tz would break local-time features downstream (silver tests pin
# tz not_null for flown airports).
TZ_BACKFILL = {
    "BIH": "America/Los_Angeles",  # Eastern Sierra Regional, Bishop CA
}

# BTS-active airports missing from OpenFlights (stale for post-2017 openings).
# Coordinates/elevation from OurAirports; tz curated. Gap-fill only: an
# upstream OpenFlights row with the same IATA code takes precedence.
SUPPLEMENTS = [
    {
        "iata": "XWA",
        "icao": "KXWA",
        "name": "Williston Basin International Airport",
        "city": "Williston",
        "country": "United States",
        "latitude": "48.260863",
        "longitude": "-103.75116",
        "elevation_ft": "2344",
        "tz": "America/Chicago",
    },
    {
        "iata": "EAR",
        "icao": "KEAR",
        "name": "Kearney Regional Airport",
        "city": "Kearney",
        "country": "United States",
        "latitude": "40.727001",
        "longitude": "-99.006798",
        "elevation_ft": "2131",
        "tz": "America/Chicago",
    },
    {
        "iata": "IFP",
        "icao": "KIFP",
        "name": "Laughlin Bullhead International Airport",
        "city": "Bullhead City",
        "country": "United States",
        "latitude": "35.154726",
        "longitude": "-114.559322",
        "elevation_ft": "701",
        "tz": "America/Phoenix",
    },
]


def is_us_airport(row: dict[str, str]) -> bool:
    iata = row["iata"]
    return (
        row["country"] in US_COUNTRIES
        and row["type"] == "airport"
        and iata not in ("", "\\N")
        and len(iata) == 3
    )


def build_seed(raw_path: Path) -> list[dict[str, str]]:
    with open(raw_path, encoding="utf-8") as f:
        rows = [dict(zip(OPENFLIGHTS_COLUMNS, r, strict=True)) for r in csv.reader(f)]
    keep = [r for r in rows if is_us_airport(r)]

    # A few IATA codes appear twice (e.g. renamed fields kept by OpenFlights);
    # keep the lowest airport_id per code, deterministically.
    keep.sort(key=lambda r: (r["iata"], int(r["airport_id"])))
    seen: set[str] = set()
    out = []
    for r in keep:
        if r["iata"] in seen:
            continue
        seen.add(r["iata"])
        tz = "" if r["tz"] == "\\N" else r["tz"]
        if not tz and r["iata"] in TZ_BACKFILL:
            tz = TZ_BACKFILL[r["iata"]]
            log.info("backfilled tz for %s: %s", r["iata"], tz)
        out.append(
            {
                "iata": r["iata"],
                "icao": r["icao"],
                "name": r["name"],
                "city": r["city"],
                "country": r["country"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "elevation_ft": r["altitude_ft"],
                "tz": tz,
            }
        )
    for extra in SUPPLEMENTS:
        if extra["iata"] not in seen:
            seen.add(extra["iata"])
            out.append(dict(extra))
        else:
            log.info("supplement %s now covered upstream", extra["iata"])
    out.sort(key=lambda r: r["iata"])
    return out


def generate_seed() -> None:
    """Entry point (also wrapped by orchestration): refresh dbt/seeds/airports.csv."""
    with tempfile.TemporaryDirectory(prefix="openflights_") as tmp:
        raw_path = Path(tmp) / "airports.dat"
        log.info("downloading %s", OPENFLIGHTS_URL)
        download_with_retries(OPENFLIGHTS_URL, raw_path)
        seed_rows = build_seed(raw_path)

    if len(seed_rows) < 500:
        raise SystemExit(f"only {len(seed_rows)} US airports parsed — source format likely changed")

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(seed_rows)
    log.info("wrote %d airports to %s", len(seed_rows), SEED_PATH)


def main() -> None:
    setup_logging()
    generate_seed()


if __name__ == "__main__":
    main()
