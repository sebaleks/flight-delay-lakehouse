# Cloud Run image for the Streamlit dashboard (dashboard/README.md).
# Installs only the base deps + `dashboard` extra via uv; auth is ADC from the
# Cloud Run runtime service account (no key file, per CLAUDE.md §2).

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependency layer: cache-friendly, locked, no dev tooling, dashboard extra only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra dashboard

# App code. `ingestion/` is needed only for ingestion/config.py (env loading);
# the project itself is not installed, so make /app importable.
COPY ingestion/ ingestion/
COPY dashboard/ dashboard/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

# Cloud Run injects PORT (default 8080).
EXPOSE 8080
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]
