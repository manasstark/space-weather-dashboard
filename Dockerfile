# Basic reproducible packaging for the app + its Python environment.
# Not full orchestration (no compose/k8s here) — just "clone, build, run"
# parity with the documented local setup in README.md.
FROM python:3.11-slim

# libgomp1: OpenMP runtime needed by xgboost/lightgbm/catboost at import
# time, not just build time — missing on the slim base image otherwise.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# data/, models/, logs/ are generated at runtime (see .gitignore) — mount
# them as volumes to persist forecasts/trained models across container
# restarts, e.g.:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ...
ENV PYTHONPATH=/app/src:/app
EXPOSE 8501

# Default command runs the dashboard. The other long-running process,
# the forecast engine's live_update loop, is a separate container command
# (not started here — see README "Running the Project"):
#   docker run <image> python -m swdss.features.live_update
CMD ["streamlit", "run", "dashboard/home.py", "--server.address=0.0.0.0", "--server.port=8501"]
