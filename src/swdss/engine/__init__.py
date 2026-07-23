"""Operational Forecast Engine — orchestration layer, not a new model.

Continuously ensures every production model in swdss.engine.matrix.
PRODUCTION_MATRIX has an active forecast job (via swdss.models.jobs),
derives confidence/physics/outlook context around the current job state,
and writes ready-to-render "forecast products" (swdss.engine.storage)
that dashboard/lib/command_centre.py reads — the dashboard itself never
calls a prediction model directly.

Entry points (swdss.engine.orchestrator): run_forecast_cycle(),
evaluate_due_forecasts(), refresh_dashboard_products() — each callable
independently; see live_update.py for how they're chained into the
existing continuous background process.
"""
