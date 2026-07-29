"""Automated retraining + promotion pipeline — the missing capability
flagged repeatedly in the Forecast Engine Review: "model refreshes are a
manual, ~6-month, human-run script." This replaces the MECHANISM (manual
script → callable, schedulable pipeline) while keeping every safety
property that manual process implicitly relied on, none of it dropped:

- Never ships a regression: a fresh candidate only replaces the
  currently-deployed model if its own walk-forward CV R² beats the
  deployed model's by MEANINGFUL_IMPROVEMENT — the same "prove it before
  you trust it" bar this project already applies elsewhere (F10.7's
  harmonic-vs-naive gate, the SHARP flare/CME TSS>0 gate, the ensemble
  blend's own 5% gate in train.py). A candidate that's merely different,
  not better, is discarded, not shipped.
- Always archives what it replaces (archive/{name}_{timestamp}.joblib),
  so a bad promotion is a file copy away from reversible — identical
  convention to promote_ae_to_production/promote_kp_to_production's
  existing manual archive/install pattern in the three Research Labs.
- Logs every decision, not just promotions — a slot that was retrained
  and NOT promoted is exactly as important a fact as one that was, for
  understanding whether this pipeline is doing anything at all.

Deliberately reuses train.py's train_slot/train_kp_interval_slot (built
alongside this module) rather than duplicating the walk-forward CV
selection logic — retraining and promotion are two separate concerns,
and this module owns only the second one.

This module does not schedule itself — call run_automated_retrain_cycle()
from a cron job, a manual script, or any other trigger. Retraining is
expensive (the same walk-forward CV over 3+ candidate algorithms every
production slot already pays at initial training time), so it is NOT
wired into the 60s live_update.py loop the way the Solar Forecast tab's
F10.7/CME computations are.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from swdss.engine.matrix import PRODUCTION_MATRIX
from swdss.models.registry import kp_interval_model_path, metrics_path, model_path
from swdss.models.train import prepare_dataset_frame, train_kp_interval_slot, train_slot
from swdss.paths import FORECASTS_DIR

RETRAIN_LOG_PATH = FORECASTS_DIR / "history" / "retrain_log.jsonl"

# Lower than the ensemble blend's 5% (train.py's ENSEMBLE_MEANINGFUL_IMPROVEMENT)
# since this compares against genuinely new data (a retrain months later),
# where a real improvement is more plausible than when blending the same
# snapshot's own candidates — but still a real bar, not "any positive delta."
MEANINGFUL_IMPROVEMENT = 0.02


def _log_decision(record: dict) -> None:
    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_deployed_metrics(dataset: str, metrics_key: str) -> dict | None:
    path = metrics_path(dataset)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get(metrics_key)


def _archive_current_model(current_path: Path, archive_name: str) -> str | None:
    if not current_path.exists():
        return None
    archive_dir = current_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{archive_name}_{ts}.joblib"
    shutil.copy2(current_path, archive_path)
    return str(archive_path)


def _decide(new_cv_r2: float, deployed: dict | None) -> tuple[bool, str, float | None]:
    """Returns (should_promote, reason, relative_improvement)."""
    if deployed is None or deployed.get("cv_r2_mean") is None:
        return True, "no_prior_baseline", None

    old_cv_r2 = deployed["cv_r2_mean"]
    if old_cv_r2 > 0:
        relative_improvement = (new_cv_r2 - old_cv_r2) / old_cv_r2
    else:
        relative_improvement = 1.0 if new_cv_r2 > 0 else 0.0

    if new_cv_r2 > old_cv_r2 and relative_improvement >= MEANINGFUL_IMPROVEMENT:
        return True, "meaningful_improvement", relative_improvement
    return False, "no_meaningful_improvement", relative_improvement


def retrain_and_promote_slot(dataset: str, variable: str, horizon: int, prepared: tuple | None = None) -> dict:
    """Retrains ONE (dataset, variable, horizon) production slot and
    promotes it only if it clears MEANINGFUL_IMPROVEMENT over the
    currently-deployed model. Always returns (and logs) a decision
    record regardless of outcome.
    """
    metrics_key = f"{variable}_{horizon}h"
    deployed = _load_deployed_metrics(dataset, metrics_key)

    final_model, record = train_slot(dataset, variable, horizon, prepared=prepared)
    should_promote, reason, relative_improvement = _decide(record["cv_r2_mean"], deployed)

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "variable": variable,
        "horizon": horizon,
        "candidate_algorithm": record["algorithm"],
        "candidate_cv_r2_mean": record["cv_r2_mean"],
        "deployed_algorithm": (deployed or {}).get("algorithm"),
        "deployed_cv_r2_mean": (deployed or {}).get("cv_r2_mean"),
        "relative_improvement": relative_improvement,
        "decision": "promoted" if should_promote else "kept_existing",
        "reason": reason,
    }

    if should_promote:
        path = model_path(dataset, variable, horizon)
        decision["archive_path"] = _archive_current_model(path, f"{variable}_{horizon}h")

        import joblib
        joblib.dump(final_model, path)
        record["model_path"] = str(path)

        metrics_doc_path = metrics_path(dataset)
        metrics_doc = {}
        if metrics_doc_path.exists():
            with open(metrics_doc_path, encoding="utf-8") as f:
                metrics_doc = json.load(f)
        metrics_doc[metrics_key] = record
        with open(metrics_doc_path, "w", encoding="utf-8") as f:
            json.dump(metrics_doc, f, indent=2)

    _log_decision(decision)
    return decision


def retrain_and_promote_kp_interval(dataset: str = "analytics") -> dict:
    """Same decision/archive/log contract as retrain_and_promote_slot,
    for Kp's "next official NOAA interval" slot, which is trained and
    stored separately from the fixed-horizon slots (see
    kp_interval_model_path).
    """
    metrics_key = "kp_interval"
    deployed = _load_deployed_metrics(dataset, metrics_key)

    final_model, record = train_kp_interval_slot(dataset)
    should_promote, reason, relative_improvement = _decide(record["cv_r2_mean"], deployed)

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "variable": "kp",
        "horizon": "interval",
        "candidate_algorithm": record["algorithm"],
        "candidate_cv_r2_mean": record["cv_r2_mean"],
        "deployed_algorithm": (deployed or {}).get("algorithm"),
        "deployed_cv_r2_mean": (deployed or {}).get("cv_r2_mean"),
        "relative_improvement": relative_improvement,
        "decision": "promoted" if should_promote else "kept_existing",
        "reason": reason,
    }

    if should_promote:
        path = kp_interval_model_path(dataset)
        decision["archive_path"] = _archive_current_model(path, "kp_interval")

        import joblib
        joblib.dump(final_model, path)
        record["model_path"] = str(path)

        metrics_doc_path = metrics_path(dataset)
        metrics_doc = {}
        if metrics_doc_path.exists():
            with open(metrics_doc_path, encoding="utf-8") as f:
                metrics_doc = json.load(f)
        metrics_doc[metrics_key] = record
        with open(metrics_doc_path, "w", encoding="utf-8") as f:
            json.dump(metrics_doc, f, indent=2)

    _log_decision(decision)
    return decision


def run_automated_retrain_cycle() -> list[dict]:
    """Walks PRODUCTION_MATRIX, retraining and conditionally promoting
    every slot. Groups by dataset so prepare_dataset_frame's (training
    CSV load + feature-frame build) cost is paid once per dataset, not
    once per slot. Meant to be run on a schedule (monthly, say) — not
    from live_update.py's 60s loop.
    """
    decisions = []
    prepared_by_dataset: dict = {}

    for dataset, variable, horizons in PRODUCTION_MATRIX:
        if horizons == "interval":
            decisions.append(retrain_and_promote_kp_interval(dataset))
            continue

        if dataset not in prepared_by_dataset:
            prepared_by_dataset[dataset] = prepare_dataset_frame(dataset)
        prepared = prepared_by_dataset[dataset]

        horizon_list = horizons if isinstance(horizons, list) else [horizons]
        for horizon in horizon_list:
            decisions.append(retrain_and_promote_slot(dataset, variable, horizon, prepared=prepared))

    return decisions


def load_retrain_log(limit: int = 200) -> list[dict]:
    """Most recent `limit` decisions, newest last (matches
    append-only-log convention already used elsewhere, e.g.
    storage.load_recent_logs) — for a future dashboard tab to display
    the retrain pipeline's own history without needing a new storage
    format.
    """
    if not RETRAIN_LOG_PATH.exists():
        return []
    with open(RETRAIN_LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-limit:]]
