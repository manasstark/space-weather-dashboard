"""Forecast Packages — the synchronized operational forecast product.

A package bundles the 10 headline forecasts (Speed / Density /
Temperature / Bt / Bx / By / Bz / Dst / Kp / AE, each at its operational
horizon — 1h for everything except Kp's next official NOAA interval)
into ONE issued product with its own identity, lifecycle, and
verification. This is the object an operator should think of as "the
forecast" — not ten independent model outputs.

This module only re-packages what run_forecast_cycle/refresh_dashboard_
products already computed — it starts no job, calls no model, and
changes nothing about how any individual forecast is produced. See
orchestrator.refresh_dashboard_products for where it's assembled.

Two physical realities this design deliberately accounts for rather than
papering over:

- Kp cannot refresh on the same hourly clock as the other nine — NOAA
  publishes a new official 3-hour interval only a third as often. A
  package's Kp slot is marked "carried over" on cycles where no new
  interval has been issued since the previous package, and that alone
  never counts as PARTIALLY COMPLETE — only a genuine missing/errored
  member does.
- AE's ground truth arrives 10-20 days late via Kyoto WDC. The package's
  own VERIFIED status is gated on the other nine core members (which
  verify within the hour); AE's verification is tracked and reported as
  its own asynchronous line (evaluation_status), never holding the whole
  package hostage for weeks.
"""

import pandas as pd

from swdss.engine.confidence import CONFIDENCE_THRESHOLDS

PHYSICS_ENGINE_VERSION = "1.0"
PRODUCTION_VERSION = "1.0"

# (dataset, variable, horizon_label) for the 10 operational headline
# forecasts — same set used for the Overall Outlook classifier.
HEADLINE_KEYS = [
    ("solar_wind", "speed", "1h"),
    ("solar_wind", "density", "1h"),
    ("solar_wind", "temperature", "1h"),
    ("imf", "bt", "1h"),
    ("imf", "bx_gsm", "1h"),
    ("imf", "by_gsm", "1h"),
    ("imf", "bz_gsm", "1h"),
    ("analytics", "dst", "1h"),
    ("analytics", "kp", "interval"),
    ("ae", "ae", "1h"),
]

# AE is deliberately excluded — see module docstring.
CORE_VERIFICATION_VARIABLES = {"speed", "density", "temperature", "bt", "bx_gsm", "by_gsm", "bz_gsm", "dst", "kp"}


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _package_window(now: pd.Timestamp) -> tuple:
    generated_at = now.floor("h")
    valid_start = generated_at + pd.Timedelta(hours=1)
    valid_end = valid_start + pd.Timedelta(hours=1)
    return generated_at, valid_start, valid_end


def _package_id(generated_at: pd.Timestamp) -> str:
    return f"FC-{generated_at.strftime('%Y%m%d-%H%M')}"


def _confidence_category_from_score(score) -> str:
    if score is None:
        return "—"
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Low"


def _package_lifecycle_status(valid_start: pd.Timestamp, valid_end: pd.Timestamp, core_members: list, now: pd.Timestamp) -> str:
    if now < valid_start:
        return "LIVE"
    if now < valid_end:
        return "ACTIVE"
    core_done = bool(core_members) and all(m is not None and m.get("actual_value") is not None for m in core_members)
    return "VERIFIED" if core_done else "WAITING FOR VERIFICATION"


def _evaluation_status(core_members: list, ae_member: dict) -> str:
    core_verified = bool(core_members) and all(m is not None and m.get("actual_value") is not None for m in core_members)
    ae_verified = ae_member is not None and ae_member.get("actual_value") is not None
    if core_verified and ae_verified:
        return "Fully Verified"
    if core_verified:
        return "Verified (AE Pending Kyoto Data)"
    return "Pending"


def _build_package_summary(outlook: dict, explanations: dict) -> str:
    level = outlook.get("level", "Quiet")
    kp_sentence_source = (explanations or {}).get("kp", {})
    primary = kp_sentence_source.get("primary")
    if primary:
        return f"{level} conditions expected — driven primarily by {primary.lower()}."
    return f"{level} conditions expected."


def build_current_package(forecasts: dict, physics_summary: dict, outlook: dict, explanations: dict, recent_errors: list) -> dict:
    """Assembles the current Forecast Package from the already-computed
    per-variable forecast entries. Pure re-packaging — no new
    computation of any forecast value happens here.
    """
    now = _now()
    generated_at, valid_start, valid_end = _package_window(now)
    package_id = _package_id(generated_at)

    members = {}
    for dataset, variable, horizon_label in HEADLINE_KEYS:
        entry = (forecasts.get(dataset, {}).get(variable, {}) or {}).get(horizon_label)
        members[variable] = entry

    core_members = [members[v] for v in CORE_VERIFICATION_VARIABLES]
    ae_member = members.get("ae")

    # Kp carry-over: its forecast was generated before this package's own
    # generation hour, meaning NOAA hasn't published a new official
    # interval since the last package — normal 3h cadence, not a failure.
    kp_member = members.get("kp")
    kp_carried_over = False
    if kp_member is not None and kp_member.get("generated_at"):
        kp_generated = pd.Timestamp(kp_member["generated_at"])
        if kp_generated.tzinfo is None:
            kp_generated = kp_generated.tz_localize("UTC")
        kp_carried_over = kp_generated < generated_at

    missing_variables = [v for v, e in members.items() if e is None]
    completeness = "COMPLETE" if not missing_variables else "PARTIALLY COMPLETE"

    status = _package_lifecycle_status(valid_start, valid_end, core_members, now)

    confidences = [e["confidence"]["score"] for e in members.values() if e is not None and e.get("confidence")]
    overall_score = sum(confidences) / len(confidences) if confidences else None

    models_used = {v: (e.get("model_name") if e is not None else None) for v, e in members.items()}

    package = {
        "package_id": package_id,
        "forecast_cycle": None,  # assigned by storage.append_package_history_row
        "generated_at": generated_at.isoformat(),
        "valid_start": valid_start.isoformat(),
        "valid_end": valid_end.isoformat(),
        "status": status,
        "completeness": completeness,
        "missing_variables": missing_variables,
        "kp_carried_over": kp_carried_over,
        "models_used": models_used,
        "physics_engine_version": PHYSICS_ENGINE_VERSION,
        "production_version": PRODUCTION_VERSION,
        "overall_confidence": {"score": overall_score, "category": _confidence_category_from_score(overall_score)},
        "package_summary": _build_package_summary(outlook, explanations),
        "evaluation_status": _evaluation_status(core_members, ae_member),
        "package_archive_path": f"history/package_history.parquet#{package_id}",
        "recent_errors": recent_errors or [],
        "members": members,
        "outlook": outlook,
        "explanations": explanations,
    }
    return package


def build_verification_summary(package: dict) -> dict:
    """Returns the Package Verification Summary once every core member
    has a real observed value, else None (not ready yet). "Success" here
    reuses the SAME definition already used dataset-wide by
    jobs.get_prediction_statistics — abs_error <= 1.5x the model's own
    training MAE — rather than inventing a new accuracy metric.
    """
    core_members = {v: e for v, e in package["members"].items() if v in CORE_VERIFICATION_VARIABLES}
    verified = {v: e for v, e in core_members.items() if e is not None and e.get("actual_value") is not None}
    if len(verified) < len(core_members) or not verified:
        return None

    errors = {v: e["abs_error"] for v, e in verified.items() if e.get("abs_error") is not None}
    if not errors:
        return None

    successes = sum(
        1 for v, e in verified.items()
        if e.get("abs_error") is not None and e.get("metrics", {}).get("mae") and e["abs_error"] <= 1.5 * e["metrics"]["mae"]
    )
    worst_variable = max(errors, key=lambda v: errors[v])
    best_variable = min(errors, key=lambda v: errors[v])

    return {
        "package_id": package["package_id"],
        "verified_at": _now().isoformat(),
        "variables_verified": len(verified),
        "average_abs_error": sum(errors.values()) / len(errors),
        "worst_variable": worst_variable,
        "worst_variable_error": errors[worst_variable],
        "best_variable": best_variable,
        "best_variable_error": errors[best_variable],
        "overall_package_accuracy_pct": successes / len(verified) * 100,
    }
