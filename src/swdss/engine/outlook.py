"""Rule-based, deterministic "Overall Space Weather Outlook" — combines
the engine's own Kp/Dst/AE forecasts into one 5-level operational
assessment. Every band below is numerically identical to the thresholds
already established in swdss.models.physics_interpretation
(_geomagnetic_activity, _ring_current_response, _auroral_activity) and
swdss.models.jobs._STORM_CHECKS — restated here (not imported, since
those are private per-index narrative helpers returning text, not a
level this classifier can combine) so "Moderate Storm" here means the
same Kp/Dst/AE thresholds the rest of the dashboard already uses.
"""

OUTLOOK_LEVELS = ["Quiet", "Unsettled", "Minor Storm", "Moderate Storm", "Strong Storm"]


def _kp_level(kp):
    if kp is None:
        return 0, ""
    if kp >= 7:
        return 4, f"Predicted Kp {kp:.1f} (G3+ storm level)"
    if kp >= 6:
        return 3, f"Predicted Kp {kp:.1f} (G2 storm level)"
    if kp >= 5:
        return 2, f"Predicted Kp {kp:.1f} (G1 storm level)"
    if kp >= 4:
        return 1, f"Predicted Kp {kp:.1f} (Active)"
    return 0, f"Predicted Kp {kp:.1f} (Quiet)"


def _dst_level(dst):
    if dst is None:
        return 0, ""
    if dst <= -200:
        return 4, f"Predicted Dst {dst:.0f} nT (severe storm)"
    if dst <= -100:
        return 3, f"Predicted Dst {dst:.0f} nT (intense storm)"
    if dst <= -50:
        return 2, f"Predicted Dst {dst:.0f} nT (moderate storm)"
    if dst <= -30:
        return 1, f"Predicted Dst {dst:.0f} nT (weak disturbance)"
    return 0, f"Predicted Dst {dst:.0f} nT (quiet)"


def _ae_level(ae):
    if ae is None:
        return 0, ""
    if ae >= 500:
        return 3, f"Predicted AE {ae:.0f} nT (storm-level auroral activity)"
    if ae >= 300:
        return 2, f"Predicted AE {ae:.0f} nT (active auroral activity)"
    if ae >= 100:
        return 1, f"Predicted AE {ae:.0f} nT (unsettled auroral activity)"
    return 0, f"Predicted AE {ae:.0f} nT (quiet)"


def classify_overall_outlook(*, predicted_kp, predicted_dst, predicted_ae) -> tuple:
    """Takes the WORST (max-severity) level across Kp/Dst/AE. Returns
    (level_name, reasoning) — reasoning lists every contributing index's
    plain-English rationale, worst first.
    """
    results = [_kp_level(predicted_kp), _dst_level(predicted_dst), _ae_level(predicted_ae)]
    results = [(lvl, text) for lvl, text in results if text]
    if not results:
        return "Quiet", ["Insufficient forecast data to assess current outlook."]

    max_level = max(lvl for lvl, _ in results)
    reasoning = [text for _, text in sorted(results, key=lambda r: -r[0])]
    return OUTLOOK_LEVELS[max_level], reasoning


# Activity-regime bucketing — reuses the SAME per-index level functions
# above, collapsed from 5 outlook levels to 3 coarse regimes (Quiet /
# Active / Storm). This exists to TAG evaluated forecasts with the
# activity level they were made under, so a future version can compute
# quiet-time vs. active-time vs. storm-time error bands from real
# labeled history — see swdss.engine.confidence's RANGE_VARIABLES
# handling, which today uses one global MAE for every condition.
# Deliberately NOT used to segment anything yet (see engine's item 8
# scope: collect the label now, calibrate against it later).
ACTIVITY_REGIMES = ["Quiet", "Active", "Storm"]


def classify_activity_regime(*, predicted_kp, predicted_dst, predicted_ae) -> str:
    """Coarse 3-bucket regime from the same worst-of-three-indices logic
    as classify_overall_outlook: level 0 -> Quiet, 1-2 -> Active, 3-4 ->
    Storm.
    """
    results = [_kp_level(predicted_kp)[0], _dst_level(predicted_dst)[0], _ae_level(predicted_ae)[0]]
    max_level = max(results)
    if max_level == 0:
        return "Quiet"
    if max_level <= 2:
        return "Active"
    return "Storm"
