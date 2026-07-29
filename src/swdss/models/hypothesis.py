"""Hypothesis Testing: an experiment-management and evaluation system,
not a prediction page. A hypothesis pairs a baseline (dataset, variable)
against an experimental (dataset, variable) and lets the dashboard
automatically compare their measured, verified forecasting performance
(swdss.models.jobs.compute_variable_metrics) to report whether the
hypothesis is Supported, Not Supported, or Inconclusive — never "true."

All conclusions and confidence levels come from fixed, documented rules
over measured statistics. No LLM is used anywhere in this module.
"""

import sqlite3
import uuid
from contextlib import contextmanager

from swdss.models.jobs import compute_variable_metrics
from swdss.paths import DATA_DIR

DB_PATH = DATA_DIR / "predictions" / "predictions.db"

# Confidence bins, chosen so a 5-prediction hypothesis reads "Low" and a
# 250-prediction one reads "High" (the two example anchors in the spec).
CONFIDENCE_BINS = [
    (5, "Very Low"),
    (50, "Low"),
    (150, "Moderate"),
    (300, "High"),
]
CONFIDENCE_MAX_LABEL = "Very High"

# A change smaller than this (in either direction) isn't treated as
# meaningful — it's reported as Inconclusive rather than a false-precision
# verdict on noise.
MEANINGFUL_MAE_IMPROVEMENT_PCT = 5.0
MIN_PREDICTIONS_FOR_ANY_VERDICT = 5


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                number TEXT,
                title TEXT NOT NULL,
                description TEXT,
                scientific_motivation TEXT,
                physics_background TEXT,
                expected_improvement TEXT,
                baseline_dataset TEXT NOT NULL,
                baseline_variable TEXT NOT NULL,
                experimental_dataset TEXT NOT NULL,
                experimental_variable TEXT NOT NULL,
                notes TEXT,
                manual_conclusion TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            )
            """
        )


_init_db()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def create_hypothesis(
    title: str,
    description: str = "",
    scientific_motivation: str = "",
    physics_background: str = "",
    expected_improvement: str = "",
    baseline_dataset: str = "analytics",
    baseline_variable: str = "kp",
    experimental_dataset: str = "experimental",
    experimental_variable: str = "kp",
    notes: str = "",
    number: str = None,
) -> dict:
    import pandas as pd

    hypothesis_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, number, title, description, scientific_motivation,
                physics_background, expected_improvement, baseline_dataset,
                baseline_variable, experimental_dataset, experimental_variable,
                notes, manual_conclusion, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'active', ?)
            """,
            (
                hypothesis_id,
                number,
                title,
                description,
                scientific_motivation,
                physics_background,
                expected_improvement,
                baseline_dataset,
                baseline_variable,
                experimental_dataset,
                experimental_variable,
                notes,
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
    return get_hypothesis(hypothesis_id)


def get_hypothesis(hypothesis_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,)).fetchone()
        return _row_to_dict(row) if row else None


def list_hypotheses(status: str = None) -> list:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE status=? ORDER BY created_at ASC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM hypotheses ORDER BY created_at ASC").fetchall()
        return [_row_to_dict(r) for r in rows]


def update_hypothesis(hypothesis_id: str, **fields) -> dict:
    """Edits a hypothesis's structure (title/description/motivation/
    physics background/expected improvement/baseline & experimental
    architecture) — the mechanism behind "duplicate, then modify
    features/model/horizon/training data" from the spec: duplicate first
    (fresh notes/conclusion), then edit the copy's architecture pointers.
    """
    allowed = {
        "title",
        "description",
        "scientific_motivation",
        "physics_background",
        "expected_improvement",
        "baseline_dataset",
        "baseline_variable",
        "experimental_dataset",
        "experimental_variable",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_hypothesis(hypothesis_id)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _connect() as conn:
        conn.execute(
            f"UPDATE hypotheses SET {set_clause} WHERE hypothesis_id=?",
            (*updates.values(), hypothesis_id),
        )
    return get_hypothesis(hypothesis_id)


def update_notes(hypothesis_id: str, notes: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE hypotheses SET notes=? WHERE hypothesis_id=?", (notes, hypothesis_id))


def update_manual_conclusion(hypothesis_id: str, text: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE hypotheses SET manual_conclusion=? WHERE hypothesis_id=?", (text, hypothesis_id))


def archive_hypothesis(hypothesis_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE hypotheses SET status='archived' WHERE hypothesis_id=?", (hypothesis_id,))


def reactivate_hypothesis(hypothesis_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE hypotheses SET status='active' WHERE hypothesis_id=?", (hypothesis_id,))


def delete_hypothesis(hypothesis_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,))
        return cur.rowcount > 0


def duplicate_hypothesis(hypothesis_id: str, **overrides) -> dict:
    """Duplicates a hypothesis so a researcher can vary features, model,
    horizon, or training data and evaluate a new variant — the new
    hypothesis starts with its own fresh notes/conclusion, not copies of
    the original's collected results.
    """
    original = get_hypothesis(hypothesis_id)
    if original is None:
        raise ValueError(f"No hypothesis {hypothesis_id}")

    fields = {
        "title": original["title"] + " (copy)",
        "description": original["description"],
        "scientific_motivation": original["scientific_motivation"],
        "physics_background": original["physics_background"],
        "expected_improvement": original["expected_improvement"],
        "baseline_dataset": original["baseline_dataset"],
        "baseline_variable": original["baseline_variable"],
        "experimental_dataset": original["experimental_dataset"],
        "experimental_variable": original["experimental_variable"],
        "notes": "",
    }
    fields.update(overrides)
    return create_hypothesis(**fields)


def confidence_for_count(n: int) -> str:
    for threshold, label in CONFIDENCE_BINS:
        if n < threshold:
            return label
    return CONFIDENCE_MAX_LABEL


def evaluate_hypothesis(hypothesis: dict) -> dict:
    """Rule-based (no LLM) comparison of the hypothesis's baseline vs.
    experimental (dataset, variable) pair, using every completed &
    verified prediction collected so far. Returns metrics for both sides,
    a Supported/Not Supported/Inconclusive conclusion, a confidence
    level, and a plain-language summary — all derived from fixed rules
    over measured statistics, never a claim of "true."
    """
    baseline = compute_variable_metrics(hypothesis["baseline_dataset"], hypothesis["baseline_variable"])
    experimental = compute_variable_metrics(hypothesis["experimental_dataset"], hypothesis["experimental_variable"])

    n = min(baseline["count"], experimental["count"])
    confidence = confidence_for_count(n)

    if n < MIN_PREDICTIONS_FOR_ANY_VERDICT or baseline["mae"] is None or experimental["mae"] is None:
        return {
            "baseline": baseline,
            "experimental": experimental,
            "n": n,
            "conclusion": "Inconclusive",
            "confidence": "Very Low",
            "summary": "Additional verified predictions are required before reaching a conclusion.",
            "mae_improvement_pct": None,
            "r2_delta": None,
        }

    mae_improvement_pct = (baseline["mae"] - experimental["mae"]) / baseline["mae"] * 100
    r2_delta = (
        experimental["r2"] - baseline["r2"]
        if experimental["r2"] is not None and baseline["r2"] is not None
        else None
    )
    r2_text = ""
    if r2_delta is not None:
        r2_text = f" while improving R² by {r2_delta:.2f}" if r2_delta >= 0 else f" while reducing R² by {abs(r2_delta):.2f}"

    if mae_improvement_pct >= MEANINGFUL_MAE_IMPROVEMENT_PCT:
        conclusion = "Supported"
        summary = (
            f"The experimental architecture reduced MAE by {mae_improvement_pct:.1f}%{r2_text} "
            f"across {n} verified predictions."
        )
    elif mae_improvement_pct <= -MEANINGFUL_MAE_IMPROVEMENT_PCT:
        conclusion = "Not Supported"
        summary = (
            f"The experimental architecture increased MAE by {abs(mae_improvement_pct):.1f}%{r2_text} "
            f"across {n} verified predictions."
        )
    else:
        conclusion = "Inconclusive"
        summary = (
            f"No statistically meaningful improvement has been observed "
            f"({mae_improvement_pct:+.1f}% MAE change across {n} verified predictions)."
        )

    if n < 30:
        summary += " Additional verified predictions are recommended before treating this as a firm result."

    return {
        "baseline": baseline,
        "experimental": experimental,
        "n": n,
        "conclusion": conclusion,
        "confidence": confidence,
        "summary": summary,
        "mae_improvement_pct": mae_improvement_pct,
        "r2_delta": r2_delta,
    }


_DEFAULT_HYPOTHESES = [
    dict(
        number="001",
        title="Including Predicted AE improves Kp forecasting",
        description="Feeding the frozen AE model's predicted output forward as a Kp feature (the Physics Cascaded architecture) improves Kp forecast accuracy relative to the Independent architecture (observed history only).",
        scientific_motivation="AE reacts to solar wind driving within minutes, versus Kp's 3-hour cadence — physically an earlier signal in the Sun-Earth chain that could carry predictive information Kp's own observed history doesn't capture as quickly.",
        physics_background="Solar Wind -> IMF -> Magnetic Reconnection -> Auroral Electrojets (AE) -> Ring Current -> Kp/Dst. AE is a real-time proxy for auroral-zone magnetospheric activity driven directly by dayside/nightside reconnection.",
        expected_improvement="Lower MAE and higher R^2 for the Physics Cascaded Kp model vs. the Independent (production) Kp model.",
        baseline_dataset="analytics",
        baseline_variable="kp",
        experimental_dataset="experimental",
        experimental_variable="kp",
    ),
    dict(
        number="004",
        title="Physics Cascaded Models outperform Independent Models (Dst)",
        description="The Physics Cascaded architecture (Predicted AE feeding Dst) outperforms the Independent architecture (observed AE feeding Dst) for Dst forecasting.",
        scientific_motivation="If a fast, physics-derived AE signal genuinely carries information ahead of Dst's own slower ring-current buildup, cascading it forward should measurably help — the core question behind the AE V3 research stage.",
        physics_background="Same Sun-Earth causal chain as Hypothesis 001, applied to Dst's hour-by-hour ring-current response instead of Kp's 3-hour cadence.",
        expected_improvement="Lower MAE and higher R^2 for the Physics Cascaded Dst model vs. the Independent (production) Dst model.",
        baseline_dataset="analytics",
        baseline_variable="dst",
        experimental_dataset="experimental",
        experimental_variable="dst",
    ),
]


def seed_default_hypotheses() -> None:
    """Seeds the two example hypotheses that map onto architectures
    already built and running (001, 004) — done once, only if the table
    is empty, so it never clobbers a researcher's own edits/deletions.
    """
    if list_hypotheses():
        return
    for spec in _DEFAULT_HYPOTHESES:
        create_hypothesis(**spec)


seed_default_hypotheses()
