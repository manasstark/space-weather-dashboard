"""Burton et al. (1975) ring-current injection/decay model for Dst.

Why this exists: production Dst forecasting (see swdss.models.train) is a
straight-line (LinearRegression) fit against instantaneous solar wind/IMF
features. This project's own Storm Backtest results show that model's
error jumping 3-9x specifically during storm-regime hours (e.g. 1.4 nT
quiet vs 12.7 nT storm MAE for the May 2024 "Gannon" G5 storm) — a
straight line structurally cannot represent the ring current's nonlinear
saturation response to extreme driving, no matter how it's trained. This
module is the physics-first alternative: rather than fitting Dst directly
against instantaneous inputs, it models the ring current as a single
reservoir that fills from solar wind driving and empties on its own decay
time — a mechanism that doesn't need to have been trained on a G5 storm to
behave sensibly during one, because it's an integration of a physical law,
not a fit to a training distribution.

    dDst*/dt = a * VBz(t) - Dst*/tau

    Dst*   = pressure-corrected Dst (see below) — the quantity actually
             driven by ring current energy content
    VBz    = Solar Wind Speed x min(Bz, 0) (swdss.physics.core.vbz_series)
             — southward-only geoeffective coupling; northward IMF injects
             nothing
    a      = injection coefficient (fit per-project, see calibrate_burton_
             params — NOT assumed from the literature, since published
             values vary by an order of magnitude across studies and this
             project has its own multi-year OMNI2-derived training corpus
             to fit against directly)
    tau    = ring current decay time constant, hours (also fit, not
             assumed — see calibrate_burton_params)

Pressure correction: real Dst includes a contribution from magnetopause
currents that has nothing to do with the ring current, so Burton's own
formulation predicts a corrected Dst* rather than raw Dst:

    Dst* = Dst - PRESSURE_B * sqrt(Pdyn) + PRESSURE_C

PRESSURE_B/PRESSURE_C are kept at published literature values (O'Brien &
McPherron, 2000, "An empirical phase space analysis of ring current
dynamics: Solar wind control of injection and decay", J. Geophys. Res.)
rather than re-fit here — they correct a small, well-established magneto-
pause-current effect, not the actual physics under test (the injection/
decay dynamics), so re-deriving them from scratch would add calibration
noise without testing anything new.

Reference: Burton, R. K., McPherron, R. L., & Russell, C. T. (1975), "An
empirical relationship between interplanetary conditions and Dst", J.
Geophys. Res., 80(31), 4204-4214.

Variable-tau extension (2026-07 addition, after the constant-tau version's
first backtest): a constant tau tied production on a genuine blind-test
storm (September 2017) but lost clearly during the most extreme case in
this project's set (May 2024 "Gannon", G5) specifically in the storm
regime — real ring current decay is faster during a storm's driven main
phase and slower during recovery, which a single fixed tau structurally
cannot represent. O'Brien & McPherron (2000) found decay time is
controlled by driving strength (VBs, the southward interplanetary
electric field), not by Dst* itself — the widely-cited dependence on Dst*
being, per their analysis, an alias of intense Dst* and intense VBs
usually co-occurring:

    tau(t) = 2.4 * exp(9.74 / (4.69 + VBs(t)))   hours, VBs in mV/m

VBs = Speed x max(-Bz, 0) x 1e-3 — the southward-only interplanetary
electric field (swdss.physics.core.ey_series clipped to non-negative,
same underlying quantity, zero when IMF is northward). This functional
form's constants (2.4, 9.74, 4.69) are kept at O'Brien & McPherron's own
published values — like PRESSURE_B/PRESSURE_C, this is the specific,
well-established shape of driving-dependent decay their analysis derived,
not the one free parameter this project's own corpus should re-fit. The
injection coefficient `a` remains calibrated on this project's own
training corpus exactly as in the constant-tau version (see
calibrate_burton_variable_tau_a) — VBz's injection formulation is
unchanged; only tau's dependence on driving strength is added, isolating
that one change so any difference in backtest results is attributable to
it specifically, not conflated with also changing how injection itself
works.

Reference: O'Brien, T. P., & McPherron, R. L. (2000), "An empirical phase
space analysis of ring current dynamics: Solar wind control of injection
and decay", J. Geophys. Res., 105(A4), 7707-7719.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from swdss.physics.core import dynamic_pressure_series, ey_series, vbz_series

# O'Brien & McPherron (2000) pressure-correction constants (nT, nT/sqrt(nPa)).
PRESSURE_B = 7.26
PRESSURE_C = 11.0

# O'Brien & McPherron (2000) variable decay-time constants: tau(VBs) =
# TAU_COEF * exp(TAU_EXP_NUMERATOR / (TAU_EXP_OFFSET + VBs)), VBs in mV/m.
TAU_COEF_HOURS = 2.4
TAU_EXP_NUMERATOR = 9.74
TAU_EXP_OFFSET = 4.69

MIN_CALIBRATION_SAMPLES = 500


def southward_efield_series(speed: pd.Series, bz: pd.Series) -> pd.Series:
    """VBs — the southward-only interplanetary electric field (mV/m) that
    O'Brien & McPherron (2000) found controls ring current decay time.
    Identical quantity to swdss.physics.core.ey_series, clipped to
    non-negative: Ey is already -Speed*Bz*1e-3, which is positive exactly
    when Bz is southward (negative) and would go negative for northward
    Bz, where there is no southward driving to speak of.
    """
    return ey_series(speed, bz).clip(lower=0)


def variable_tau_hours(vbs: pd.Series) -> pd.Series:
    return TAU_COEF_HOURS * np.exp(TAU_EXP_NUMERATOR / (TAU_EXP_OFFSET + vbs))


def pressure_corrected_dst(dst: pd.Series, pdyn: pd.Series) -> pd.Series:
    return dst - PRESSURE_B * np.sqrt(pdyn.clip(lower=0)) + PRESSURE_C


def calibrate_burton_params(df: pd.DataFrame, dt_hours: float = 1.0) -> dict:
    """Fits the injection coefficient `a` and decay time `tau_hours` via
    ordinary least squares: dDst*/dt is linear in VBz(t) and Dst*(t), so
    both free parameters of the ODE fall out of a single two-feature
    regression against the finite-difference derivative — no iterative
    solver needed. `df` must carry raw (unscaled) speed/density/bz_gsm/dst
    columns with no NaNs; callers are responsible for excluding whatever
    storm window they intend to later evaluate against, so this never
    calibrates on the data it will be tested on.

    Returns a=injection coefficient, tau_hours=decay time constant,
    decay_coef=the raw (negative) regression coefficient tau is derived
    from, r2=fit quality on the training corpus, n_samples. tau_hours is
    None if the fit produced a non-physical (non-negative) decay
    coefficient — callers must check for this before integrating, since a
    non-negative decay coefficient means the corpus couldn't constrain a
    stable (mean-reverting) ring current decay at all.
    """
    required = {"speed", "density", "bz_gsm", "dst"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"calibrate_burton_params requires columns {sorted(required)}, missing {sorted(missing)}.")

    pdyn = dynamic_pressure_series(df["density"], df["speed"])
    dst_star = pressure_corrected_dst(df["dst"], pdyn)
    vbz = vbz_series(df["speed"], df["bz_gsm"])

    frame = pd.DataFrame({"vbz": vbz, "dst_star": dst_star}).dropna()
    if len(frame) < MIN_CALIBRATION_SAMPLES + 1:
        raise ValueError(
            f"Only {len(frame)} usable rows available to calibrate Burton parameters "
            f"(need at least {MIN_CALIBRATION_SAMPLES + 1}) — corpus too small or too many NaNs."
        )

    ddst_star_dt = (frame["dst_star"].shift(-1) - frame["dst_star"]) / dt_hours
    calib = frame.iloc[:-1].copy()
    calib["ddst_star_dt"] = ddst_star_dt.iloc[:-1].to_numpy()
    calib = calib.dropna()

    X = calib[["vbz", "dst_star"]].to_numpy()
    y = calib["ddst_star_dt"].to_numpy()
    model = LinearRegression().fit(X, y)

    a = float(model.coef_[0])
    decay_coef = float(model.coef_[1])
    tau_hours = -1.0 / decay_coef if decay_coef < 0 else None

    return {
        "a": a,
        "tau_hours": tau_hours,
        "decay_coef": decay_coef,
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "n_samples": int(len(calib)),
    }


def burton_one_step_forecast_dst(df: pd.DataFrame, a: float, tau_hours: float, dt_hours: float = 1.0) -> pd.Series:
    """Burton's dt_hours-ahead Dst prediction, issued fresh at every row
    from that row's own ACTUAL observed state — not a multi-hour free-
    running integration compounding the model's own prior predictions.

    This is the fair counterpart to compare against production's own Dst
    model: production's lag/rolling features are built from real observed
    history up to the issuance hour (never its own past predictions), so
    it is itself re-anchored to ground truth at every single issuance, not
    a free-running trajectory either. Matching that exactly here — rather
    than integrating Burton continuously through a multi-day storm window
    from one starting anchor — is what makes the two models' MAE numbers
    comparable at all; a free-running physics integration would accumulate
    error over a multi-day window in a way a model re-anchored every hour
    never does, and the resulting comparison would penalize Burton for a
    harder task production was never asked to do either.

    Returns a series indexed the SAME as df (issuance time t), whose value
    at t is Burton's prediction for t + dt_hours — the same convention
    swdss.models.storm_data.build_target_series uses (target = value.shift
    (-horizon)), so callers compare it directly against that shifted
    actual series, not against df["dst"] itself.

    Uses the exact solution for a constant drive over one step,
    dst_star(t+dt) = dst_star(t)*exp(-dt/tau) + a*VBz(t)*tau*(1-exp(-dt/
    tau)), evaluated from each row's own real dst_star(t) — vectorized,
    no loop needed since every row is an independent one-step projection.
    The next hour's dynamic pressure isn't known at issuance time, so the
    pressure correction is added back using the CURRENT hour's Pdyn — the
    best available estimate at the moment the forecast is actually issued.
    """
    required = {"speed", "density", "bz_gsm", "dst"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"burton_one_step_forecast_dst requires columns {sorted(required)}, missing {sorted(missing)}.")
    if df[list(required)].isna().any().any():
        raise ValueError("burton_one_step_forecast_dst requires NaN-free input — callers must dropna() the window first.")

    pdyn = dynamic_pressure_series(df["density"], df["speed"])
    dst_star_actual = pressure_corrected_dst(df["dst"], pdyn)
    vbz = vbz_series(df["speed"], df["bz_gsm"])

    decay = np.exp(-dt_hours / tau_hours)
    dst_star_next = dst_star_actual * decay + a * vbz * tau_hours * (1 - decay)
    return dst_star_next + PRESSURE_B * np.sqrt(pdyn.clip(lower=0)) - PRESSURE_C


def calibrate_burton_variable_tau_a(df: pd.DataFrame, dt_hours: float = 1.0) -> dict:
    """Fits ONLY the injection coefficient `a` — tau(t) is no longer a
    free parameter here, it's O'Brien & McPherron's published function of
    VBs(t) (see variable_tau_hours), so the ODE reduces back to a single
    unknown: dDst*/dt + Dst*(t)/tau(VBs(t)) = a*VBz(t). Moving the known
    decay term to the left leaves a plain one-feature linear regression
    for `a` against VBz(t), the same OLS approach calibrate_burton_params
    uses for the constant-tau version, just with one fewer free parameter
    since tau's shape is no longer being fit.
    """
    required = {"speed", "density", "bz_gsm", "dst"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"calibrate_burton_variable_tau_a requires columns {sorted(required)}, missing {sorted(missing)}.")

    pdyn = dynamic_pressure_series(df["density"], df["speed"])
    dst_star = pressure_corrected_dst(df["dst"], pdyn)
    vbz = vbz_series(df["speed"], df["bz_gsm"])
    vbs = southward_efield_series(df["speed"], df["bz_gsm"])
    tau = variable_tau_hours(vbs)

    frame = pd.DataFrame({"vbz": vbz, "dst_star": dst_star, "tau": tau}).dropna()
    if len(frame) < MIN_CALIBRATION_SAMPLES + 1:
        raise ValueError(
            f"Only {len(frame)} usable rows available to calibrate Burton parameters "
            f"(need at least {MIN_CALIBRATION_SAMPLES + 1}) — corpus too small or too many NaNs."
        )

    ddst_star_dt = (frame["dst_star"].shift(-1) - frame["dst_star"]) / dt_hours
    calib = frame.iloc[:-1].copy()
    calib["ddst_star_dt"] = ddst_star_dt.iloc[:-1].to_numpy()
    calib = calib.dropna()

    adjusted_target = calib["ddst_star_dt"] + calib["dst_star"] / calib["tau"]
    X = calib[["vbz"]].to_numpy()
    y = adjusted_target.to_numpy()
    model = LinearRegression().fit(X, y)

    return {
        "a": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "n_samples": int(len(calib)),
    }


def burton_variable_tau_one_step_forecast_dst(df: pd.DataFrame, a: float, dt_hours: float = 1.0) -> pd.Series:
    """Variable-tau counterpart to burton_one_step_forecast_dst: identical
    one-step, re-anchored-to-real-state design (see that function's
    docstring for why this is the fair comparison against production, not
    a free-running multi-hour integration), except tau is recomputed at
    every row from that row's own driving strength (variable_tau_hours)
    instead of held constant.
    """
    required = {"speed", "density", "bz_gsm", "dst"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"burton_variable_tau_one_step_forecast_dst requires columns {sorted(required)}, missing {sorted(missing)}.")
    if df[list(required)].isna().any().any():
        raise ValueError("burton_variable_tau_one_step_forecast_dst requires NaN-free input — callers must dropna() the window first.")

    pdyn = dynamic_pressure_series(df["density"], df["speed"])
    dst_star_actual = pressure_corrected_dst(df["dst"], pdyn)
    vbz = vbz_series(df["speed"], df["bz_gsm"])
    vbs = southward_efield_series(df["speed"], df["bz_gsm"])
    tau = variable_tau_hours(vbs)

    decay = np.exp(-dt_hours / tau)
    dst_star_next = dst_star_actual * decay + a * vbz * tau * (1 - decay)
    return dst_star_next + PRESSURE_B * np.sqrt(pdyn.clip(lower=0)) - PRESSURE_C
