# Codebase Map

A map of where things live and why, so you can navigate confidently and explain design choices out loud without needing the code reorganized first. Every file/line reference below was checked against the actual repo, not guessed.

---

## 1. Repo shape, one line per top-level piece

```
src/swdss/models/    → production model training + inference (24 files)
src/swdss/engine/    → the Operational Forecast Engine (15 files)
src/swdss/physics/   → physics formulas (DBM, geometry, persistence — pure math, no I/O)
src/swdss/features/  → data ingestion pipeline (fetch, clean, build master dataset)
dashboard/           → the Streamlit app
dashboard/lib/       → one file per dashboard topic/page-section (14 files)
data/                → raw → processed → features → forecasts (the pipeline's own data lake)
```

Why so many files instead of a few big ones: each file has one job and its name says what that job is. `ls src/swdss/models/` alone tells a reader the shape of the ML system before opening a single file. This is the opposite direction from "fewer, bigger files" — and it's why the code has held up to a lot of debugging without one change rippling unpredictably into another file.

---

## 2. Production Models — `src/swdss/models/`

**The key idea to lead with:** there is no per-variable code. One generic pipeline, driven by config, handles every variable. This is the single most interview-relevant fact about this part of the codebase.

**`registry.py`** — the config. Single source of truth both training and inference import from, so they can never drift apart (that's stated directly in its own docstring).
```python
SOLAR_WIND_VARIABLES = ["speed", "density", "temperature"]
IMF_VARIABLES        = ["bt", "bx_gsm", "by_gsm", "bz_gsm"]
KP_VARIABLES          = ["kp"]
DST_VARIABLES         = ["dst"]
AE_VARIABLES          = ["ae"]
HORIZONS              = [1, 3, 6, 12, 24]   # hours
```
`DATASETS` (line 142) ties each dataset name to its variables, its raw source paths, and its feature-engineering function.

**`train.py`** — generic training, not per-variable:
- `train_slot(dataset_key, variable, horizon, ...)` — trains ONE model for one (dataset, variable, horizon). Candidate models are compared (`_fit_best`), evaluated with walk-forward CV, and can be blended into a `WeightedEnsembleRegressor` when 2+ candidates show positive out-of-sample weight.
- `train_dataset(dataset_key)` — loops that same `train_slot` over every variable × horizon for one dataset. This is the function that "trains all the Solar Wind models" — there's no separate speed/density/temperature training code, just this loop calling the same function three times with different arguments.
- `train_kp_interval_model()` — Kp is the one exception: it forecasts a fixed 3-hour Kp *interval* rather than an N-hour-ahead point value (matches how NOAA actually issues Kp), so it gets its own function rather than fitting the same `train_slot` shape.

**`predict.py`** — same pattern for live inference:
- `predict(dataset, variable, horizon)` (line 370) — the one function every production forecast in the app ultimately calls.
- `predict_kp_interval()` / `predict_kp_rolling()` — Kp's own inference path, mirroring its own training path.
- `latest_minute_observation()` (line 151) — pulls the most recent raw reading for "current conditions" displays, independent of any model.

**Talking point:** if asked "walk me through how the Speed model works," the honest answer is "there's no Speed-specific code — `registry.py` lists it as a Solar Wind variable, `train_dataset('solar_wind')` calls `train_slot` with `variable='speed'`, and the same function produces the Bt, Bz, Dst, and Kp models too, just with different config." That's a stronger answer than pointing at a dedicated block of code — it shows the pipeline is genuinely reusable, not copy-pasted.

---

## 3. Forecast Engine — `src/swdss/engine/orchestrator.py`

This is the thing that turns individual `predict()` calls into one coherent operational product every cycle. `refresh_dashboard_products()` runs in 8 named stages (real section-banner comments already in the file):

| Stage | Line | What it does |
|---|---|---|
| 1 | 709 | Walks every (dataset, variable, horizon), builds the live forecast tree + the permanent history log + evaluation rows |
| 2 | 907 | Physics summary + Overall Outlook classification |
| 3 | 929 | Freshness checks, alert generation, drift detection |
| 4 | 955 | Explanation engine — rule-based driver ranking ("why this Dst forecast") |
| 5 | 986 | Assembles the final snapshot dict |
| 6 | 1001 | Builds the Forecast Package (the 10-headline-variable bundle with a lifecycle: LIVE → ACTIVE → VERIFIED) |
| 7 | 1021 | Persists history + verification science to disk |
| 8 | 1045 | Solar Forecast (F10.7 harmonic regression + CME Drag-Based-Model arrival) |

Each stage is wrapped in its own `try/except` so one stage failing doesn't take the rest down — a real, deliberate fault-isolation pattern.

**`storage.py`** — the engine's only interface to disk. Every write is atomic (temp file + `os.replace`, never a direct overwrite) and `forecast_snapshot_history.parquet` is a rolling 72-hour window, not a permanent archive.

**`packages.py`** — defines `HEADLINE_KEYS`, the 10 (dataset, variable, horizon) combinations that count as "operational": 1h for everything, Kp's 3h interval. This is the canonical definition reused everywhere something needs to know "is this one of the real headline forecasts or an extended-horizon one" — the Verification tab's Operational/Extended split, the Forecast Package's membership, and the history-logging filter all read from this one list.

**`calibration.py` / `skill.py` / `drift.py`** — verification science: confidence calibration bands by activity regime, persistence-based skill scoring, and drift detection (is a model's live error creeping above its training-time MAE).

---

## 4. Dashboard — `dashboard/`

**`home.py` (2,259 lines)** — masthead, top nav, and one function per page:

| Page | Function | Line |
|---|---|---|
| Home | `home_page()` | 1441 |
| Heliosphere | `heliosphere_page()` | 1471 |
| Geospace | `geospace_page()` | 1512 |
| Photosphere | `photosphere_page()` | 1525 |
| Analytics | `analytics_page()` | 1770 |
| Research Lab | `research_lab_page()` | 2053 |

The router at the bottom is literally `if page == "Photosphere": photosphere_page(df)` — if an interviewer wants the "page → what's on it" view, this is the exact section to show them.

Each page function is short and mostly delegates. E.g. `photosphere_page()` is ~18 lines: it just opens tabs ("Solar Events", "CME", "F10.7") and calls `solar_events_analysis()`, `cme_analysis()`, `f107_analysis()` — all of which live in `solar_activity.py`, not in `home.py`. That's where heliomaps live too (that file's own docstring: "news feed, CME/solar analysis, heliomaps, F10.7").

**`dashboard/lib/command_centre.py` (1,497 lines)** — the Home page's "SW Operational Command Centre," a dense terminal-style instrument, not a SaaS dashboard (a deliberate design choice, stated in its own module docstring). 10 tabs, each with a real section banner:

| Tab | Line |
|---|---|
| Forecast | 556 |
| Current | 724 |
| Physics | 772 |
| Timeline | 820 |
| Verification | 872 |
| Logs | 1160 |
| Downloads | 1193 |
| Alerts | 1268 |
| System | 1292 |
| Solar Forecast | 1332 |

This file reads *only* from `swdss.engine.storage` — never calls a model directly, never touches the jobs database — matching the engine's own "compute once, dashboard only reads" architecture.

**Other `dashboard/lib/` files, one line each:**
- `data_helpers.py` — shared low-level loaders (`load_master_data`, `load_processed_data`, formatting helpers)
- `event_explorer.py` — the "what caused this" reverse lookup: CME → source flare/burst → Earth arrival chain
- `forecast_dialogs.py` — Kp/Dst/AE/Experimental detail dialogs
- `library.py` — Saved Events CRUD
- `shared_ui.py` — retro terminal styling, auto-refresh timer, dialog open/close plumbing
- `project_status.py` — internal progress notes page

---

## 5. Research Labs — `dashboard/lib/*_research_lab.py` + `storm_lab.py`

Every lab shares one non-negotiable design rule, stated directly in each lab's own on-screen caption: **fully isolated from production.** Nothing trained here ever overwrites the production model; "Promote" only labels a run for your own tracking — wiring a model into production is always a separate, manual, deliberate step. This is enforced by construction (research code lives in different files, writes to different paths) rather than relying on convention.

**Entry point:** `research_lab_page()` in `home.py` (line 2053) — 6 top-level tabs:

| Tab | What it is |
|---|---|
| Forecasting Architectures | Compares **Independent Models** (each variable predicted alone) vs. **Physics Cascaded Models** (predicted AE fed forward into Kp/Dst), same live NOAA data. 4 inner tabs — functions still live directly in `home.py` (lines 1796–2006), not yet extracted to their own module. |
| Research Laboratory | The consolidated Bz/IMF, Kp, and AE labs — see below. |
| Physics Interpretation | `render_physics_interpretation_panel()`, `home.py:2007`. |
| Hypothesis Testing | Standalone version of the hypothesis tool, defined in `ae_research_lab.py:837` (predates that module's boundary, kept there and re-exported). |
| Storm Backtest | `storm_lab.py:90` |
| Storm Learning | `storm_lab.py:213` |

**The "Research Laboratory" tab is a single consolidated home for three labs, selected by variable** (`RESEARCH_LAB_VARIABLES` dict, `home.py:2100`) — replacing three copies that used to be buried in different pages (IMF under Heliosphere, AE under Analytics, etc.). Dst is deliberately excluded here — its storm-tested evidence lives in Storm Backtest/Storm Learning instead, and a dedicated Dst lab is deferred until there's a specific hypothesis worth building ablation tooling for.

| Variable | Module | Lines | Sub-tabs |
|---|---|---|---|
| Bz / IMF | `imf_research_lab.py` | 1,314 | 🔬 Bz Optimization Study, Training Runs, Horizon Analysis, Model Comparison, Feature Engineering, Physics Experiments, Sequence Models, Hyperparameter Tuning, Hypothesis Testing (9 tabs) |
| Kp | `kp_research_lab.py` | 2,159 | 🔬 Kp Optimization Study, Model Comparison, Feature Ablation, Physics Experiments, Sequence Models, Experiment Tracking, Hypothesis Testing, Visualization (8 tabs) |
| AE | `ae_research_lab.py` | 1,114 | 🔬 AE Optimization Study, Model Comparison, Feature Ablation, Physics Experiments, Sequence Models, Horizon Analysis, Experiment Tracking, Hypothesis Testing (8 tabs) |

Each lab's own module docstring notes it was "extracted verbatim from `dashboard/home.py`" — same extraction pattern as the Command Centre split, applied to the research side.

**The "Optimization Study" tab is the flagship in all three labs**, and each one drives through a shared UI shell (`render_automl_shell`, imported from `shared_ui.py`) rather than three separately-built interfaces — the three studies were originally built independently and later consolidated onto one AutoML runner, so the UI is now consistent across Kp/IMF/AE even though the underlying experiments (grids, feature sets, physics constraints) differ per variable.

**`imf_research_lab.py` deserves its own callout on cadence.** It deliberately operates on raw *minute-level* IMF data, not the hourly-resampled data production trains on — Bz's physics genuinely lives at minute timescales (e.g. "how many consecutive minutes has Bz been southward" only means something at that resolution; resampling to hourly first would destroy the exact information the feature exists to capture). This is a real, documented, intentional divergence from production's data contract — not an oversight — and a good example if asked "did you ever choose a different data resolution for a good physical reason."

**`storm_lab.py` (343 lines) — the honesty check.** Its own docstring states the problem plainly: every accuracy number this project has ever reported (R², skill scores, calibration) was computed against live 2026 NOAA data, which has been geomagnetically quiet the entire time (Kp never exceeded ~3, Dst never dropped below -14 nT). A forecasting engine's hardest real job is a storm, and this had genuinely never been checked against one. Two deliberately separate tools, because a single "train small, test on storm" design would have conflated two different questions:
- **Storm Backtest** (`render_storm_backtest_tab`, line 90) — does the *existing, already-trained production model* generalize to a storm it never saw? No retraining.
- **Storm Learning** (`render_storm_learning_tab`, line 213) — if a *new* model is trained on real historical storm data, does that measurably help? Evaluated only, never deployed.

Dev-only in both cases: nothing here writes to `data/forecasts/` or the jobs database, and nothing here is ever deployed to production.

---

## 6. Real engineering stories (genuinely good interview material)

These aren't hypothetical — each was found and fixed with verified before/after evidence, which is worth more in an interview than a rehearsed story:

- **Non-atomic writes causing data corruption.** Every parquet writer used to call `to_parquet()` directly on the live file path. `to_parquet()` isn't atomic — it writes its footer last — so a reader landing mid-write saw a truncated file (`ArrowInvalid: magic bytes not found`). Fixed by routing every write through temp-file + `os.replace()`, the same pattern the JSON writers already used correctly.
- **A silent 188-second, 2.1GB render-blocking bug.** A CSV download button was evaluating `.to_csv()` on a 7-million-row DataFrame as a plain Python function argument — which Streamlit runs on *every single page render*, not just on click, because `st.tabs()` executes every tab's code regardless of which is visible. That's very likely why several tabs had been silently blank for a long time before this was found — not broken, just never reached. Measured and confirmed with a direct timing test before fixing it.
- **A cache TTL that collided with its own refresh interval.** A cache was set to expire at the same interval as the auto-refresh timer that would invalidate it anyway — so it was barely caching at all. Decoupling the two constants was the actual fix, not adding more caching.
- **Unbounded historical data growth.** `forecast_snapshot_history.parquet` was designed to grow forever (by original, documented design). At 7M+ rows it started blocking dashboard renders outright. Fixed with a genuine architectural change, not a workaround: log only the 10 headline forecasts (not every horizon), and enforce a rolling 72-hour retention window on every write, so the file's size is now self-correcting rather than needing a separate cleanup job.

---

*Generated 2026-08-14. Every file path and line number above was checked against the repo at the time of writing — line numbers will drift as the code changes, but the architecture described (config-driven models, staged engine, page-per-function dashboard, isolated research labs) is stable.*
