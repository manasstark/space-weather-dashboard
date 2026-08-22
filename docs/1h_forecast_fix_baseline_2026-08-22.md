# 1h/interval forecast skew fix — baseline snapshot (2026-08-22)

Saved so the next check (2026-08-25, "Monday") has an exact before/after to compare
against, rather than relying on memory of what the numbers were.

## Context

Every headline (1h/interval) forecast was scoring "No Skill Over Persistence" in
`skill_scores.json`. Root-caused to a train/serve input-freshness mismatch:
`continuous_reissue` jobs are often created within a minute of a new hour starting,
so the model's live "current value" feature was frequently built from only 1-2
minutes of real data instead of a full 60-minute average like training saw.

**Fix**: `src/swdss/models/predict.py`'s new `_latest_complete_hour` helper falls
back to the last genuinely complete hour's values for the model's input when the
current hour hasn't finished yet — without changing `observed_at`/`predicted_for`
(the actual forecast target hour stays correctly anchored to "now", not shifted
backward). Scoped to the shared `predict()` path only — persistence's own baseline
and Kp (`predict_kp_interval`, separate code path) were deliberately left untouched.

Also fixed same day: `package_history.parquet` unbounded growth (dedup on real
state change instead of every cycle), corrupted-parquet self-healing
(`storage._safe_read_parquet`), a `flare_cme_predict.py` crash on zero scorable
SHARP regions, and a Streamlit `cache_data`/`cachetools` KeyError race
(`retry_on_cache_race` in `data_helpers.py`).

`live_update` has been running continuously under the fixed code since
**2026-08-22 09:00 UTC** (~14:37 IST).

## Pre-fix baseline (full history, from `README.md` → Known Limitations)

| Variable | Skill vs. persistence | Beats persistence (all-time) |
|---|---|---|
| Kp | −4.90 | 5/158 (3.2%) |
| Density | −4.07 | 93/364 (25.5%) |
| Temperature | −1.06 | 127/364 (34.9%) |
| Speed | −0.93 | 97/364 (26.6%) |
| Bt | −0.90 | 104/359 (29.0%) |
| Bx | −0.65 | 131/359 (36.5%) |
| Bz | −0.57 | 122/359 (34.0%) |
| Dst | −0.28 | 190/374 (50.8%) |
| By | −0.13 | 144/359 (40.1%) |

## Snapshot as of 2026-08-22 20:46 UTC (~11.75h into the fixed run)

### Aggregate `skill_scores.json` (full history — still mostly pre-fix data, diluted)

| Variable | n | skill | verdict |
|---|---|---|---|
| Kp (interval) | 161 | −4.933 | No Skill Over Persistence |
| Density | 371 | −4.053 | No Skill Over Persistence |
| Temperature | 371 | −1.049 | No Skill Over Persistence |
| Speed | 371 | −0.901 | No Skill Over Persistence |
| Bt | 366 | −0.892 | No Skill Over Persistence |
| Bx | 366 | −0.642 | No Skill Over Persistence |
| Bz | 366 | −0.569 | No Skill Over Persistence |
| Dst | 380 | −0.279 | No Skill Over Persistence |
| By | 366 | −0.127 | No Skill Over Persistence |

### Fresh evaluations completed since 09:00 UTC today, i.e. genuinely under the fix (n=93 total)

| Variable | n | beat-persistence rate | model MAE | persistence MAE | beats on MAE? |
|---|---|---|---|---|---|
| Bz | 11 | 63.6% | 0.9609 | 0.9921 | **YES** |
| Dst | 12 | 50.0% | 3.0760 | 2.6667 | no |
| By | 11 | 54.5% | 1.6747 | 1.6645 | no |
| Bt | 11 | 45.5% | 0.3286 | 0.2513 | no |
| Bx | 11 | 45.5% | 1.2008 | 1.0281 | no |
| Temperature | 11 | 45.5% | 13555.02 | 11519.31 | no |
| Speed | 11 | 36.4% | 9.3017 | 8.4624 | no |
| Density | 11 | 27.3% | 0.4519 | 0.3473 | no |
| Kp | 4 | 0.0% | 0.2422 | 0.0000 | no |

## Read as of this snapshot

Win-rate (fraction of individual hours where the model beat persistence) improved
for every variable except Density and Dst (roughly flat) and Kp (unchanged,
expected — separate code path, not touched by this fix). But on the stricter
mean-error basis, only **Bz** currently beats persistence outright. n=11-12 is
still small enough that a couple of bad predictions can swing the mean a lot, and
conditions have been geomagnetically quiet the whole run, which makes the naive
"assume no change" baseline unusually strong right now — both likely still
suppressing how much of the real improvement is visible yet.

**Not a verdict — a checkpoint.** The real read is whichever way this moves by
Monday, once each variable has meaningfully more than 11-12 fresh samples.

## What to do Monday

Ask to "check the skill scores and compare against the 2026-08-22 baseline" —
pull `data/forecasts/current/skill_scores.json` and
`data/forecasts/history/evaluation_history.parquet` filtered to
`completed_at >= 2026-08-22T09:00:00Z`, and compare against the two tables above.
