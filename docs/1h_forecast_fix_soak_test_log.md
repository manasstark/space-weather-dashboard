# 1h/interval forecast skew fix — soak test log

Running log of every checkpoint in this investigation, oldest first — not a single
snapshot. Renamed from `1h_forecast_fix_baseline_2026-08-22.md` once it became clear
this would span multiple fix versions and multiple check-ins, not one before/after.
Each new checkpoint gets appended below with its own dated section; nothing here is
ever overwritten, so the full history of what was tried and what actually happened
stays intact.

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

**Not a verdict — a checkpoint.** The real read is whichever way this moves once
each variable has meaningfully more than 11-12 fresh samples.

---

## 2026-08-24 — v1 reversed at n=23 (do not trust the 08-22 checkpoint above)

Live_update kept running from the 08-22 checkpoint through Monday. At n=23/variable
(roughly double the first checkpoint), **every variable's win-rate had dropped**,
including Bz — the one variable that had genuinely beaten persistence at n=11-12.

| Variable | n=11-12 win-rate (08-22) | n=23 win-rate (08-24) | n=11-12 beats MAE? | n=23 beats MAE? |
|---|---|---|---|---|
| Bz | 63.6% | 47.8% | **YES** | no |
| By | 54.5% | 47.8% | no | no |
| Dst | 50.0% | 43.5% | no | no |
| Bt | 45.5% | 43.5% | no | no |
| Temperature | 45.5% | 34.8% | no | no |
| Bx | 45.5% | 39.1% | no | no |
| Speed | 36.4% | 34.8% | no | no |
| Density | 27.3% | 26.1% | no | no |
| Kp | 0.0% | 0.0% | no | no |

**Conclusion at the time: the early "improvement" was mostly noise from a small
sample regressing back down, not a real trend.** This is exactly why nothing here
gets treated as proven at n=11-12 — see the general caution repeated at every
checkpoint below.

### Root cause of v1's failure (why "v1" needed a "v2" at all)

v1 (the fix described in the Context section above) kept `observed_at` pinned to
the true latest hour while only the model's *feature row* fell back to the prior
complete hour. That meant the model reasoned from data an hour older than its own
timestamp while still being asked to predict a full hour past the *current* hour —
a 2-hour real gap between the data used and the target that training never saw
(training only ever saw exact 1-hour gaps). Confirmed live: a job created at 18:06
was using 17:00's data while targeting 19:00.

Separately, and worse: **persistence's own anchor was untouched by v1**, so it kept
reading whatever was freshest — confirmed live, at a real evaluation moment
persistence's anchor value exactly matched the raw partial-hour value v1's own fix
was specifically built to avoid feeding the model. Persistence was quietly getting
a full hour's freshness advantage over the model for the same forecast — not a fair
"zero-transformation on the same input" comparison, a different, easier input.

## 2026-08-24 — v2 shipped (what actually changed in the model/pipeline)

Two code changes, both correcting the asymmetry found above:

1. **`src/swdss/models/predict.py`'s `_latest_complete_hour`** — now advances the
   feature row *and* its timestamp together. `observed_at` and `predicted_for` both
   fall back to the last complete hour when the current one hasn't finished, so the
   gap between the data used and the target is always exactly 1 hour, matching
   training exactly. (v1 only advanced the feature row, keeping `observed_at` at
   the true latest hour — see the root-cause section above for why that was wrong.)
2. **`src/swdss/engine/orchestrator.py`'s `_resolve_persistence_anchor`** — now
   anchors to the same last-complete-hour definition (`created_at`'s own hour,
   floored, minus one hour) instead of whatever's freshest. Verified live: the
   model's `current_value` and persistence's anchor now resolve to the identical
   number for the same forecast — neither side gets a freshness advantage anymore.

Honest tradeoff accepted: a forecast now only refreshes once per hour (right after
the previous hour closes) instead of within the first few minutes of a new one.

Checked before shipping: does a target hour that's technically "in the past"
relative to when the job was created let it resolve against an incomplete actual
value? No — `jobs.py`'s own finalize logic already gates on
`now >= target_hour + 1 hour` before ever calling `resolve_actual_value`,
independent of anything this fix touches, so evaluation still always waits for
genuine hour completion either way.

`live_update` was restarted under v2 code on **2026-08-25 ~13:03 IST (~07:33 UTC)**.

## 2026-08-25 — first v2 checkpoint (~24h in, n=11-12/variable)

| Variable | n | win-rate | model MAE | persistence MAE | beats persistence? |
|---|---|---|---|---|---|
| Speed | 12 | 83.3% | 5.0040 | 5.1549 | **YES** |
| Dst | 11 | 72.7% | 1.8400 | 2.7273 | **YES** (by a lot) |
| Bz | 12 | 58.3% | 0.3937 | 0.4378 | **YES** |
| Kp | 4 | 50.0% | 0.2083 | 0.3325 | **YES** |
| Temperature | 12 | 66.7% | 7228.96 | 6907.76 | no (close) |
| Density | 12 | 33.3% | 0.5500 | 0.3681 | no |
| Bt | 12 | 33.3% | 0.4331 | 0.4020 | no |
| Bx | 12 | 25.0% | 0.8444 | 0.8406 | no |
| By | 12 | 25.0% | 0.8399 | 0.4308 | no |

**4 of 9 variables now genuinely beat persistence — up from 0-1 at every previous
checkpoint (v1's own best moment was 1/9, and that one reversed).** The notable
detail: **Kp improved too, and Kp's prediction model was never touched by either
fix** — its only change was the persistence-anchor correction. That's a real
cross-check, not just noise on the variables the model-side fix touched directly:
it corroborates that the anchor-fairness theory was genuinely part of the problem,
independent of the model-input fix.

Same caution as every checkpoint above: n=11-12 is still a small sample — this is
the *first* genuinely promising v2 checkpoint, not proof v2 holds up. Given v1
looked like a clear win at a similar sample size and then reversed by n=23, this
result needs to survive a larger sample before being trusted.

## What to check next

Ask to "check the skill scores against the log" — pull
`data/forecasts/history/evaluation_history.parquet` filtered to
`completed_at >= 2026-08-24T18:40:00Z` (v2's ship time) and compare against the
2026-08-25 table above, the same way this checkpoint compared against 08-22's.
Specifically watch whether the 4 current winners (Speed, Dst, Bz, Kp) hold up as
n grows past ~23 — that's the sample size where v1's signal reversed, so it's the
number that actually matters, not just "more data is better."
