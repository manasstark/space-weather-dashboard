"""Builds the Reports tab's downloadable PDF from hourly_report.parquet.

Deliberately plain — a printed operational log, not a designed document:
monospace type, dash rules, black-on-white tables. Mirrors the exact data
dashboard/lib/command_centre.py's Reports section shows on screen; this
module only renders it to a page. See storage.append_report_rows for what
each row means and storage.REPORT_HISTORY_RETENTION_DAYS for the window.
"""

import io

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MONO = "monospace"
_ROWS_PER_PAGE = 42
_PAGE_SIZE = (11, 8.5)  # landscape letter
_TABLE_COLUMNS = ["HOUR", "VARIABLE", "HORIZON", "PREDICTED", "ACTUAL", "ERROR", "CONFIDENCE", "VS. PERSISTENCE", "FLAGS"]
# Fractions of table width, sized for each column's actual worst-case
# content ("underperformed_persistence,low_confidence" is the long pole) —
# ax.table doesn't auto-size columns, so without this the last two columns
# overlap illegibly.
_TABLE_COL_WIDTHS = [0.06, 0.09, 0.07, 0.09, 0.09, 0.08, 0.09, 0.13, 0.30]


def _fmt(value, decimals=2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{decimals}f}"


def _fmt_signed(value, decimals=2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def _dash_rule(width: int = 100) -> str:
    return "-" * width


def _vs_persistence(row) -> str:
    persistence_abs_error = row.get("persistence_abs_error")
    if persistence_abs_error is None or pd.isna(persistence_abs_error):
        return "—"
    return "beat persistence" if row["abs_error"] <= persistence_abs_error else "underperformed"


def _hours_expected_today(now: pd.Timestamp) -> int:
    """Number of hour boundaries already closed today by wall-clock time —
    see the day-header 'ran X of Y expected hours' note this mirrors in
    command_centre._render_downloads_tab."""
    return now.hour


def _day_summaries(reports: pd.DataFrame, now: pd.Timestamp) -> list:
    """One entry per calendar day present in reports: (date, hours_covered,
    hours_expected, pending_note or None). Chronological, oldest first —
    a report reads like a log, not a live dashboard."""
    last_closed_boundary = now.floor("h")
    all_boundaries = set(reports["valid_end_ts"].dt.floor("h"))
    summaries = []
    for date, day_group in reports.groupby(reports["valid_end_ts"].dt.date):
        hours_covered = day_group["valid_end_ts"].dt.floor("h").nunique()
        is_today = date == now.date()
        hours_expected = _hours_expected_today(now) if is_today else 24
        pending_note = None
        if is_today and last_closed_boundary not in all_boundaries:
            pending_note = (
                f"Hour ending {last_closed_boundary.strftime('%H:%M')} UTC still evaluating "
                "- not yet included in this report."
            )
        summaries.append((date, hours_covered, hours_expected, pending_note))
    return sorted(summaries, key=lambda s: s[0])


def _title_page(pdf, reports: pd.DataFrame, day_summaries: list, now: pd.Timestamp, retention_days: int) -> None:
    fig = plt.figure(figsize=_PAGE_SIZE)
    fig.patch.set_facecolor("white")
    lines = []
    lines.append("SPACE WEATHER FORECAST -- OPERATIONAL PERFORMANCE REPORT")
    lines.append(_dash_rule())
    date_min = reports["valid_end_ts"].min().strftime("%d %b %Y")
    date_max = reports["valid_end_ts"].max().strftime("%d %b %Y")
    lines.append(f"Report window   : {date_min} to {date_max}  (rolling {retention_days}-day window)")
    lines.append(f"Generated       : {now.strftime('%d %b %Y %H:%M UTC')}")
    lines.append(f"Days covered    : {len(day_summaries)}")
    lines.append(f"Rows (var-hour) : {len(reports)}")
    lines.append("")
    lines.append("DAILY COVERAGE")
    lines.append(_dash_rule(60))
    for date, covered, expected, pending in day_summaries:
        lines.append(f"  {date.strftime('%d %b %Y')}  -- ran {covered:>2d} of {expected:>2d} expected hours")
        if pending:
            lines.append(f"      NOTE: {pending}")
    lines.append("")

    flagged = reports[reports["flags"].fillna("") != ""]
    lines.append(f"HOURS WITH FLAGS: {len(flagged)} of {len(reports)} variable-hours")
    lines.append(_dash_rule(60))
    if flagged.empty:
        lines.append("  None -- every evaluated hour was clean this window.")
    else:
        for flag_name in ("low_confidence", "underperformed_persistence"):
            count = flagged["flags"].str.contains(flag_name, na=False).sum()
            if count:
                lines.append(f"  {flag_name:<28s} {count:>3d} occurrence(s)")

    y = 0.95
    for line in lines:
        weight = "bold" if line.isupper() and line.strip() else "normal"
        fig.text(0.06, y, line, fontsize=10, fontfamily=_MONO, fontweight=weight, va="top")
        y -= 0.028
    pdf.savefig(fig)
    plt.close(fig)


def _predicted_vs_actual_page(pdf, reports: pd.DataFrame) -> None:
    """One page, every headline variable, however many points exist —
    a variable with 5 evaluated hours plots a 5-point line; a variable
    with 1 plots 1 point. No minimum-data gate: the chart is honest about
    however much history has actually accumulated so far."""
    variables = sorted(reports["variable"].unique())
    nrows = 2
    ncols = max(1, -(-len(variables) // nrows))  # ceil division — fits any headline-variable count on one page

    # matplotlib's datetime autoscale picks a wildly wrong default range
    # (multi-year) when a subplot has only one or two points — pin every
    # subplot to the same real report window instead, padded by an hour.
    window_start = reports["valid_end_ts"].min() - pd.Timedelta(hours=1)
    window_end = reports["valid_end_ts"].max() + pd.Timedelta(hours=1)

    fig, axes = plt.subplots(nrows, ncols, figsize=_PAGE_SIZE)
    fig.suptitle("PREDICTED VS. ACTUAL -- BY VARIABLE", fontfamily=_MONO, fontweight="bold", fontsize=11)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, variable in zip(axes_flat, variables):
        sub = reports[reports["variable"] == variable].sort_values("valid_end_ts")
        ax.plot(sub["valid_end_ts"], sub["predicted_value"], marker="o", markersize=2, linewidth=1, label="Predicted")
        ax.plot(sub["valid_end_ts"], sub["actual_value"], marker="o", markersize=2, linewidth=1, linestyle="--", label="Actual")
        ax.set_xlim(window_start, window_end)
        ax.set_title(f"{variable.upper()} (n={len(sub)})", fontfamily=_MONO, fontsize=8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
        ax.legend(fontsize=5, loc="upper left")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    for ax in axes_flat[len(variables):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def _day_table_pages(pdf, date, day_group: pd.DataFrame, covered: int, expected: int, pending_note) -> None:
    rows = []
    for _, r in day_group.sort_values(["valid_end_ts", "variable"]).iterrows():
        rows.append([
            r["valid_end_ts"].strftime("%H:%M"),
            str(r["variable"]).upper(),
            str(r["horizon"]),
            _fmt(r["predicted_value"]),
            _fmt(r["actual_value"]),
            _fmt_signed(r["error"]),
            str(r.get("confidence_category") or "—"),
            _vs_persistence(r),
            str(r.get("flags") or "—"),
        ])

    total_pages = max(1, (len(rows) + _ROWS_PER_PAGE - 1) // _ROWS_PER_PAGE)
    for page_num in range(total_pages):
        page_rows = rows[page_num * _ROWS_PER_PAGE:(page_num + 1) * _ROWS_PER_PAGE]
        fig = plt.figure(figsize=_PAGE_SIZE)

        header_lines = [f"DAY -- {date.strftime('%d %b %Y')}  (ran {covered} of {expected} expected hours)"]
        if total_pages > 1:
            header_lines[0] += f"   [page {page_num + 1} of {total_pages}]"
        if pending_note and page_num == total_pages - 1:
            header_lines.append(f"NOTE: {pending_note}")
        header_lines.append(_dash_rule())

        ax_text = fig.add_axes((0.04, 0.90, 0.92, 0.08))
        ax_text.axis("off")
        y = 1.0
        for line in header_lines:
            ax_text.text(0, y, line, fontsize=9, fontfamily=_MONO, va="top")
            y -= 0.4

        ax = fig.add_axes((0.04, 0.04, 0.92, 0.82))
        ax.axis("off")
        table = ax.table(
            cellText=page_rows, colLabels=_TABLE_COLUMNS, colWidths=_TABLE_COL_WIDTHS,
            loc="upper center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.3)
        for (i, _j), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.4)
            cell.set_text_props(fontfamily=_MONO, fontweight="bold" if i == 0 else "normal")

        pdf.savefig(fig)
        plt.close(fig)


def build_report_pdf(reports: pd.DataFrame, retention_days: int) -> bytes:
    """Renders hourly_report.parquet to a plain, monospace PDF: title +
    coverage summary, predicted-vs-actual charts per variable, then one
    table per calendar day (chronological, oldest first -- a report reads
    like a log). Returns PDF bytes ready for a download button."""
    from matplotlib.backends.backend_pdf import PdfPages

    reports = reports.copy()
    reports["valid_end_ts"] = pd.to_datetime(reports["valid_end"], utc=True, errors="coerce")
    reports = reports.dropna(subset=["valid_end_ts"])
    now = pd.Timestamp.now(tz="UTC")

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        summaries = _day_summaries(reports, now)
        _title_page(pdf, reports, summaries, now, retention_days)
        _predicted_vs_actual_page(pdf, reports)
        for date, covered, expected, pending_note in summaries:
            day_group = reports[reports["valid_end_ts"].dt.date == date]
            _day_table_pages(pdf, date, day_group, covered, expected, pending_note)

    return buffer.getvalue()
