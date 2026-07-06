"""Kyoto WDC "Quicklook" real-time AE graph — approximate, image-based AE
estimation for immediate visual comparison right after a prediction
completes.

This is explicitly NOT the official verification source. It exists only
so a user doesn't have to wait the ~10-20 days Kyoto WDC's official
digital AE values (swdss.ingest.kyoto_ae) take to publish before getting
SOME sense of how a prediction compares. The estimate this module
produces is stored separately (never overwrites a job's official
`actual_value`/`verification_status`) and the official digital data
remains the only authoritative verification source.

Image source:
    https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/{YYYYMM}/rtae_{YYYYMMDD}.png
A fixed 700x450 auto-generated two-panel plot (AU/AL on top, AE/AO on
bottom, both sharing this one image), updated continuously through the
day. The pixel calibration constants below were measured directly off
this fixed template's gridlines/axis ticks and cross-validated against
Kyoto's own published official digital hourly means for two separate
days — mean absolute error came out to ~3-5 nT, confirming this is a
reasonable *approximate* estimate, not a substitute for the official
data.
"""

import io

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

QUICKLOOK_BASE_URL = "https://wdc.kugi.kyoto-u.ac.jp/ae_realtime"

# Pixel calibration for the bottom (AE/AO) panel of the fixed 700x450
# template: x=PLOT_X0 is hour 0 (00 UT), x=PLOT_X1 is hour 24; y=PLOT_Y0
# is the panel's top gridline (Y_TOP_VALUE nT), y=PLOT_Y1 is the bottom
# gridline (Y_BOTTOM_VALUE nT).
PLOT_X0, PLOT_X1 = 81, 648
PLOT_Y0, PLOT_Y1 = 254, 395
Y_TOP_VALUE, Y_BOTTOM_VALUE = 2000, -500


def _x_for_hour(hour: float) -> float:
    return PLOT_X0 + hour * (PLOT_X1 - PLOT_X0) / 24


def _y_to_ae(y: float) -> float:
    return Y_TOP_VALUE + (y - PLOT_Y0) * (Y_BOTTOM_VALUE - Y_TOP_VALUE) / (PLOT_Y1 - PLOT_Y0)


def _y_for_ae(value: float) -> float:
    return PLOT_Y0 + (value - Y_TOP_VALUE) * (PLOT_Y1 - PLOT_Y0) / (Y_BOTTOM_VALUE - Y_TOP_VALUE)


def _is_curve_pixel(px) -> bool:
    """Gridlines, the axis border, and text are always exactly grayscale
    (r==g==b); every plotted data pixel — the orange/yellow AE/AO fill
    *and* the distinct teal/cyan marker Kyoto uses for the most recent
    (still-provisional) minutes — is not. Testing "not grayscale" catches
    both without needing to hardcode every color Kyoto's plotter uses.

    An earlier, stricter version of this check only matched warm
    (orange/yellow) tones and silently missed that teal "latest reading"
    marker, which made the estimate wrongly report "no data yet" for an
    hour that Kyoto had actually already started plotting.
    """
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    return not (r == g == b)


def quicklook_image_url(date: pd.Timestamp) -> str:
    return f"{QUICKLOOK_BASE_URL}/{date.strftime('%Y%m')}/rtae_{date.strftime('%Y%m%d')}.png"


def fetch_quicklook_image(date: pd.Timestamp) -> Image.Image:
    response = requests.get(quicklook_image_url(date), timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def fetch_quicklook_image_with_meta(date: pd.Timestamp) -> tuple:
    """Same fetch as fetch_quicklook_image, but also returns when Kyoto's
    server last regenerated this day's graph (the response's Last-Modified
    header), so the dashboard can show how stale the image is. Returns
    (image, graph_created_or_None) — None only if the server omits the
    header, which has not been observed in practice but is handled
    defensively.
    """
    response = requests.get(quicklook_image_url(date), timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    last_modified = response.headers.get("Last-Modified")
    graph_created = None
    if last_modified:
        graph_created = pd.Timestamp(last_modified)
        if graph_created.tzinfo is not None:
            graph_created = graph_created.tz_convert(None)
    return image, graph_created


def estimate_ae_at_hour(image: Image.Image, hour: float) -> float | None:
    """Estimates the AE value at `hour` (0-24, UT) within one day's
    quicklook image, by averaging the curve's pixel height across that
    hour's column range — approximating an hourly mean the same way the
    official digital data reports it, rather than reading a single,
    noisier instantaneous pixel column.
    """
    arr = np.array(image)
    x0 = int(round(_x_for_hour(hour)))
    x1 = int(round(_x_for_hour(min(hour + 1, 24))))
    if x1 <= x0:
        x1 = x0 + 1

    values = []
    for x in range(x0, min(x1, arr.shape[1])):
        column = arr[PLOT_Y0 : PLOT_Y1 + 1, x]
        for i, px in enumerate(column):
            if _is_curve_pixel(px):
                values.append(_y_to_ae(PLOT_Y0 + i))
                break
    if not values:
        return None
    return sum(values) / len(values)


# Thresholds for curve_quality_at_hour's confidence label. Coverage is the
# fraction of the target hour's 60 minutes for which the AE curve is
# actually present in the graph — confidence is a direct read of that
# fraction: <40% Low, 40-80% Moderate, >80% High. A prediction's target
# hour is only ever partially drawn until Kyoto's real-time graph catches
# up to (or past) that hour, so low coverage here almost always means
# "not enough of the hour has been plotted yet", not a detection failure.
QUICKLOOK_CONFIDENCE_LOW_MAX = 0.4
QUICKLOOK_CONFIDENCE_MODERATE_MAX = 0.8


def curve_quality_at_hour(image: Image.Image, hour: float) -> dict:
    """Image-processing quality diagnostics for the same hour window
    estimate_ae_at_hour scans, computed independently and never fed back
    into the AE estimate itself. Used only to derive a confidence label
    and an estimated range for the dashboard.

    Returns {"confidence": "high"|"moderate"|"low", "coverage": float,
    "low": float|None, "high": float|None}.
    """
    arr = np.array(image)
    x0 = int(round(_x_for_hour(hour)))
    x1 = int(round(_x_for_hour(min(hour + 1, 24))))
    if x1 <= x0:
        x1 = x0 + 1

    values = []
    columns_scanned = 0
    columns_with_curve = 0
    for x in range(x0, min(x1, arr.shape[1])):
        columns_scanned += 1
        column = arr[PLOT_Y0 : PLOT_Y1 + 1, x]
        for i, px in enumerate(column):
            if _is_curve_pixel(px):
                values.append(_y_to_ae(PLOT_Y0 + i))
                columns_with_curve += 1
                break

    if not values or columns_scanned == 0:
        return {"confidence": "low", "coverage": 0.0, "low": None, "high": None}

    coverage = columns_with_curve / columns_scanned
    std = float(np.std(values)) if len(values) > 1 else 0.0
    mean_val = sum(values) / len(values)
    margin = max(std, 5.0)

    if coverage > QUICKLOOK_CONFIDENCE_MODERATE_MAX:
        confidence = "high"
    elif coverage >= QUICKLOOK_CONFIDENCE_LOW_MAX:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "confidence": confidence,
        "coverage": coverage,
        "low": mean_val - margin,
        "high": mean_val + margin,
    }


def estimate_kyoto_quicklook_ae(target_hour) -> tuple:
    """Fetches the quicklook graph covering target_hour's UT day and
    estimates the AE value for that hour. Returns
    (estimated_ae_or_None, image_url) — the URL is returned too so the
    dashboard can display the same image the estimate came from.
    """
    ts = pd.Timestamp(target_hour)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    day = ts.normalize()
    url = quicklook_image_url(day)
    image = fetch_quicklook_image(day)
    hour = ts.hour + ts.minute / 60
    return estimate_ae_at_hour(image, hour), url


def estimate_kyoto_quicklook_ae_full(target_hour) -> dict:
    """Combined estimate + quality read using a single image fetch: the
    same AE estimate estimate_kyoto_quicklook_ae would produce, plus the
    image-processing diagnostics (confidence, estimated range) and the
    graph's Last-Modified time, all for dashboard presentation. Does not
    change how the estimate itself is computed.
    """
    ts = pd.Timestamp(target_hour)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    day = ts.normalize()
    url = quicklook_image_url(day)
    image, graph_created = fetch_quicklook_image_with_meta(day)
    hour = ts.hour + ts.minute / 60
    quality = curve_quality_at_hour(image, hour)
    return {
        "estimate": estimate_ae_at_hour(image, hour),
        "image_url": url,
        "graph_created": graph_created,
        "confidence": quality["confidence"],
        "range_low": quality["low"],
        "range_high": quality["high"],
        # Fraction of the target hour's 60 minutes that Kyoto had actually
        # drawn a curve for at capture time. Low coverage (e.g. the graph
        # was only regenerated 15 minutes into the hour) means the
        # "estimate" above is really an average of that partial slice, not
        # a stand-in for the eventual full-hour value — this is reported
        # honestly rather than silently extrapolated.
        "hour_coverage": quality["coverage"],
    }


def annotate_quicklook_image(
    image: Image.Image, target_hour, predicted_value: float, estimated_value: float | None
) -> Image.Image:
    """Draws the target-time marker, predicted-AE line, and estimated-AE
    point onto a COPY of the quicklook image for visual comparison — pure
    presentation, drawn after the numeric estimate has already been
    computed by estimate_ae_at_hour. Never mutates the original image.
    """
    ts = pd.Timestamp(target_hour)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    hour = ts.hour + ts.minute / 60

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x = _x_for_hour(hour)

    def _label(pos, text, color):
        # Small white-filled backing box so the label stays legible over
        # the graph's own curve/gridlines.
        tx, ty = pos
        bbox = draw.textbbox((tx, ty), text)
        pad = 2
        draw.rectangle(
            [(bbox[0] - pad, bbox[1] - pad), (bbox[2] + pad, bbox[3] + pad)],
            fill=(255, 255, 255),
            outline=color,
        )
        draw.text(pos, text, fill=color)

    # Vertical dashed line at the target time, spanning the AE/AO panel.
    y = PLOT_Y0
    while y < PLOT_Y1:
        draw.line([(x, y), (x, min(y + 6, PLOT_Y1))], fill=(0, 90, 220), width=2)
        y += 10
    _label((x + 4, PLOT_Y0 - 14), "Target Time", (0, 90, 220))

    # Horizontal line at the predicted AE value, if it falls within the
    # panel's plotted range.
    pred_y = _y_for_ae(predicted_value)
    if PLOT_Y0 <= pred_y <= PLOT_Y1:
        draw.line([(PLOT_X0, pred_y), (PLOT_X1, pred_y)], fill=(0, 90, 220), width=2)
        _label((PLOT_X0 + 4, pred_y - 16), "Prediction", (0, 90, 220))

    # Highlight the estimated point read off the curve.
    if estimated_value is not None:
        est_y = _y_for_ae(estimated_value)
        r = 5
        draw.ellipse([(x - r, est_y - r), (x + r, est_y + r)], outline=(210, 0, 0), width=2)
        _label((x + r + 4, est_y - 6), "Estimated AE", (210, 0, 0))

    return annotated
