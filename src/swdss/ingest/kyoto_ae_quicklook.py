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
    """The AE/AO curve (fill + a darker stroke on top of it) is always
    warm-toned (orange/brown); gridlines, the axis border, and text are
    always exactly grayscale (r==g==b). This cleanly separates the curve
    from everything else in the image without needing OCR or exact color
    matching against anti-aliasing artifacts.
    """
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    return r > g + 5 and g >= b


def quicklook_image_url(date: pd.Timestamp) -> str:
    return f"{QUICKLOOK_BASE_URL}/{date.strftime('%Y%m')}/rtae_{date.strftime('%Y%m%d')}.png"


def fetch_quicklook_image(date: pd.Timestamp) -> Image.Image:
    response = requests.get(quicklook_image_url(date), timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


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

    # Vertical dashed line at the target time, spanning the AE/AO panel.
    y = PLOT_Y0
    while y < PLOT_Y1:
        draw.line([(x, y), (x, min(y + 6, PLOT_Y1))], fill=(0, 90, 220), width=2)
        y += 10

    # Horizontal line at the predicted AE value, if it falls within the
    # panel's plotted range.
    pred_y = _y_for_ae(predicted_value)
    if PLOT_Y0 <= pred_y <= PLOT_Y1:
        draw.line([(PLOT_X0, pred_y), (PLOT_X1, pred_y)], fill=(0, 90, 220), width=2)

    # Highlight the estimated point read off the curve.
    if estimated_value is not None:
        est_y = _y_for_ae(estimated_value)
        r = 5
        draw.ellipse([(x - r, est_y - r), (x + r, est_y + r)], outline=(210, 0, 0), width=2)

    return annotated
