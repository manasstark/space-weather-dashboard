"""Pure-Python duplicate of dashboard/home.py's variable_meaning_and_risk
— intentionally NOT an import: home.py executes module-level Streamlit
script code at import time (st.set_page_config, live data loads, page
dispatch), so importing it from the engine or live_update.py (neither of
which run inside a Streamlit runtime) would attempt to run a full
Streamlit script outside of one. Same thresholds, kept in sync manually
— if home.py's bands change, update both.
"""


def classify_current_reading(column: str, value) -> tuple:
    if value is None:
        return "No data", "N/A"

    value = float(value)

    if column == "speed":
        if value < 400:
            return "Slow solar wind", "Usually quiet"
        if value < 500:
            return "Moderate speed", "Normal/active"
        if value < 700:
            return "Fast solar wind", "Storm possible with southward Bz"
        return "Very fast wind", "Enhanced storm potential"

    if column == "density":
        if value < 5:
            return "Low density", "Weak pressure"
        if value < 10:
            return "Moderate density", "Normal solar wind"
        if value < 30:
            return "High density", "Compression possible"
        return "Very high density", "Shock/CME sheath possible"

    if column == "temperature":
        if value < 50000:
            return "Cool wind", "Usually quiet"
        if value < 150000:
            return "Typical wind", "Normal"
        if value < 500000:
            return "Hot wind", "Disturbed flow possible"
        return "Very hot plasma", "Shock/CME heating possible"

    if column in ("bz", "bz_gsm"):
        if value > 0:
            return "Northward IMF", "Low coupling"
        if value >= -5:
            return "Weak southward IMF", "Minor activity possible"
        if value >= -10:
            return "Moderate southward IMF", "Storm possible"
        if value >= -20:
            return "Strong southward IMF", "Strong storm coupling"
        return "Extreme southward IMF", "Severe storm potential"

    if column == "kp":
        if value <= 3:
            return "Quiet", "Normal"
        if value < 5:
            return "Active", "Unsettled field"
        if value < 6:
            return "G1 storm", "Minor storm"
        if value < 7:
            return "G2 storm", "Moderate storm"
        if value < 8:
            return "G3 storm", "Strong storm"
        return "G4-G5 storm", "Severe/extreme storm"

    if column == "dst":
        if value > -30:
            return "Quiet", "Low storm activity"
        if value > -50:
            return "Weak storm", "Minor ring current"
        if value > -100:
            return "Moderate storm", "Storm underway"
        if value > -200:
            return "Intense storm", "Strong disturbance"
        return "Superstorm", "Extreme disturbance"

    if column == "ae":
        if value < 100:
            return "Quiet", "Low auroral activity"
        if value < 300:
            return "Unsettled", "Moderate auroral activity"
        if value < 500:
            return "Active", "Elevated auroral activity"
        return "Storm-level", "High auroral activity"

    return "", ""
