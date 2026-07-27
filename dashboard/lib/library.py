"""Two independent, JSON-backed personal-collection features, extracted
verbatim from dashboard/home.py: Saved Events (bookmark a solar event
from the News Feed for later) and the Space Weather Concepts Library
(upload/organize reference documents by category). Grouped together
because both are simple CRUD-over-JSON-plus-a-dialog features with no
other module depending on their internals — dashboard/lib/event_explorer.py
imports the Saved Events functions (load_saved_events, save_event_record,
remove_saved_event, record_to_series) to render its own "Saved Solar
Events" dialog, since that dialog also needs Event Explorer helpers this
module deliberately doesn't own.
"""

import json

import pandas as pd
import streamlit as st

from swdss.paths import DATA_DIR, PROJECT_ROOT

from dashboard.lib.shared_ui import render_dialog_close_button

# ==================== Saved Events ====================

SAVED_EVENTS_PATH = DATA_DIR / "saved_events.json"


def load_saved_events() -> list[dict]:
    if not SAVED_EVENTS_PATH.exists():
        return []
    try:
        with open(SAVED_EVENTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def _write_saved_events(records: list[dict]) -> None:
    SAVED_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVED_EVENTS_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, default=str)


def save_event_record(row: pd.Series) -> bool:
    record = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            record[key] = value.isoformat()
        elif pd.isna(value):
            record[key] = None
        else:
            record[key] = value
    record["saved_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    records = load_saved_events()
    new_key = (record.get("timestamp_utc"), record.get("event_type"))
    if any((r.get("timestamp_utc"), r.get("event_type")) == new_key for r in records):
        return False

    records.append(record)
    _write_saved_events(records)
    return True


def remove_saved_event(index: int) -> None:
    records = load_saved_events()
    if 0 <= index < len(records):
        records.pop(index)
        _write_saved_events(records)


def record_to_series(record: dict) -> pd.Series:
    data = dict(record)
    if data.get("timestamp_utc"):
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    return pd.Series(data)


# ==================== Space Weather Concepts Library ====================

LIBRARY_DIR = DATA_DIR / "library"
LIBRARY_INDEX_PATH = DATA_DIR / "library_index.json"
LIBRARY_CATEGORIES = ["Concepts", "Articles", "Research Papers"]
LIBRARY_CATEGORY_COLORS = {
    "Concepts": "#1f4a7a",
    "Articles": "#1f7a3a",
    "Research Papers": "#5a1f7a",
}


def load_library_index() -> list[dict]:
    if not LIBRARY_INDEX_PATH.exists():
        return []
    try:
        with open(LIBRARY_INDEX_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def _write_library_index(records: list[dict]) -> None:
    LIBRARY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIBRARY_INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, default=str)


def add_library_document(category: str, title: str, uploaded_file) -> None:
    category_dir = LIBRARY_DIR / category.lower().replace(" ", "_")
    category_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
    file_path = category_dir / stored_name
    with open(file_path, "wb") as out_file:
        out_file.write(uploaded_file.getbuffer())

    records = load_library_index()
    records.append(
        {
            "title": title or uploaded_file.name,
            "category": category,
            "filename": uploaded_file.name,
            "stored_path": str(file_path.relative_to(PROJECT_ROOT)),
            "added_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    )
    _write_library_index(records)


def remove_library_document(index: int) -> None:
    records = load_library_index()
    if 0 <= index < len(records):
        record = records.pop(index)
        stored_path = PROJECT_ROOT / record.get("stored_path", "")
        if stored_path.exists():
            try:
                stored_path.unlink()
            except OSError:
                pass
        _write_library_index(records)


@st.dialog("Space Weather Concepts", width="large", dismissible=False)
def show_space_weather_library() -> None:
    render_dialog_close_button("close_library")

    records = load_library_index()
    tabs = st.tabs(LIBRARY_CATEGORIES)

    for tab, category in zip(tabs, LIBRARY_CATEGORIES):
        with tab:
            with st.expander("➕ Add Document"):
                title_input = st.text_input("Title", key=f"library_title_{category}")
                uploaded_file = st.file_uploader("Choose a file", key=f"library_upload_{category}")
                if st.button("Save", key=f"library_save_{category}"):
                    if uploaded_file is not None:
                        add_library_document(category, title_input, uploaded_file)
                        st.toast("Document saved.")
                        st.rerun()
                    else:
                        st.warning("Choose a file first.")

            category_records = [(i, r) for i, r in enumerate(records) if r.get("category") == category]

            if not category_records:
                st.info(f"No {category.lower()} saved yet.")
                continue

            color = LIBRARY_CATEGORY_COLORS.get(category, "#3a3a3a")
            cols_per_row = 4

            for row_start in range(0, len(category_records), cols_per_row):
                chunk = category_records[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, (idx, record) in zip(cols, chunk):
                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                background:{color};
                                border:2px solid #808080;
                                border-radius:4px;
                                height:90px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                margin-bottom:6px;
                                font-size:2rem;
                            ">📄</div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.caption(record.get("title", "Untitled"))

                        stored_path = PROJECT_ROOT / record.get("stored_path", "")
                        if stored_path.exists():
                            with open(stored_path, "rb") as file:
                                st.download_button(
                                    "Open",
                                    data=file.read(),
                                    file_name=record.get("filename", "document"),
                                    key=f"library_open_{idx}",
                                    use_container_width=True,
                                )

                        if st.button("🗑", key=f"library_remove_{idx}", use_container_width=True):
                            remove_library_document(idx)
                            st.rerun()
