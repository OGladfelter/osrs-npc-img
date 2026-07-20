"""Streamlit app to review CLIP-flagged non-creature candidates one at a time.

Run from the python/ directory:
    streamlit run review_app.py

Reads:  ../data/flagged_candidates.csv   (produced by flag_junk_images.ipynb)
Writes: ../data/review_progress.csv      (full decision log, keep + remove — lets you resume)
        ../data/flagged_junk_images.csv  (remove decisions only — feeds delete_flagged_images.ipynb)
"""

import os

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "..", "data")
CANDIDATES_CSV = os.path.join(DATA_DIR, "flagged_candidates.csv")
PROGRESS_CSV = os.path.join(DATA_DIR, "review_progress.csv")
DELETE_CSV = os.path.join(DATA_DIR, "flagged_junk_images.csv")
FOLDER = {"chathead": "chatheads", "body": "bodies"}

st.set_page_config(page_title="Junk Image Review", page_icon="🧹", layout="centered")


@st.cache_data
def load_candidates():
    return pd.read_csv(CANDIDATES_CSV)


def load_progress():
    if os.path.exists(PROGRESS_CSV):
        return pd.read_csv(PROGRESS_CSV)
    return pd.DataFrame(columns=["id", "type", "Name", "confidence", "decision"])


def save_progress(progress):
    progress.to_csv(PROGRESS_CSV, index=False)
    removals = progress[progress["decision"] == "remove"][["id", "type", "Name", "confidence"]]
    removals.to_csv(DELETE_CSV, index=False)


def record_decision(row, decision):
    progress = load_progress()
    records = [
        r for r in progress.to_dict("records")
        if not (r["id"] == row["id"] and r["type"] == row["type"])
    ]
    records.append({**row[["id", "type", "Name", "confidence"]].to_dict(), "decision": decision})
    save_progress(pd.DataFrame(records))


def undo_last():
    progress = load_progress()
    if len(progress) == 0:
        return
    progress = progress.iloc[:-1]
    save_progress(progress)


if not os.path.exists(CANDIDATES_CSV):
    st.error(f"No candidates file found at `{CANDIDATES_CSV}`. Run `flag_junk_images.ipynb` first.")
    st.stop()

candidates = load_candidates()
progress = load_progress()
reviewed_keys = set(zip(progress["id"], progress["type"]))
remaining = candidates[
    ~candidates.apply(lambda r: (r["id"], r["type"]) in reviewed_keys, axis=1)
].reset_index(drop=True)

st.title("🧹 Junk Image Review")

n_removed = (progress["decision"] == "remove").sum() if len(progress) else 0
n_kept = (progress["decision"] == "keep").sum() if len(progress) else 0
st.progress(len(progress) / len(candidates) if len(candidates) else 1.0)
st.caption(
    f"{len(progress)} / {len(candidates)} reviewed &nbsp;·&nbsp; "
    f"🗑 {n_removed} to remove &nbsp;·&nbsp; ✅ {n_kept} kept",
    unsafe_allow_html=True,
)

if st.button("↩ Undo last decision", disabled=len(progress) == 0):
    undo_last()
    st.rerun()

st.divider()

if len(remaining) == 0:
    st.success("All candidates reviewed! `flagged_junk_images.csv` is ready for delete_flagged_images.ipynb.")
    st.stop()

row = remaining.iloc[0]
folder = FOLDER[row["type"]]
img_path = os.path.join(DATA_DIR, folder, f"{int(row['id'])}.png")

col1, col2 = st.columns([1, 1])
with col1:
    if os.path.exists(img_path):
        st.image(img_path, width=320)
    else:
        st.warning("Image file not found on disk")
with col2:
    st.markdown(f"### {row['Name']}")
    st.write(f"**ID:** {row['id']}")
    st.write(f"**Type:** {row['type']}")
    st.write(f"**CLIP non-creature confidence:** {row['confidence']:.3f}")

b1, b2 = st.columns(2)
if b1.button("🗑 Remove from dataset", use_container_width=True, type="primary"):
    record_decision(row, "remove")
    st.rerun()
if b2.button("✅ Keep in dataset", use_container_width=True):
    record_decision(row, "keep")
    st.rerun()
