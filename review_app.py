from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from cohortcoder.review_service import ReviewQueue

st.set_page_config(page_title="MedCode Review", layout="wide")
st.title("MedCode — grounded Top-K clinical coding review")

DB = os.environ.get("MEDCODE_REVIEW_DB", "review_queue.sqlite3")
queue = ReviewQueue(DB)
items = queue.pending(limit=200)

if not items:
    st.success("No pending review items.")
    st.json(queue.summary())
    st.stop()

index = st.number_input("Pending item", min_value=1, max_value=len(items), value=1, step=1) - 1
item = items[int(index)]
st.subheader(f"Record {item.get('record_id')} — {item.get('route')}")
st.write(f"Confidence: {item.get('confidence')}")
st.text_area("Original clinical text", value=str(item.get("text", "")), height=180, disabled=True)
st.write("**Task mention:**", item.get("mention", ""))

options = item.get("candidate_options", [])
for option in options:
    with st.expander(f"#{option.get('rank')}  {option.get('code')} — {option.get('term')}  | score={option.get('model_score')}", expanded=option.get("rank") == 1):
        st.write("**Evidence**")
        quotes = option.get("evidence_quotes", [])
        if quotes:
            for q in quotes:
                st.code(q)
        else:
            st.warning("No exact grounded source span was found for this option.")
        st.write("**Why this candidate?**")
        st.write(option.get("rationale", ""))
        st.write("**Terminology support**")
        st.json(option.get("terminology_support", {}))
        st.write("**Historical expert-coded support**")
        st.json(option.get("historical_support", []))

codes = [str(o.get("code", "")) for o in options]
selected = st.selectbox("Final action", ["ACCEPT_TOP1", "SELECT_ALTERNATIVE", "RECODE_OUTSIDE_TOPK", "ESCALATE", "NO_CODE"])
selected_code = st.text_input("Selected/final code", value=codes[0] if codes and selected == "ACCEPT_TOP1" else "")
reason = st.text_area("Reviewer reason / correction note")
reviewer = st.text_input("Reviewer identifier (stored only as hash)", type="password")

if st.button("Submit review decision", type="primary"):
    reviewer_hash = hashlib.sha256(reviewer.encode()).hexdigest()[:16] if reviewer else ""
    queue.decide(str(item.get("record_id")), action=selected, selected_code=selected_code, reviewer_id_hash=reviewer_hash, reason=reason)
    st.success("Decision recorded in the audit trail. Refresh to load the next pending item.")
