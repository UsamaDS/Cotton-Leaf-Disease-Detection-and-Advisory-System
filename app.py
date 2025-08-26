"""
app.py
Streamlit UI that ties everything together.

Features:
- Image upload
- YOLO detection (Ultralytics)
- Severity calculation (bbox union or GrabCut refine)
- Advisory generation (LLM via OpenAI or rule-based fallback)
- Feedback collection (SQLite)
"""

import streamlit as st
from PIL import Image
import numpy as np
import io
import sqlite3
import time
import os

from utils import load_yolo_model, detect_image, compute_severity_from_boxes, draw_overlay
from advisory import llm_advisory, rule_based_advisory

# -----------------------
# Configuration
# -----------------------
MODEL_PATH = "best.pt"  # <<-- set to your weights
METHOD = "grabcut"   # choose "bbox" (fast) or "grabcut" (more accurate)

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def get_model(path):
    print("Load Model")
    return load_yolo_model(path)

model = get_model(MODEL_PATH)

# -----------------------
# Feedback DB functions
# -----------------------
DB_PATH = "outputs/feedback.db"
os.makedirs("outputs", exist_ok=True)
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp INTEGER,
      image_name TEXT,
      detections TEXT,
      overall_ratio REAL,
      overall_label TEXT,
      advisory TEXT,
      rating INTEGER,  -- 1 thumbs up, 0 thumbs down
      comment TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_feedback(image_name, detections, overall_ratio, overall_label, advisory, rating, comment):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO feedback (timestamp, image_name, detections, overall_ratio, overall_label, advisory, rating, comment)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (int(time.time()), image_name, str(detections), overall_ratio, overall_label, advisory, rating, comment))
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Streamlit layout
# -----------------------
st.set_page_config(layout="centered", page_title="Cotton Disease Detector & Advisory")
st.title("🌿 Cotton Disease Detector & Advisory (YOLO + LLM)")
st.markdown("Upload an image of a cotton leaf. The system will highlight detected disease areas, estimate severity, and provide a short advisory. Use the feedback buttons to help us improve.")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

use_llm = st.checkbox("Use LLM for advisory (requires OPENAI/GROK API_KEY)", value=False)
confidence_slider = st.slider("Confidence threshold for detections", 0.05, 0.95, 0.25, 0.05)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)[:,:,::-1]  # PIL->OpenCV BGR for our utils
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Run detection
    detections = detect_image(model, img_np, conf=confidence_slider)
    st.write(f"Detected {len(detections)} object(s).")
    # Optionally show raw detection table
    if len(detections) > 0:
        df_rows = []
        for d in detections:
            df_rows.append({"name": d["name"], "conf": f"{d['conf']:.2f}", "box": d["xyxy"]})
        st.table(df_rows)

    # Compute severity
    severity = compute_severity_from_boxes(img_np, detections, method=METHOD)
    st.write("Overall severity:", severity.get("overall_label"), f"({severity.get('overall_ratio'):.2%})")

    # Advisory generation
    if use_llm:
        advisory_text = llm_advisory(detections, severity, location="Pakistan")
    else:
        advisory_text = rule_based_advisory(detections, severity)



    #=====================================
    # Generate advisory
    # advisory = generate_advisory(detected_classes, use_llm)

    st.subheader("📋 Advisory (English)")
    if isinstance(advisory_text, dict):
        st.write(advisory_text["en"])  # or advisory_text.get("en", "")
    st.subheader("📋 مشورہ (اردو)")
    if isinstance(advisory_text, dict):
        st.markdown(f"<div dir='rtl' style='font-family:Noto Nastaliq Urdu, Arial;'>{advisory_text['ur']}</div>", unsafe_allow_html=True)


    # Visualization overlay
    overlay = draw_overlay(img_np, detections, severity)
    overlay_rgb = overlay[:,:,::-1]  # convert back to RGB for PIL
    st.image(Image.fromarray(overlay_rgb), caption="Detections & Severity Overlay", use_container_width=True)

    # Feedback UI
    st.subheader("✍️ Feedback")
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        if st.button("👍 Helpful"):
            log_feedback(uploaded.name, [d["name"] for d in detections], severity.get("overall_ratio"), severity.get("overall_label"), advisory_text, 1, "")
            st.success("Thanks — your thumbs up recorded.")
    with col2:
        if st.button("👎 Not helpful"):
            comment = st.text_input("Tell us what was wrong (optional)")
            # If user clicks Not helpful again (or we could show a modal), read comment
            # For simplicity, record immediate with possibly empty comment
            log_feedback(uploaded.name, [d["name"] for d in detections], severity.get("overall_ratio"), severity.get("overall_label"), advisory_text, 0, comment or "")
            st.warning("Thanks — your feedback recorded.")
    with col3:
        comment = st.text_input("Other comments (optional)", "")
        if st.button("Submit comment"):
            log_feedback(uploaded.name, [d["name"] for d in detections], severity.get("overall_ratio"), severity.get("overall_label"), advisory_text, None, comment or "")
            st.success("Comment recorded.")

    st.markdown("---")
    # st.info("Tip: For more accurate severity estimates, enable 'grabcut' method in app.py (default) or provide leaf masks.")

st.markdown("App developed to help farmers to identify the diseases affecting their cotton crops and receive timely advisories.")
st.markdown("Designed By: Engr. Usama Shafique")