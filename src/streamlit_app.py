#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import joblib
import streamlit as st

# --- Import Deep Learning Model ---
try:
    from dl_model import dl_predict
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# ---------- text cleaning ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- path helpers ----------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def default_paths():
    root = project_root()
    out = root / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
    }

def load_pipeline_or_parts(pipeline_path: Path, model_path: Path, vectorizer_path: Path):
    if pipeline_path and pipeline_path.exists():
        return joblib.load(pipeline_path), None, None
    if model_path.exists() and vectorizer_path.exists():
        clf = joblib.load(model_path)
        vec = joblib.load(vectorizer_path)
        return None, clf, vec
    return None, None, None

# ---------- streamlit app ----------
def main():
    # parse CLI overrides but give safe defaults relative to repo root
    dp = default_paths()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    ap.add_argument("--model", default=str(dp["model"]))
    ap.add_argument("--vectorizer", default=str(dp["vectorizer"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    model_path = Path(args.model).resolve()
    vectorizer_path = Path(args.vectorizer).resolve()

    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.title("📰 Fake News & Misinformation Detector")

    # --- SIDEBAR SETTINGS ---
    with st.sidebar:
        st.subheader("Settings")
        
        # 1. Model Selector (UPDATED)
        model_type = st.radio(
            "Select AI Model", 
            ["Classic ML (Logistic Regression)", "Deep Learning (DistilBERT)"]
        )

        st.divider()
        st.subheader("File Paths")
        st.code(f"pipeline:  {pipeline_path.name}\nmodel:     {model_path.name}\nvectorizer:{vectorizer_path.name}")
        st.caption(f"Files found: {pipeline_path.exists() or model_path.exists()}")

    # --- MAIN APP LOGIC ---
    
    # 1. Load Classic Model (Always needed for the 'Classic' option)
    pipe, clf, vec = load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path)
    
    # 2. Check if Classic Model is missing
    classic_model_missing = (pipe is None and (clf is None or vec is None))

    # 3. Text Input
    txt = st.text_area("Paste headline or article text:", height=200)
    
    # 4. Threshold Slider (Only relevant for Classic ML usually)
    if model_type == "Classic ML (Logistic Regression)":
        threshold = st.slider("FAKE decision threshold", 0.05, 0.95, 0.50, 0.01)

    if st.button("Analyze") and txt.strip():
        
        # --- OPTION A: CLASSIC ML ---
        if model_type == "Classic ML (Logistic Regression)":
            if classic_model_missing:
                st.error("Classic model files not found in 'outputs/'. Please train the model first.")
            else:
                s = clean_text(txt)
                if pipe is not None:
                    prob_fake = float(pipe.predict_proba([s])[0, 1])
                else:
                    X = vec.transform([s])
                    prob_fake = float(clf.predict_proba(X)[0, 1])

                label = "FAKE" if prob_fake >= threshold else "REAL"
                
                # Display Results
                st.metric("Prediction", label, delta=f"{prob_fake:.1%} Fake Probability")
                st.progress(prob_fake, text=f"Fake Probability (Threshold: {threshold})")

        # --- OPTION B: DEEP LEARNING (UPDATED LOGIC) ---
        elif model_type == "Deep Learning (DistilBERT)":
            if not DL_AVAILABLE:
                st.error("Deep Learning module not found. Check if 'dl_model.py' is in the 'src' folder.")
            else:
                with st.spinner("Analyzing with DistilBERT..."):
                    # Call the function from dl_model.py
                    label, score = dl_predict(txt)
                
                st.metric("Prediction", label, delta=f"{score:.1%} Confidence")

if __name__ == "__main__":
    main()